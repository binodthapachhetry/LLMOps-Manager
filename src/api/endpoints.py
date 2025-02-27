"""
FastAPI endpoints for the LLMOps Manager.
"""
from typing import Dict, List, Optional, Any, Union
import logging
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import yaml
import os
from pathlib import Path

from ..core.manager import LLMOpsManager, ProviderType

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLMOps Manager API",
    description="API for managing LLM deployments across multiple cloud providers",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)


# Rate limiting middleware
class RateLimiter:
    """Simple rate limiter for API endpoints."""
    
    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
        """
        self.requests_per_minute = requests_per_minute
        self.request_timestamps = {}
        self.window_size = 60  # 1 minute window
    
    async def __call__(self, request: Request):
        """
        Check if the request is within rate limits.
        
        Args:
            request: FastAPI request
            
        Raises:
            HTTPException: If rate limit is exceeded
        """
        client_ip = request.client.host
        current_time = time.time()
        
        # Initialize client's request history if not present
        if client_ip not in self.request_timestamps:
            self.request_timestamps[client_ip] = []
        
        # Remove timestamps outside the current window
        self.request_timestamps[client_ip] = [
            ts for ts in self.request_timestamps[client_ip]
            if current_time - ts < self.window_size
        ]
        
        # Check if rate limit is exceeded
        if len(self.request_timestamps[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
            )
        
        # Add current timestamp to history
        self.request_timestamps[client_ip].append(current_time)


# Initialize rate limiter
rate_limiter = RateLimiter()


# Authentication dependency
async def get_api_key(api_key: str = Depends(api_key_header)):
    """
    Validate API key.
    
    Args:
        api_key: API key from request header
        
    Returns:
        API key if valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    # In a real implementation, this would validate against a database
    # For simplicity, we'll use an environment variable
    valid_api_key = os.environ.get("LLMOPS_API_KEY", "test-api-key")
    
    if api_key != valid_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    return api_key


# Initialize LLMOps Manager
def get_llmops_manager():
    """
    Get or create the LLMOps Manager instance.
    
    Returns:
        LLMOps Manager instance
    """
    # In a real implementation, this would be properly initialized
    # and potentially stored in a database or cache
    config_path = os.environ.get("LLMOPS_CONFIG_PATH", "config/llmops.yaml")
    
    # Create config directory and file if they don't exist
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Create a default configuration
        default_config = {
            "aws": {
                "region": "us-west-2",
                "role_arn": "arn:aws:iam::123456789012:role/LLMOpsManagerRole",
            },
            "gcp": {
                "project_id": "llmops-project",
                "region": "us-central1",
            },
            "azure": {
                "subscription_id": "subscription-id",
                "resource_group": "llmops-resource-group",
                "workspace_name": "llmops-workspace",
            },
            "deployment": {
                "default_instance_type": "ml.g4dn.xlarge",
                "default_instance_count": 1,
                "canary_evaluation_duration": 600,
            },
            "monitoring": {
                "collectors": {
                    "collection_interval_seconds": 60,
                    "enabled_metrics": [
                        "latency",
                        "throughput",
                        "token_usage",
                        "error_rate",
                        "gpu_utilization",
                    ],
                },
                "alerts": {
                    "thresholds": {
                        "latency": {
                            "warning": 500,
                            "error": 1000,
                            "critical": 2000,
                            "direction": "above",
                        },
                        "error_rate": {
                            "warning": 0.01,
                            "error": 0.05,
                            "critical": 0.1,
                            "direction": "above",
                        },
                        "gpu_utilization": {
                            "warning": 85,
                            "error": 95,
                            "direction": "above",
                        },
                    },
                },
            },
            "scaling": {
                "check_interval_seconds": 60,
                "default_min_replicas": 1,
                "default_max_replicas": 10,
                "default_target_gpu_utilization": 70.0,
            },
            "versioning": {
                "model_registry": {
                    "storage_prefix": "models",
                    "local_cache_dir": "model_cache",
                },
                "prompt_registry": {
                    "storage_prefix": "prompts",
                    "local_cache_dir": "prompt_cache",
                },
            },
        }
        
        with open(config_path, "w") as f:
            yaml.dump(default_config, f)
    
    # Get provider type from environment
    provider_type_str = os.environ.get("LLMOPS_PROVIDER", "aws").lower()
    provider_type = ProviderType.AWS
    
    if provider_type_str == "gcp":
        provider_type = ProviderType.GCP
    elif provider_type_str == "azure":
        provider_type = ProviderType.AZURE
    
    # Create and return manager
    return LLMOpsManager(
        config_path=config_path,
        provider_type=provider_type,
        enable_monitoring=True,
        enable_autoscaling=True,
    )


# Request/response models
class DeployModelRequest(BaseModel):
    """Request model for deploying a model."""
    
    model_id: str = Field(..., description="Identifier for the model")
    model_version: str = Field(..., description="Version of the model to deploy")
    canary_percentage: Optional[float] = Field(
        None,
        description="Percentage of traffic to route to new version (0-100)",
        ge=0,
        le=100,
    )
    instance_type: Optional[str] = Field(None, description="Type of compute instance")
    instance_count: Optional[int] = Field(None, description="Number of instances")
    auto_scaling: Optional[bool] = Field(None, description="Whether to enable auto-scaling")
    environment_variables: Optional[Dict[str, str]] = Field(
        None,
        description="Environment variables for the deployment",
    )


class UpdateDeploymentRequest(BaseModel):
    """Request model for updating a deployment."""
    
    instance_count: Optional[int] = Field(None, description="New number of instances")
    canary_percentage: Optional[float] = Field(
        None,
        description="New percentage of traffic for canary",
        ge=0,
        le=100,
    )


class ScalingPolicyRequest(BaseModel):
    """Request model for updating a scaling policy."""
    
    min_replicas: Optional[int] = Field(None, description="Minimum number of replicas")
    max_replicas: Optional[int] = Field(None, description="Maximum number of replicas")
    target_gpu_utilization: Optional[float] = Field(
        None,
        description="Target GPU utilization percentage (0-100)",
        ge=0,
        le=100,
    )
    max_queue_length: Optional[int] = Field(None, description="Maximum queue length before scaling up")


class RegisterModelRequest(BaseModel):
    """Request model for registering a model."""
    
    model_id: str = Field(..., description="Identifier for the model")
    model_version: str = Field(..., description="Version of the model")
    model_path: str = Field(..., description="Path to model artifacts")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class RegisterPromptRequest(BaseModel):
    """Request model for registering a prompt."""
    
    prompt_id: str = Field(..., description="Identifier for the prompt")
    prompt_version: str = Field(..., description="Version of the prompt")
    prompt_text: str = Field(..., description="The prompt text")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class InferenceRequest(BaseModel):
    """Request model for model inference."""
    
    model_id: str = Field(..., description="Identifier for the model")
    model_version: Optional[str] = Field(None, description="Version of the model (None for latest)")
    inputs: Dict[str, Any] = Field(..., description="Input data for inference")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Inference parameters")


# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post(
    "/models/deploy",
    dependencies=[Depends(rate_limiter), Depends(get_api_key)],
)
async def deploy_model(
    request: DeployModelRequest,
    background_tasks: BackgroundTasks,
):
    """
    Deploy a model.
    
    Args:
        request: Deployment request
        background_tasks: FastAPI background tasks
        
    Returns:
        Deployment information
    """
    try:
        manager = get_llmops_manager()
        
        # Prepare deployment configuration
        deployment_config = {}
        
        if request.instance_type:
            deployment_config["instance_type"] = request.instance_type
        
        if request.instance_count:
            deployment_config["instance_count"] = request.instance_count
        
        if request.auto_scaling is not None:
            deployment_config["auto_scaling"] = request.auto_scaling
        
        if request.environment_variables:
            deployment_config["environment_variables"] = request.environment_variables
        
        # Deploy the model
        deployment_id = manager.deploy_model(
            model_id=request.model_id,
            model_version=request.model_version,
            canary_percentage=request.canary_percentage,
            deployment_config=deployment_config,
        )
        
        return {
            "deployment_id": deployment_id,
            "model_id": request.model_id,
            "model_version": request.model_version,
            "status": "deploying",
        }
    except Exception as e:
        logger.error(f"Error deploying model: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deploying model: {str(e)}",
        )


@app.get(
    "/deployments/{deployment_id}",
    dependencies=[Depends(rate_limiter), Depends(get_api_key)],
)
async def get_deployment(deployment_id: str):
    """
    Get deployment status.
    
    Args:
        deployment_id: ID of the deployment
        
    Returns:
        Deployment status
    """
    try:
        manager = get_llmops_manager()
        
        # Get deployment status from the deployment pipeline
        status = manager.deployment.get_deployment_status(deployment_id)
        
        return status
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error getting deployment status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting deployment status: {str(e)}",
        )


@app.put(
    "/deployments/{deployment_id}",
    dependencies=[Depends(rate_limiter), Depends(get_api_key)],
)
async def update_deployment(deployment_id: str, request: UpdateDeploymentRequest):
    """
    Update a deployment.
    
    Args:
        deployment_id: ID of the deployment
        request: Update request
        
    Returns:
        Updated deployment information
    """
    try:
        manager = get_llmops_manager()
        
        # Update canary percentage if provided
        if request.canary_percentage is not None:
            manager.promote_canary(
                deployment_id=deployment_id,
                percentage=request.canary_percentage,
            )
        
        # Update instance count if provided
        if request.instance_count is not None:
            # This would be handled by the auto-scaler in a real implementation
            pass
        
        return {
            "deployment_id": deployment_id,
            "status": "updating",
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error updating deployment: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating deployment: {str(e)}",
        )


@app.delete(
    "/deployments/{deployment_id}",
    dependencies=[Depends(rate_limiter), Depends(get_api_key)],
)
async def delete_deployment(deployment_id: str):
    """
    Delete a deployment.
    
    Args:
        deployment_id: ID of the deployment
        
    Returns:
        Deletion confirmation
    """
    try:
        manager = get_llmops_manager()
        
        # Roll back the deployment (which will delete it)
        manager.rollback_deployment(deployment_id)
        
        return {
            "deployment_id": deployment_id,
            "status": "deleted",
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error deleting deployment: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting deployment: {str(e)}",
        )


@app.get(
    "/deployments/{deployment_id}/metrics",
    dependencies=[Depends(rate_limiter), Depends(get_api_key)],
)
async def get_deployment_metrics(
    deployment_id: str,
    metric_names: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    """
    Get metrics for a deployment.
    
    Args:
        deployment_id: ID of the deployment
        metric_names: Comma-separated list of metrics to retrieve
        start_time: Start time for metrics query (ISO format)
        end_time: End time for metrics query (ISO format)
        
    Returns:
        Metrics data
    """
    try:
        manager = get_llmops_manager()
        
        # Parse metric names
        metric_list = None
        if metric_names:
            metric_list = [name.strip() for name in metric_names.split(",")]
        
        # Get metrics from the monitoring system
        metrics = manager.get_metrics(
            deployment_id=deployment_id,
            metric_names=metric_list,
            start_time=start_time,
            end_time=end_time,
        )
        
        return metrics
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting metrics: {str(e)}",
        )


@app.put(
    "/deployments/{deployment_id}/scaling",
    dependencies=[Depends(rate_limiter), Depends(get_api_key)],
)
async def update_scaling_policy(deployment_id: str, request: ScalingPolicyRequest):
    """
    Update the scaling policy for a deployment.
    
    Args:
        deployment_id: ID of the deployment
        request: Scaling policy request
        
    Returns:
        Updated policy confirmation
    """
    try:
        manager = get_llmops_manager()
        
        # Update the scaling policy
        manager.update_scaling_policy(
            deployment_id=deployment_id,
            min_replicas=request.min_replicas,
            max_replicas=request.max_replicas,
            target_gpu_utilization=request.target_gpu_utilization,
            max_queue_length=request.max_queue_length,
        )
        
        return {
            "deployment_id": deployment_id,
            "status": "policy_updated",
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error updating scaling policy: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating scaling policy: {str(e)}",
        )


@app.post(
    "/models/register",
    dependencies=[Depends(rate_limiter), Depends(get_api_key)],
)
async def register_model(request: RegisterModelRequest):
    """
    Register a model with the version manager.
    
    Args:
        request: Model registration request
        
    Returns:
        Registration confirmation
    """
    try:
        manager = get_llmops_manager()
        
        # Register the model
        model_uri = manager.register_model(
            model_id=request.model_id,
            model_version=request.model_version,
            model_path=request.model_path,
            metadata=request.metadata,
        )
        
        return {
            "model_id": request.model_id,
            "model_version": request.model_version,
            "model_uri": model_uri,
            "status": "registered",
        }
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error registering model: {str(e)}",
        )


@app.post(
    "/prompts/register",
    dependencies=[Depends(rate_limiter), Depends(get_api_key)],
)
async def register_prompt(request: RegisterPromptRequest):
    """
    Register a prompt with the version manager.
    
    Args:
        request: Prompt registration request
        
    Returns:
        Registration confirmation
    """
    try:
        manager = get_llmops_manager()
        
        # Register the prompt
        prompt_uri = manager.register_prompt(
            prompt_id=request.prompt_id,
            prompt_version=request.prompt_version,
            prompt_text=request.prompt_text,
            metadata=request.metadata,
        )
        
        return {
            "prompt_id": request.prompt_id,
            "prompt_version": request.prompt_version,
            "prompt_uri": prompt_uri,
            "status": "registered",
        }
    except Exception as e:
        logger.error(f"Error registering prompt: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error registering prompt: {str(e)}",
        )


@app.get(
    "/models",
    dependencies=[Depends(rate_limiter), Depends(get_api_key)],
)
async def list_models():
    """
    List all registered models.
    
    Returns:
        List of model IDs
    """
    try:
        manager = get_llmops_manager()
        
        # List models
        models = manager.version_manager.list_models()
        
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}",
        )


@app.get(
    "/models/{model_id}/versions",
    dependencies=[Depends(rate_limiter), Depends(get_api_key)],
)
async def list_model_versions(model_id: str):
    """
    List all versions of a model.
    
    Args:
        model_id: Identifier for the model
        
    Returns:
        List of model versions
    """
    try:
        manager = get_llmops_manager()
        
        # List model versions
        versions = manager.version_manager.list_model_versions(model_id)
        
        return {"model_id": model_id, "versions": versions}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error listing model versions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing model versions: {str(e)}",
        )


@app.get(
    "/prompts",
    dependencies=[Depends(rate_limiter), Depends(get_api_key)],
)
async def list_prompts():
    """
    List all registered prompts.
    
    Returns:
        List of prompt IDs
    """
    try:
        manager = get_llmops_manager()
        
        # List prompts
        prompts = manager.version_manager.list_prompts()
        
        return {"prompts": prompts}
    except Exception as e:
        logger.error(f"Error listing prompts: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing prompts: {str(e)}",
        )


@app.get(
    "/prompts/{prompt_id}/versions",
    dependencies=[Depends(rate_limiter), Depends(get_api_key)],
)
async def list_prompt_versions(prompt_id: str):
    """
    List all versions of a prompt.
    
    Args:
        prompt_id: Identifier for the prompt
        
    Returns:
        List of prompt versions
    """
    try:
        manager = get_llmops_manager()
        
        # List prompt versions
        versions = manager.version_manager.list_prompt_versions(prompt_id)
        
        return {"prompt_id": prompt_id, "versions": versions}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error listing prompt versions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing prompt versions: {str(e)}",
        )


@app.get(
    "/prompts/{prompt_id}",
    dependencies=[Depends(rate_limiter), Depends(get_api_key)],
)
async def get_prompt(prompt_id: str, version: Optional[str] = None):
    """
    Get a prompt.
    
    Args:
        prompt_id: Identifier for the prompt
        version: Version of the prompt (None for latest)
        
    Returns:
        Prompt text and metadata
    """
    try:
        manager = get_llmops_manager()
        
        # Get prompt text
        prompt_text = manager.version_manager.get_prompt(
            prompt_id=prompt_id,
            prompt_version=version,
        )
        
        # Get prompt metadata
        metadata = manager.version_manager.get_prompt_metadata(
            prompt_id=prompt_id,
            prompt_version=version,
        )
        
        return {
            "prompt_id": prompt_id,
            "prompt_version": metadata["version"],
            "prompt_text": prompt_text,
            "metadata": metadata["metadata"],
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error getting prompt: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting prompt: {str(e)}",
        )


@app.post(
    "/inference",
    dependencies=[Depends(rate_limiter), Depends(get_api_key)],
)
async def run_inference(request: InferenceRequest):
    """
    Run inference on a model.
    
    Args:
        request: Inference request
        
    Returns:
        Inference results
    """
    try:
        # In a real implementation, this would:
        # 1. Get the model from the version manager
        # 2. Run inference on the model
        # 3. Return the results
        
        # For simplicity, we'll return a mock response
        return {
            "model_id": request.model_id,
            "model_version": request.model_version or "latest",
            "outputs": {
                "generated_text": "This is a mock response from the LLM.",
                "token_count": 8,
                "processing_time_ms": 150,
            },
        }
    except Exception as e:
        logger.error(f"Error running inference: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running inference: {str(e)}",
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An unexpected error occurred"},
    )
