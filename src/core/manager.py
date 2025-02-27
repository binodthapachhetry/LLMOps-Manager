"""
Core LLMOps Manager module that orchestrates all components of the system.
"""
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import logging
from pathlib import Path

from .providers import CloudProvider, AWSProvider, GCPProvider, AzureProvider
from ..deployment.pipeline import DeploymentPipeline
from ..monitoring.metrics import MonitoringSystem
from ..scaling.autoscaler import AutoScaler
from ..versioning.artifacts import VersionManager

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported cloud provider types"""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class LLMOpsManager:
    """
    Central orchestrator for LLM operations across multiple cloud providers.
    
    This class serves as the main entry point for the LLMOps system, coordinating
    deployment, monitoring, scaling, and versioning of LLM models.
    """
    
    def __init__(
        self,
        config_path: Union[str, Path],
        provider_type: ProviderType = ProviderType.AWS,
        enable_monitoring: bool = True,
        enable_autoscaling: bool = True,
    ):
        """
        Initialize the LLMOps Manager.
        
        Args:
            config_path: Path to configuration file
            provider_type: Cloud provider to use
            enable_monitoring: Whether to enable monitoring
            enable_autoscaling: Whether to enable autoscaling
        """
        self.config_path = Path(config_path)
        self.provider_type = provider_type
        self._load_config()
        
        # Initialize cloud provider
        self.provider = self._initialize_provider()
        
        # Initialize components
        self.deployment = DeploymentPipeline(self.provider, self.config)
        self.monitoring = MonitoringSystem(self.provider, self.config) if enable_monitoring else None
        self.autoscaler = AutoScaler(self.provider, self.config) if enable_autoscaling else None
        self.version_manager = VersionManager(self.provider, self.config)
        
        logger.info(f"LLMOps Manager initialized with {provider_type.value} provider")
    
    def _load_config(self) -> None:
        """Load configuration from file"""
        import yaml
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        logger.debug(f"Loaded configuration from {self.config_path}")
    
    def _initialize_provider(self) -> CloudProvider:
        """Initialize the appropriate cloud provider based on configuration"""
        if self.provider_type == ProviderType.AWS:
            return AWSProvider(self.config.get("aws", {}))
        elif self.provider_type == ProviderType.GCP:
            return GCPProvider(self.config.get("gcp", {}))
        elif self.provider_type == ProviderType.AZURE:
            return AzureProvider(self.config.get("azure", {}))
        else:
            raise ValueError(f"Unsupported provider type: {self.provider_type}")
    
    def deploy_model(
        self,
        model_id: str,
        model_version: str,
        canary_percentage: Optional[float] = None,
        deployment_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Deploy a model with optional canary deployment.
        
        Args:
            model_id: Identifier for the model
            model_version: Version of the model to deploy
            canary_percentage: Percentage of traffic to route to new version (0-100)
            deployment_config: Additional deployment configuration
            
        Returns:
            Deployment ID
        """
        logger.info(f"Deploying model {model_id} version {model_version}")
        
        # Register deployment with version manager
        self.version_manager.register_deployment(model_id, model_version)
        
        # Execute deployment
        deployment_id = self.deployment.deploy(
            model_id=model_id,
            model_version=model_version,
            canary_percentage=canary_percentage,
            config=deployment_config or {},
        )
        
        # Set up monitoring for the deployment
        if self.monitoring:
            self.monitoring.register_deployment(deployment_id, model_id, model_version)
        
        # Configure autoscaling for the deployment
        if self.autoscaler:
            self.autoscaler.register_deployment(deployment_id)
        
        return deployment_id
    
    def promote_canary(self, deployment_id: str, percentage: float = 100.0) -> None:
        """
        Promote a canary deployment to receive more traffic.
        
        Args:
            deployment_id: ID of the deployment
            percentage: New percentage of traffic to route to the canary (0-100)
        """
        logger.info(f"Promoting canary deployment {deployment_id} to {percentage}%")
        self.deployment.update_canary(deployment_id, percentage)
    
    def rollback_deployment(self, deployment_id: str) -> None:
        """
        Rollback a deployment to the previous stable version.
        
        Args:
            deployment_id: ID of the deployment to rollback
        """
        logger.info(f"Rolling back deployment {deployment_id}")
        self.deployment.rollback(deployment_id)
        
        # Update monitoring and scaling
        if self.monitoring:
            self.monitoring.update_deployment(deployment_id)
        
        if self.autoscaler:
            self.autoscaler.update_deployment(deployment_id)
    
    def get_metrics(
        self, 
        deployment_id: str, 
        metric_names: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get metrics for a specific deployment.
        
        Args:
            deployment_id: ID of the deployment
            metric_names: List of metrics to retrieve (None for all)
            start_time: Start time for metrics query (ISO format)
            end_time: End time for metrics query (ISO format)
            
        Returns:
            Dictionary of metrics data
        """
        if not self.monitoring:
            raise RuntimeError("Monitoring is not enabled")
        
        return self.monitoring.get_metrics(
            deployment_id=deployment_id,
            metric_names=metric_names,
            start_time=start_time,
            end_time=end_time,
        )
    
    def update_scaling_policy(
        self, 
        deployment_id: str, 
        min_replicas: Optional[int] = None,
        max_replicas: Optional[int] = None,
        target_gpu_utilization: Optional[float] = None,
        max_queue_length: Optional[int] = None,
    ) -> None:
        """
        Update the autoscaling policy for a deployment.
        
        Args:
            deployment_id: ID of the deployment
            min_replicas: Minimum number of replicas
            max_replicas: Maximum number of replicas
            target_gpu_utilization: Target GPU utilization percentage (0-100)
            max_queue_length: Maximum queue length before scaling up
        """
        if not self.autoscaler:
            raise RuntimeError("Autoscaling is not enabled")
        
        self.autoscaler.update_policy(
            deployment_id=deployment_id,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            target_gpu_utilization=target_gpu_utilization,
            max_queue_length=max_queue_length,
        )
        
        logger.info(f"Updated scaling policy for deployment {deployment_id}")
    
    def register_model(
        self,
        model_id: str,
        model_version: str,
        model_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a model with the version manager.
        
        Args:
            model_id: Identifier for the model
            model_version: Version of the model
            model_path: Path to model artifacts
            metadata: Additional metadata for the model
            
        Returns:
            Model registry ID
        """
        return self.version_manager.register_model(
            model_id=model_id,
            model_version=model_version,
            model_path=model_path,
            metadata=metadata or {},
        )
    
    def register_prompt(
        self,
        prompt_id: str,
        prompt_version: str,
        prompt_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a prompt with the version manager.
        
        Args:
            prompt_id: Identifier for the prompt
            prompt_version: Version of the prompt
            prompt_text: The prompt text
            metadata: Additional metadata for the prompt
            
        Returns:
            Prompt registry ID
        """
        return self.version_manager.register_prompt(
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_text=prompt_text,
            metadata=metadata or {},
        )
