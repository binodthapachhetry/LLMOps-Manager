"""
Integration tests for the API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import json
from unittest.mock import patch

from src.api.endpoints import app


def test_health_check(test_api_client):
    """Test the health check endpoint."""
    response = test_api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "timestamp" in response.json()


def test_deploy_model(test_api_client):
    """Test deploying a model."""
    request_data = {
        "model_id": "test-model",
        "model_version": "v1",
        "canary_percentage": 10.0,
        "instance_type": "ml.g4dn.xlarge",
        "instance_count": 2,
        "auto_scaling": True,
        "environment_variables": {"KEY": "VALUE"},
    }
    
    response = test_api_client.post(
        "/models/deploy",
        json=request_data,
        headers={"X-API-Key": "test-api-key"},
    )
    
    assert response.status_code == 200
    assert response.json()["deployment_id"] == "test-deployment-id"
    assert response.json()["model_id"] == "test-model"
    assert response.json()["model_version"] == "v1"
    assert response.json()["status"] == "deploying"


def test_get_deployment(test_api_client):
    """Test getting deployment status."""
    response = test_api_client.get(
        "/deployments/test-deployment-id",
        headers={"X-API-Key": "test-api-key"},
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "deployed"


def test_update_deployment(test_api_client):
    """Test updating a deployment."""
    request_data = {
        "canary_percentage": 50.0,
        "instance_count": 3,
    }
    
    response = test_api_client.put(
        "/deployments/test-deployment-id",
        json=request_data,
        headers={"X-API-Key": "test-api-key"},
    )
    
    assert response.status_code == 200
    assert response.json()["deployment_id"] == "test-deployment-id"
    assert response.json()["status"] == "updating"


def test_delete_deployment(test_api_client):
    """Test deleting a deployment."""
    response = test_api_client.delete(
        "/deployments/test-deployment-id",
        headers={"X-API-Key": "test-api-key"},
    )
    
    assert response.status_code == 200
    assert response.json()["deployment_id"] == "test-deployment-id"
    assert response.json()["status"] == "deleted"


def test_update_scaling_policy(test_api_client):
    """Test updating scaling policy."""
    request_data = {
        "min_replicas": 2,
        "max_replicas": 8,
        "target_gpu_utilization": 75.0,
        "max_queue_length": 100,
    }
    
    response = test_api_client.put(
        "/deployments/test-deployment-id/scaling",
        json=request_data,
        headers={"X-API-Key": "test-api-key"},
    )
    
    assert response.status_code == 200
    assert response.json()["deployment_id"] == "test-deployment-id"
    assert response.json()["status"] == "policy_updated"


def test_register_model(test_api_client):
    """Test registering a model."""
    request_data = {
        "model_id": "test-model",
        "model_version": "v1",
        "model_path": "/path/to/model",
        "metadata": {"framework": "pytorch"},
    }
    
    response = test_api_client.post(
        "/models/register",
        json=request_data,
        headers={"X-API-Key": "test-api-key"},
    )
    
    assert response.status_code == 200
    assert response.json()["model_id"] == "test-model"
    assert response.json()["model_version"] == "v1"
    assert response.json()["model_uri"] == "test-model-uri"
    assert response.json()["status"] == "registered"


def test_register_prompt(test_api_client):
    """Test registering a prompt."""
    request_data = {
        "prompt_id": "test-prompt",
        "prompt_version": "v1",
        "prompt_text": "This is a test prompt.",
        "metadata": {"language": "en"},
    }
    
    response = test_api_client.post(
        "/prompts/register",
        json=request_data,
        headers={"X-API-Key": "test-api-key"},
    )
    
    assert response.status_code == 200
    assert response.json()["prompt_id"] == "test-prompt"
    assert response.json()["prompt_version"] == "v1"
    assert response.json()["prompt_uri"] == "test-prompt-uri"
    assert response.json()["status"] == "registered"


def test_list_models(test_api_client):
    """Test listing models."""
    response = test_api_client.get(
        "/models",
        headers={"X-API-Key": "test-api-key"},
    )
    
    assert response.status_code == 200
    assert "models" in response.json()
    assert "test-model" in response.json()["models"]


def test_list_model_versions(test_api_client):
    """Test listing model versions."""
    response = test_api_client.get(
        "/models/test-model/versions",
        headers={"X-API-Key": "test-api-key"},
    )
    
    assert response.status_code == 200
    assert response.json()["model_id"] == "test-model"
    assert "versions" in response.json()
    assert "v1" in response.json()["versions"]
    assert "v2" in response.json()["versions"]


def test_list_prompts(test_api_client):
    """Test listing prompts."""
    response = test_api_client.get(
        "/prompts",
        headers={"X-API-Key": "test-api-key"},
    )
    
    assert response.status_code == 200
    assert "prompts" in response.json()
    assert "test-prompt" in response.json()["prompts"]


def test_list_prompt_versions(test_api_client):
    """Test listing prompt versions."""
    response = test_api_client.get(
        "/prompts/test-prompt/versions",
        headers={"X-API-Key": "test-api-key"},
    )
    
    assert response.status_code == 200
    assert response.json()["prompt_id"] == "test-prompt"
    assert "versions" in response.json()
    assert "v1" in response.json()["versions"]


def test_get_prompt(test_api_client):
    """Test getting a prompt."""
    response = test_api_client.get(
        "/prompts/test-prompt",
        headers={"X-API-Key": "test-api-key"},
    )
    
    assert response.status_code == 200
    assert response.json()["prompt_id"] == "test-prompt"
    assert response.json()["prompt_version"] == "v1"
    assert response.json()["prompt_text"] == "This is a test prompt."
    assert "metadata" in response.json()


def test_run_inference(test_api_client):
    """Test running inference."""
    request_data = {
        "model_id": "test-model",
        "model_version": "v1",
        "inputs": {"text": "Hello, world!"},
        "parameters": {"temperature": 0.7},
    }
    
    response = test_api_client.post(
        "/inference",
        json=request_data,
        headers={"X-API-Key": "test-api-key"},
    )
    
    assert response.status_code == 200
    assert response.json()["model_id"] == "test-model"
    assert response.json()["model_version"] == "v1"
    assert "outputs" in response.json()
    assert "generated_text" in response.json()["outputs"]


def test_authentication_failure(test_api_client):
    """Test authentication failure."""
    response = test_api_client.get(
        "/models",
        headers={"X-API-Key": "invalid-api-key"},
    )
    
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["error"]


def test_rate_limiting():
    """Test rate limiting."""
    # Override the dependency to use a real rate limiter
    from src.api.endpoints import RateLimiter
    
    original_call = RateLimiter.__call__
    
    # Mock the rate limiter to always raise a rate limit exception
    async def mock_call(self, request):
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
        )
    
    # Apply the mock
    RateLimiter.__call__ = mock_call
    
    # Create a test client
    client = TestClient(app)
    
    # Test the rate limiting
    response = client.get(
        "/models",
        headers={"X-API-Key": "test-api-key"},
    )
    
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["error"]
    
    # Restore the original method
    RateLimiter.__call__ = original_call
