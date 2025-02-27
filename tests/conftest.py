"""
Pytest configuration and fixtures for testing the LLMOps Manager.
"""
import os
import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock

from src.core.manager import LLMOpsManager, ProviderType
from src.core.providers import AWSProvider, GCPProvider, AzureProvider


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return {
        "aws": {
            "region": "us-west-2",
            "role_arn": "arn:aws:iam::123456789012:role/TestRole",
        },
        "gcp": {
            "project_id": "test-project",
            "region": "us-central1",
        },
        "azure": {
            "subscription_id": "test-subscription",
            "resource_group": "test-resource-group",
            "workspace_name": "test-workspace",
        },
        "deployment": {
            "default_instance_type": "ml.g4dn.xlarge",
            "default_instance_count": 1,
            "canary_evaluation_duration": 60,
        },
        "monitoring": {
            "collectors": {
                "collection_interval_seconds": 10,
                "enabled_metrics": [
                    "latency",
                    "throughput",
                    "error_rate",
                ],
            },
            "alerts": {
                "thresholds": {
                    "latency": {
                        "warning": 500,
                        "error": 1000,
                    },
                },
            },
        },
        "scaling": {
            "check_interval_seconds": 10,
            "default_min_replicas": 1,
            "default_max_replicas": 5,
        },
        "versioning": {
            "model_registry": {
                "storage_prefix": "test-models",
                "local_cache_dir": "test_model_cache",
            },
            "prompt_registry": {
                "storage_prefix": "test-prompts",
                "local_cache_dir": "test_prompt_cache",
            },
        },
    }


@pytest.fixture
def test_config_file(test_config, tmp_path):
    """Create a temporary config file."""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)
    return config_path


@pytest.fixture
def mock_aws_provider():
    """Create a mock AWS provider."""
    provider = MagicMock(spec=AWSProvider)
    provider.deploy_model.return_value = "aws-test-deployment-id"
    provider.get_deployment_status.return_value = {"status": "InService"}
    provider.get_metrics.return_value = {"latency": [], "throughput": []}
    provider.store_artifact.return_value = "s3://test-bucket/test-path"
    provider.retrieve_artifact.return_value = Path("test_path")
    return provider


@pytest.fixture
def mock_gcp_provider():
    """Create a mock GCP provider."""
    provider = MagicMock(spec=GCPProvider)
    provider.deploy_model.return_value = "gcp-test-deployment-id"
    provider.get_deployment_status.return_value = {"status": "DEPLOYED"}
    provider.get_metrics.return_value = {"latency": [], "throughput": []}
    provider.store_artifact.return_value = "gs://test-bucket/test-path"
    provider.retrieve_artifact.return_value = Path("test_path")
    return provider


@pytest.fixture
def mock_azure_provider():
    """Create a mock Azure provider."""
    provider = MagicMock(spec=AzureProvider)
    provider.deploy_model.return_value = "azure-test-deployment-id"
    provider.get_deployment_status.return_value = {"status": "Healthy"}
    provider.get_metrics.return_value = {"latency": [], "throughput": []}
    provider.store_artifact.return_value = "azure://test-container/test-path"
    provider.retrieve_artifact.return_value = Path("test_path")
    return provider


@pytest.fixture
def mock_llmops_manager(monkeypatch, mock_aws_provider, test_config_file):
    """Create a mock LLMOps Manager with mocked components."""
    # Mock the provider initialization
    monkeypatch.setattr("src.core.manager.AWSProvider", lambda config: mock_aws_provider)
    
    # Create the manager
    manager = LLMOpsManager(
        config_path=test_config_file,
        provider_type=ProviderType.AWS,
        enable_monitoring=True,
        enable_autoscaling=True,
    )
    
    # Mock the components
    manager.deployment = MagicMock()
    manager.deployment.deploy.return_value = "test-deployment-id"
    manager.deployment.get_deployment_status.return_value = {"status": "deployed"}
    
    manager.monitoring = MagicMock()
    manager.monitoring.get_metrics.return_value = {"latency": [], "throughput": []}
    
    manager.autoscaler = MagicMock()
    
    manager.version_manager = MagicMock()
    manager.version_manager.register_model.return_value = "s3://test-bucket/models/test-model/v1"
    manager.version_manager.register_prompt.return_value = "s3://test-bucket/prompts/test-prompt/v1"
    manager.version_manager.list_models.return_value = ["test-model"]
    manager.version_manager.list_prompts.return_value = ["test-prompt"]
    
    return manager


@pytest.fixture
def test_model_path(tmp_path):
    """Create a test model path."""
    model_path = tmp_path / "test_model"
    model_path.mkdir()
    (model_path / "model.bin").write_text("test model content")
    return model_path


@pytest.fixture
def test_api_client():
    """Create a test API client."""
    from fastapi.testclient import TestClient
    from src.api.endpoints import app
    
    # Mock the get_llmops_manager dependency
    def mock_get_llmops_manager():
        manager = MagicMock()
        manager.deploy_model.return_value = "test-deployment-id"
        manager.get_metrics.return_value = {"latency": [], "throughput": []}
        manager.register_model.return_value = "test-model-uri"
        manager.register_prompt.return_value = "test-prompt-uri"
        
        # Mock deployment pipeline
        manager.deployment = MagicMock()
        manager.deployment.get_deployment_status.return_value = {"status": "deployed"}
        
        # Mock version manager
        manager.version_manager = MagicMock()
        manager.version_manager.list_models.return_value = ["test-model"]
        manager.version_manager.list_model_versions.return_value = ["v1", "v2"]
        manager.version_manager.list_prompts.return_value = ["test-prompt"]
        manager.version_manager.list_prompt_versions.return_value = ["v1"]
        manager.version_manager.get_prompt.return_value = "This is a test prompt."
        manager.version_manager.get_prompt_metadata.return_value = {
            "version": "v1",
            "metadata": {"created_at": "2023-01-01T00:00:00"}
        }
        
        return manager
    
    # Override the dependency
    app.dependency_overrides = {
        "src.api.endpoints.get_llmops_manager": mock_get_llmops_manager,
        "src.api.endpoints.get_api_key": lambda: "test-api-key",
        "src.api.endpoints.rate_limiter": lambda: None,
    }
    
    return TestClient(app)
