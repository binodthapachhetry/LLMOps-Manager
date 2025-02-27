"""
End-to-end tests for the LLMOps Manager.
"""
import pytest
import os
import yaml
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.manager import LLMOpsManager, ProviderType


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config_file(temp_dir):
    """Create a test configuration file."""
    config = {
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
        },
        "monitoring": {
            "collectors": {
                "collection_interval_seconds": 60,
                "enabled_metrics": ["latency", "throughput", "error_rate"],
            },
        },
        "scaling": {
            "check_interval_seconds": 60,
            "default_min_replicas": 1,
            "default_max_replicas": 5,
        },
        "versioning": {
            "model_registry": {
                "storage_prefix": "models",
                "local_cache_dir": os.path.join(temp_dir, "model_cache"),
            },
            "prompt_registry": {
                "storage_prefix": "prompts",
                "local_cache_dir": os.path.join(temp_dir, "prompt_cache"),
            },
        },
    }
    
    config_path = os.path.join(temp_dir, "test_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path


@pytest.fixture
def test_model_path(temp_dir):
    """Create a test model path."""
    model_dir = os.path.join(temp_dir, "test_model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a dummy model file
    with open(os.path.join(model_dir, "model.bin"), "w") as f:
        f.write("dummy model content")
    
    return model_dir


def test_end_to_end_aws(test_config_file, test_model_path, temp_dir):
    """Test end-to-end workflow with AWS provider."""
    # Mock AWS provider and its methods
    with patch("src.core.providers.AWSProvider") as mock_aws_provider_class:
        mock_aws_provider = MagicMock()
        mock_aws_provider.deploy_model.return_value = "aws-test-deployment-id"
        mock_aws_provider.get_deployment_status.return_value = {"status": "InService"}
        mock_aws_provider.get_metrics.return_value = {"latency": [], "throughput": []}
        mock_aws_provider.store_artifact.return_value = "s3://test-bucket/models/test-model/v1"
        mock_aws_provider.retrieve_artifact.return_value = Path(test_model_path)
        
        mock_aws_provider_class.return_value = mock_aws_provider
        
        # Initialize LLMOps Manager
        manager = LLMOpsManager(
            config_path=test_config_file,
            provider_type=ProviderType.AWS,
            enable_monitoring=True,
            enable_autoscaling=True,
        )
        
        # 1. Register a model
        model_uri = manager.register_model(
            model_id="test-model",
            model_version="v1",
            model_path=test_model_path,
            metadata={"framework": "pytorch"},
        )
        
        assert model_uri is not None
        mock_aws_provider.store_artifact.assert_called_once()
        
        # 2. Register a prompt
        prompt_uri = manager.register_prompt(
            prompt_id="test-prompt",
            prompt_version="v1",
            prompt_text="This is a test prompt.",
            metadata={"language": "en"},
        )
        
        assert prompt_uri is not None
        
        # 3. Deploy the model
        deployment_id = manager.deploy_model(
            model_id="test-model",
            model_version="v1",
            canary_percentage=10.0,
            deployment_config={
                "instance_type": "ml.g4dn.xlarge",
                "instance_count": 2,
            },
        )
        
        assert deployment_id is not None
        mock_aws_provider.deploy_model.assert_called_once()
        
        # 4. Get deployment status
        status = manager.deployment.get_deployment_status(deployment_id)
        assert status is not None
        
        # 5. Update scaling policy
        manager.update_scaling_policy(
            deployment_id=deployment_id,
            min_replicas=2,
            max_replicas=8,
            target_gpu_utilization=75.0,
        )
        
        # 6. Get metrics
        metrics = manager.get_metrics(
            deployment_id=deployment_id,
            metric_names=["latency", "throughput"],
        )
        
        assert metrics is not None
        mock_aws_provider.get_metrics.assert_called_once()
        
        # 7. Promote canary
        manager.promote_canary(
            deployment_id=deployment_id,
            percentage=50.0,
        )
        
        # 8. Rollback deployment
        manager.rollback_deployment(deployment_id)
        
        # Verify that all components were used
        assert manager.provider == mock_aws_provider
        assert manager.deployment is not None
        assert manager.monitoring is not None
        assert manager.autoscaler is not None
        assert manager.version_manager is not None


def test_end_to_end_multi_provider(test_config_file, test_model_path, temp_dir):
    """Test end-to-end workflow with multiple providers."""
    # Mock providers
    with patch("src.core.providers.AWSProvider") as mock_aws_provider_class, \
         patch("src.core.providers.GCPProvider") as mock_gcp_provider_class:
        
        # AWS provider
        mock_aws_provider = MagicMock()
        mock_aws_provider.deploy_model.return_value = "aws-test-deployment-id"
        mock_aws_provider.store_artifact.return_value = "s3://test-bucket/models/test-model/v1"
        mock_aws_provider_class.return_value = mock_aws_provider
        
        # GCP provider
        mock_gcp_provider = MagicMock()
        mock_gcp_provider.deploy_model.return_value = "gcp-test-deployment-id"
        mock_gcp_provider.store_artifact.return_value = "gs://test-bucket/models/test-model/v1"
        mock_gcp_provider_class.return_value = mock_gcp_provider
        
        # 1. Initialize AWS LLMOps Manager and register a model
        aws_manager = LLMOpsManager(
            config_path=test_config_file,
            provider_type=ProviderType.AWS,
        )
        
        aws_model_uri = aws_manager.register_model(
            model_id="test-model",
            model_version="v1",
            model_path=test_model_path,
            metadata={"framework": "pytorch"},
        )
        
        assert aws_model_uri is not None
        assert aws_model_uri.startswith("s3://")
        
        # 2. Initialize GCP LLMOps Manager and register the same model
        gcp_manager = LLMOpsManager(
            config_path=test_config_file,
            provider_type=ProviderType.GCP,
        )
        
        gcp_model_uri = gcp_manager.register_model(
            model_id="test-model",
            model_version="v1",
            model_path=test_model_path,
            metadata={"framework": "pytorch"},
        )
        
        assert gcp_model_uri is not None
        assert gcp_model_uri.startswith("gs://")
        
        # 3. Deploy the model on both providers
        aws_deployment_id = aws_manager.deploy_model(
            model_id="test-model",
            model_version="v1",
        )
        
        gcp_deployment_id = gcp_manager.deploy_model(
            model_id="test-model",
            model_version="v1",
        )
        
        assert aws_deployment_id != gcp_deployment_id
        assert aws_deployment_id.startswith("aws-")
        assert gcp_deployment_id.startswith("gcp-")
        
        # Verify that the correct providers were used
        assert aws_manager.provider == mock_aws_provider
        assert gcp_manager.provider == mock_gcp_provider
