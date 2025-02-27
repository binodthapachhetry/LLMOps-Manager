"""
Unit tests for the LLMOps Manager core module.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.core.manager import LLMOpsManager, ProviderType


def test_llmops_manager_initialization(test_config_file, mock_aws_provider):
    """Test LLMOps Manager initialization."""
    with patch("src.core.manager.AWSProvider", return_value=mock_aws_provider):
        manager = LLMOpsManager(
            config_path=test_config_file,
            provider_type=ProviderType.AWS,
            enable_monitoring=True,
            enable_autoscaling=True,
        )
        
        assert manager.provider == mock_aws_provider
        assert manager.provider_type == ProviderType.AWS
        assert manager.deployment is not None
        assert manager.monitoring is not None
        assert manager.autoscaler is not None
        assert manager.version_manager is not None


def test_deploy_model(mock_llmops_manager):
    """Test deploying a model."""
    deployment_id = mock_llmops_manager.deploy_model(
        model_id="test-model",
        model_version="v1",
        canary_percentage=10.0,
        deployment_config={"instance_type": "ml.g4dn.2xlarge"},
    )
    
    assert deployment_id == "test-deployment-id"
    mock_llmops_manager.version_manager.register_deployment.assert_called_once_with(
        "test-model", "v1"
    )
    mock_llmops_manager.deployment.deploy.assert_called_once()
    mock_llmops_manager.monitoring.register_deployment.assert_called_once()
    mock_llmops_manager.autoscaler.register_deployment.assert_called_once()


def test_promote_canary(mock_llmops_manager):
    """Test promoting a canary deployment."""
    mock_llmops_manager.promote_canary(
        deployment_id="test-deployment-id",
        percentage=50.0,
    )
    
    mock_llmops_manager.deployment.update_canary.assert_called_once_with(
        "test-deployment-id", 50.0
    )


def test_rollback_deployment(mock_llmops_manager):
    """Test rolling back a deployment."""
    mock_llmops_manager.rollback_deployment("test-deployment-id")
    
    mock_llmops_manager.deployment.rollback.assert_called_once_with("test-deployment-id")
    mock_llmops_manager.monitoring.update_deployment.assert_called_once()
    mock_llmops_manager.autoscaler.update_deployment.assert_called_once()


def test_get_metrics(mock_llmops_manager):
    """Test getting metrics."""
    metrics = mock_llmops_manager.get_metrics(
        deployment_id="test-deployment-id",
        metric_names=["latency", "throughput"],
        start_time="2023-01-01T00:00:00",
        end_time="2023-01-02T00:00:00",
    )
    
    assert metrics == {"latency": [], "throughput": []}
    mock_llmops_manager.monitoring.get_metrics.assert_called_once()


def test_update_scaling_policy(mock_llmops_manager):
    """Test updating scaling policy."""
    mock_llmops_manager.update_scaling_policy(
        deployment_id="test-deployment-id",
        min_replicas=2,
        max_replicas=10,
        target_gpu_utilization=75.0,
    )
    
    mock_llmops_manager.autoscaler.update_policy.assert_called_once_with(
        deployment_id="test-deployment-id",
        min_replicas=2,
        max_replicas=10,
        target_gpu_utilization=75.0,
        max_queue_length=None,
    )


def test_register_model(mock_llmops_manager, test_model_path):
    """Test registering a model."""
    model_uri = mock_llmops_manager.register_model(
        model_id="test-model",
        model_version="v1",
        model_path=test_model_path,
        metadata={"framework": "pytorch"},
    )
    
    assert model_uri == "s3://test-bucket/models/test-model/v1"
    mock_llmops_manager.version_manager.register_model.assert_called_once_with(
        model_id="test-model",
        model_version="v1",
        model_path=test_model_path,
        metadata={"framework": "pytorch"},
    )


def test_register_prompt(mock_llmops_manager):
    """Test registering a prompt."""
    prompt_uri = mock_llmops_manager.register_prompt(
        prompt_id="test-prompt",
        prompt_version="v1",
        prompt_text="This is a test prompt.",
        metadata={"language": "en"},
    )
    
    assert prompt_uri == "s3://test-bucket/prompts/test-prompt/v1"
    mock_llmops_manager.version_manager.register_prompt.assert_called_once_with(
        prompt_id="test-prompt",
        prompt_version="v1",
        prompt_text="This is a test prompt.",
        metadata={"language": "en"},
    )


def test_provider_initialization_error(test_config_file):
    """Test error handling for unsupported provider type."""
    with pytest.raises(ValueError, match="Unsupported provider type"):
        with patch("src.core.manager.ProviderType") as mock_provider_type:
            mock_provider_type.AWS = ProviderType.AWS
            mock_provider_type.GCP = ProviderType.GCP
            mock_provider_type.AZURE = ProviderType.AZURE
            
            # Create an invalid provider type
            invalid_type = MagicMock()
            invalid_type.value = "invalid"
            
            LLMOpsManager(
                config_path=test_config_file,
                provider_type=invalid_type,
            )


def test_monitoring_disabled(test_config_file, mock_aws_provider):
    """Test LLMOps Manager with monitoring disabled."""
    with patch("src.core.manager.AWSProvider", return_value=mock_aws_provider):
        manager = LLMOpsManager(
            config_path=test_config_file,
            provider_type=ProviderType.AWS,
            enable_monitoring=False,
            enable_autoscaling=True,
        )
        
        assert manager.monitoring is None
        
        # Should raise an error when trying to get metrics
        with pytest.raises(RuntimeError, match="Monitoring is not enabled"):
            manager.get_metrics("test-deployment-id")


def test_autoscaling_disabled(test_config_file, mock_aws_provider):
    """Test LLMOps Manager with autoscaling disabled."""
    with patch("src.core.manager.AWSProvider", return_value=mock_aws_provider):
        manager = LLMOpsManager(
            config_path=test_config_file,
            provider_type=ProviderType.AWS,
            enable_monitoring=True,
            enable_autoscaling=False,
        )
        
        assert manager.autoscaler is None
        
        # Should raise an error when trying to update scaling policy
        with pytest.raises(RuntimeError, match="Autoscaling is not enabled"):
            manager.update_scaling_policy("test-deployment-id")
