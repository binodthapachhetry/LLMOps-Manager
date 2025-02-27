"""
Unit tests for the cloud provider implementations.
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.core.providers import AWSProvider, GCPProvider, AzureProvider


def test_aws_provider_initialization():
    """Test AWS provider initialization."""
    config = {
        "region": "us-west-2",
        "role_arn": "arn:aws:iam::123456789012:role/TestRole",
    }
    
    provider = AWSProvider(config)
    
    assert provider.config == config


def test_aws_provider_validation_error():
    """Test AWS provider validation error."""
    # Missing required configuration
    config = {"region": "us-west-2"}
    
    with pytest.raises(ValueError, match="Missing required AWS configuration"):
        AWSProvider(config)


def test_aws_provider_deploy_model():
    """Test AWS provider deploy_model method."""
    config = {
        "region": "us-west-2",
        "role_arn": "arn:aws:iam::123456789012:role/TestRole",
    }
    
    with patch("boto3.client") as mock_boto3:
        mock_sagemaker = MagicMock()
        mock_boto3.return_value = mock_sagemaker
        
        provider = AWSProvider(config)
        deployment_id = provider.deploy_model(
            model_path="models/test-model/v1",
            model_name="test-model",
            model_version="v1",
            instance_type="ml.g4dn.xlarge",
            instance_count=1,
        )
        
        assert "aws-test-model-v1" in deployment_id
        mock_boto3.assert_called_once_with("sagemaker", region_name="us-west-2")


def test_gcp_provider_initialization():
    """Test GCP provider initialization."""
    config = {
        "project_id": "test-project",
        "region": "us-central1",
    }
    
    provider = GCPProvider(config)
    
    assert provider.config == config


def test_gcp_provider_validation_error():
    """Test GCP provider validation error."""
    # Missing required configuration
    config = {"project_id": "test-project"}
    
    with pytest.raises(ValueError, match="Missing required GCP configuration"):
        GCPProvider(config)


def test_azure_provider_initialization():
    """Test Azure provider initialization."""
    config = {
        "subscription_id": "test-subscription",
        "resource_group": "test-resource-group",
        "workspace_name": "test-workspace",
    }
    
    provider = AzureProvider(config)
    
    assert provider.config == config


def test_azure_provider_validation_error():
    """Test Azure provider validation error."""
    # Missing required configuration
    config = {
        "subscription_id": "test-subscription",
        "resource_group": "test-resource-group",
    }
    
    with pytest.raises(ValueError, match="Missing required Azure configuration"):
        AzureProvider(config)


def test_aws_provider_store_artifact():
    """Test AWS provider store_artifact method."""
    config = {
        "region": "us-west-2",
        "role_arn": "arn:aws:iam::123456789012:role/TestRole",
    }
    
    with patch("boto3.client") as mock_boto3:
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        
        provider = AWSProvider(config)
        artifact_uri = provider.store_artifact(
            artifact_path="test_artifact.bin",
            destination_path="models/test-model/v1/artifact.bin",
        )
        
        assert artifact_uri.startswith("s3://")
        mock_boto3.assert_called_once_with("s3", region_name="us-west-2")


def test_aws_provider_retrieve_artifact():
    """Test AWS provider retrieve_artifact method."""
    config = {
        "region": "us-west-2",
        "role_arn": "arn:aws:iam::123456789012:role/TestRole",
    }
    
    with patch("boto3.client") as mock_boto3:
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        
        provider = AWSProvider(config)
        
        # Create a temporary destination path
        destination_path = Path("test_destination")
        
        result_path = provider.retrieve_artifact(
            artifact_uri="s3://test-bucket/models/test-model/v1/artifact.bin",
            destination_path=destination_path,
        )
        
        assert result_path == destination_path
        mock_boto3.assert_called_once_with("s3", region_name="us-west-2")
