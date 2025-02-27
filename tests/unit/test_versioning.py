"""
Unit tests for the versioning system.
"""
import pytest
from unittest.mock import MagicMock, patch, mock_open
import json
import os
from pathlib import Path
from datetime import datetime

from src.versioning.artifacts import ArtifactMetadata, ModelRegistry, PromptRegistry, VersionManager


def test_artifact_metadata():
    """Test artifact metadata."""
    created_at = datetime.now()
    metadata = ArtifactMetadata(
        artifact_id="test-artifact",
        artifact_type="model",
        version="v1",
        created_at=created_at,
        created_by="test-user",
        storage_uri="s3://test-bucket/models/test-artifact/v1",
        metadata={"framework": "pytorch"},
    )
    
    # Check initial values
    assert metadata.artifact_id == "test-artifact"
    assert metadata.artifact_type == "model"
    assert metadata.version == "v1"
    assert metadata.created_at == created_at
    assert metadata.created_by == "test-user"
    assert metadata.storage_uri == "s3://test-bucket/models/test-artifact/v1"
    assert metadata.metadata == {"framework": "pytorch"}
    
    # Convert to dictionary
    metadata_dict = metadata.to_dict()
    assert metadata_dict["artifact_id"] == "test-artifact"
    assert metadata_dict["artifact_type"] == "model"
    assert metadata_dict["version"] == "v1"
    assert metadata_dict["created_at"] == created_at.isoformat()
    assert metadata_dict["created_by"] == "test-user"
    assert metadata_dict["storage_uri"] == "s3://test-bucket/models/test-artifact/v1"
    assert metadata_dict["metadata"] == {"framework": "pytorch"}
    
    # Create from dictionary
    new_metadata = ArtifactMetadata.from_dict(metadata_dict)
    assert new_metadata.artifact_id == "test-artifact"
    assert new_metadata.artifact_type == "model"
    assert new_metadata.version == "v1"
    assert new_metadata.created_at.isoformat() == created_at.isoformat()
    assert new_metadata.created_by == "test-user"
    assert new_metadata.storage_uri == "s3://test-bucket/models/test-artifact/v1"
    assert new_metadata.metadata == {"framework": "pytorch"}


def test_model_registry():
    """Test the model registry."""
    provider = MagicMock()
    provider.store_artifact.return_value = "s3://test-bucket/models/test-model/v1"
    provider.retrieve_artifact.return_value = Path("test_path")
    
    config = {
        "storage_prefix": "models",
        "local_cache_dir": "test_model_cache",
        "user_id": "test-user",
    }
    
    # Mock os.makedirs and open
    with patch("os.makedirs") as mock_makedirs, \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("json.load", return_value={}):
        
        registry = ModelRegistry(provider, config)
        
        # Initial state
        assert registry.provider == provider
        assert registry.config == config
        assert registry.storage_prefix == "models"
        assert registry.local_cache_dir == Path("test_model_cache")
        assert registry.models == {}
        
        # Register a model
        model_path = Path("test_model_path")
        with patch("pathlib.Path.exists", return_value=True):
            storage_uri = registry.register_model(
                model_id="test-model",
                model_version="v1",
                model_path=model_path,
                metadata={"framework": "pytorch"},
            )
        
        assert storage_uri == "s3://test-bucket/models/test-model/v1"
        assert "test-model" in registry.models
        assert "v1" in registry.models["test-model"]
        assert registry.models["test-model"]["v1"].artifact_id == "test-model"
        assert registry.models["test-model"]["v1"].version == "v1"
        assert registry.models["test-model"]["v1"].metadata == {"framework": "pytorch"}
        
        # Get model metadata
        metadata = registry.get_model_metadata("test-model", "v1")
        assert metadata["artifact_id"] == "test-model"
        assert metadata["version"] == "v1"
        assert metadata["metadata"] == {"framework": "pytorch"}
        
        # List models and versions
        models = registry.list_models()
        assert "test-model" in models
        
        versions = registry.list_model_versions("test-model")
        assert "v1" in versions
        
        # Get model
        with patch("pathlib.Path.exists", return_value=True):
            model_path = registry.get_model("test-model", "v1")
        
        assert model_path == Path("test_path")
        provider.retrieve_artifact.assert_called_once()


def test_prompt_registry():
    """Test the prompt registry."""
    provider = MagicMock()
    provider.store_artifact.return_value = "s3://test-bucket/prompts/test-prompt/v1.txt"
    provider.retrieve_artifact.return_value = Path("test_path")
    
    config = {
        "storage_prefix": "prompts",
        "local_cache_dir": "test_prompt_cache",
        "user_id": "test-user",
    }
    
    # Mock os.makedirs, open, and hashlib
    with patch("os.makedirs") as mock_makedirs, \
         patch("builtins.open", mock_open(read_data="This is a test prompt.")) as mock_file, \
         patch("json.load", return_value={}), \
         patch("hashlib.sha256") as mock_hashlib:
        
        mock_hashlib.return_value.hexdigest.return_value = "test_hash"
        
        registry = PromptRegistry(provider, config)
        
        # Initial state
        assert registry.provider == provider
        assert registry.config == config
        assert registry.storage_prefix == "prompts"
        assert registry.local_cache_dir == Path("test_prompt_cache")
        assert registry.prompts == {}
        
        # Register a prompt
        storage_uri = registry.register_prompt(
            prompt_id="test-prompt",
            prompt_version="v1",
            prompt_text="This is a test prompt.",
            metadata={"language": "en"},
        )
        
        assert storage_uri == "s3://test-bucket/prompts/test-prompt/v1.txt"
        assert "test-prompt" in registry.prompts
        assert "v1" in registry.prompts["test-prompt"]
        assert registry.prompts["test-prompt"]["v1"].artifact_id == "test-prompt"
        assert registry.prompts["test-prompt"]["v1"].version == "v1"
        assert registry.prompts["test-prompt"]["v1"].metadata["language"] == "en"
        assert registry.prompts["test-prompt"]["v1"].metadata["content_hash"] == "test_hash"
        
        # Get prompt metadata
        metadata = registry.get_prompt_metadata("test-prompt", "v1")
        assert metadata["artifact_id"] == "test-prompt"
        assert metadata["version"] == "v1"
        assert metadata["metadata"]["language"] == "en"
        
        # List prompts and versions
        prompts = registry.list_prompts()
        assert "test-prompt" in prompts
        
        versions = registry.list_prompt_versions("test-prompt")
        assert "v1" in versions
        
        # Get prompt
        prompt_text = registry.get_prompt("test-prompt", "v1")
        assert prompt_text == "This is a test prompt."
        provider.retrieve_artifact.assert_called_once()


def test_version_manager():
    """Test the version manager."""
    provider = MagicMock()
    config = {
        "model_registry": {
            "storage_prefix": "models",
            "local_cache_dir": "test_model_cache",
        },
        "prompt_registry": {
            "storage_prefix": "prompts",
            "local_cache_dir": "test_prompt_cache",
        },
    }
    
    # Mock the registries
    with patch("src.versioning.artifacts.ModelRegistry") as mock_model_registry, \
         patch("src.versioning.artifacts.PromptRegistry") as mock_prompt_registry:
        
        mock_model_registry_instance = MagicMock()
        mock_model_registry_instance.register_model.return_value = "s3://test-bucket/models/test-model/v1"
        mock_model_registry_instance.get_model.return_value = Path("test_model_path")
        mock_model_registry_instance.get_model_metadata.return_value = {"version": "v1", "metadata": {}}
        mock_model_registry_instance.list_models.return_value = ["test-model"]
        mock_model_registry_instance.list_model_versions.return_value = ["v1"]
        
        mock_prompt_registry_instance = MagicMock()
        mock_prompt_registry_instance.register_prompt.return_value = "s3://test-bucket/prompts/test-prompt/v1.txt"
        mock_prompt_registry_instance.get_prompt.return_value = "This is a test prompt."
        mock_prompt_registry_instance.get_prompt_metadata.return_value = {"version": "v1", "metadata": {}}
        mock_prompt_registry_instance.list_prompts.return_value = ["test-prompt"]
        mock_prompt_registry_instance.list_prompt_versions.return_value = ["v1"]
        
        mock_model_registry.return_value = mock_model_registry_instance
        mock_prompt_registry.return_value = mock_prompt_registry_instance
        
        manager = VersionManager(provider, config)
        
        # Initial state
        assert manager.provider == provider
        assert manager.config == config
        assert manager.model_registry == mock_model_registry_instance
        assert manager.prompt_registry == mock_prompt_registry_instance
        
        # Register a model
        model_uri = manager.register_model(
            model_id="test-model",
            model_version="v1",
            model_path=Path("test_model_path"),
            metadata={"framework": "pytorch"},
        )
        
        assert model_uri == "s3://test-bucket/models/test-model/v1"
        mock_model_registry_instance.register_model.assert_called_once_with(
            model_id="test-model",
            model_version="v1",
            model_path=Path("test_model_path"),
            metadata={"framework": "pytorch"},
        )
        
        # Register a prompt
        prompt_uri = manager.register_prompt(
            prompt_id="test-prompt",
            prompt_version="v1",
            prompt_text="This is a test prompt.",
            metadata={"language": "en"},
        )
        
        assert prompt_uri == "s3://test-bucket/prompts/test-prompt/v1.txt"
        mock_prompt_registry_instance.register_prompt.assert_called_once_with(
            prompt_id="test-prompt",
            prompt_version="v1",
            prompt_text="This is a test prompt.",
            metadata={"language": "en"},
        )
        
        # Get a model
        model_path = manager.get_model("test-model", "v1")
        assert model_path == Path("test_model_path")
        mock_model_registry_instance.get_model.assert_called_once()
        
        # Get a prompt
        prompt_text = manager.get_prompt("test-prompt", "v1")
        assert prompt_text == "This is a test prompt."
        mock_prompt_registry_instance.get_prompt.assert_called_once()
        
        # List models and prompts
        models = manager.list_models()
        assert models == ["test-model"]
        
        prompts = manager.list_prompts()
        assert prompts == ["test-prompt"]
        
        # Register a deployment
        manager.register_deployment(
            model_id="test-model",
            model_version="v1",
            deployment_id="test-deployment-id",
            metadata={"environment": "production"},
        )
        
        mock_model_registry_instance.get_model_metadata.assert_called_with("test-model", "v1")
