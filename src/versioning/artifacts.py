"""
Version control system for model artifacts and prompts.
"""
from typing import Dict, List, Optional, Any, Union
import logging
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
import hashlib

from ..core.providers import CloudProvider

logger = logging.getLogger(__name__)


class ArtifactMetadata:
    """Metadata for a versioned artifact."""
    
    def __init__(
        self,
        artifact_id: str,
        artifact_type: str,
        version: str,
        created_at: datetime,
        created_by: str,
        storage_uri: str,
        metadata: Dict[str, Any],
    ):
        """
        Initialize artifact metadata.
        
        Args:
            artifact_id: Identifier for the artifact
            artifact_type: Type of artifact (model, prompt, etc.)
            version: Version of the artifact
            created_at: Creation timestamp
            created_by: Creator identifier
            storage_uri: URI where the artifact is stored
            metadata: Additional metadata
        """
        self.artifact_id = artifact_id
        self.artifact_type = artifact_type
        self.version = version
        self.created_at = created_at
        self.created_by = created_by
        self.storage_uri = storage_uri
        self.metadata = metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "storage_uri": self.storage_uri,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArtifactMetadata':
        """
        Create metadata from a dictionary.
        
        Args:
            data: Dictionary representation of metadata
            
        Returns:
            ArtifactMetadata instance
        """
        return cls(
            artifact_id=data["artifact_id"],
            artifact_type=data["artifact_type"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"],
            storage_uri=data["storage_uri"],
            metadata=data["metadata"],
        )


class ModelRegistry:
    """Registry for versioned models."""
    
    def __init__(self, provider: CloudProvider, config: Dict[str, Any]):
        """
        Initialize the model registry.
        
        Args:
            provider: Cloud provider instance
            config: Registry configuration
        """
        self.provider = provider
        self.config = config
        self.storage_prefix = config.get("storage_prefix", "models")
        self.local_cache_dir = Path(config.get("local_cache_dir", "model_cache"))
        self.metadata_file = self.local_cache_dir / "model_metadata.json"
        self.models = {}  # Map of model_id to dict of version -> metadata
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.local_cache_dir, exist_ok=True)
        
        # Load existing metadata
        self._load_metadata()
        
        logger.info("Model registry initialized")
    
    def _load_metadata(self) -> None:
        """Load metadata from file."""
        if not self.metadata_file.exists():
            return
        
        try:
            with open(self.metadata_file, "r") as f:
                data = json.load(f)
            
            for model_id, versions in data.items():
                self.models[model_id] = {}
                for version, metadata in versions.items():
                    self.models[model_id][version] = ArtifactMetadata.from_dict(metadata)
            
            logger.info(f"Loaded metadata for {len(self.models)} models")
        except Exception as e:
            logger.error(f"Error loading model metadata: {str(e)}", exc_info=True)
    
    def _save_metadata(self) -> None:
        """Save metadata to file."""
        try:
            data = {}
            for model_id, versions in self.models.items():
                data[model_id] = {}
                for version, metadata in versions.items():
                    data[model_id][version] = metadata.to_dict()
            
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Saved model metadata")
        except Exception as e:
            logger.error(f"Error saving model metadata: {str(e)}", exc_info=True)
    
    def register_model(
        self,
        model_id: str,
        model_version: str,
        model_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a model with the registry.
        
        Args:
            model_id: Identifier for the model
            model_version: Version of the model
            model_path: Path to model artifacts
            metadata: Additional metadata
            
        Returns:
            Storage URI of the registered model
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Generate storage path
        storage_path = f"{self.storage_prefix}/{model_id}/{model_version}"
        
        # Upload model artifacts to storage
        storage_uri = self.provider.store_artifact(
            artifact_path=model_path,
            destination_path=storage_path,
        )
        
        # Create metadata
        model_metadata = ArtifactMetadata(
            artifact_id=model_id,
            artifact_type="model",
            version=model_version,
            created_at=datetime.now(),
            created_by=self.config.get("user_id", "unknown"),
            storage_uri=storage_uri,
            metadata=metadata or {},
        )
        
        # Add to registry
        if model_id not in self.models:
            self.models[model_id] = {}
        
        self.models[model_id][model_version] = model_metadata
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Registered model {model_id} version {model_version}")
        
        return storage_uri
    
    def get_model(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        destination_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Get a model from the registry.
        
        Args:
            model_id: Identifier for the model
            model_version: Version of the model (None for latest)
            destination_path: Path to download the model (None for cache)
            
        Returns:
            Path to the downloaded model
        """
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        
        # Get the requested version or latest
        if model_version is None:
            model_version = self._get_latest_version(model_id)
        
        if model_version not in self.models[model_id]:
            raise ValueError(f"Model version not found: {model_id} version {model_version}")
        
        model_metadata = self.models[model_id][model_version]
        
        # Determine destination path
        if destination_path is None:
            destination_path = self.local_cache_dir / model_id / model_version
        else:
            destination_path = Path(destination_path)
        
        # Create destination directory
        os.makedirs(destination_path, exist_ok=True)
        
        # Download model artifacts
        self.provider.retrieve_artifact(
            artifact_uri=model_metadata.storage_uri,
            destination_path=destination_path,
        )
        
        logger.info(f"Retrieved model {model_id} version {model_version}")
        
        return destination_path
    
    def get_model_metadata(
        self,
        model_id: str,
        model_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get metadata for a model.
        
        Args:
            model_id: Identifier for the model
            model_version: Version of the model (None for latest)
            
        Returns:
            Model metadata
        """
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        
        # Get the requested version or latest
        if model_version is None:
            model_version = self._get_latest_version(model_id)
        
        if model_version not in self.models[model_id]:
            raise ValueError(f"Model version not found: {model_id} version {model_version}")
        
        return self.models[model_id][model_version].to_dict()
    
    def list_models(self) -> List[str]:
        """
        List all registered models.
        
        Returns:
            List of model IDs
        """
        return list(self.models.keys())
    
    def list_model_versions(self, model_id: str) -> List[str]:
        """
        List all versions of a model.
        
        Args:
            model_id: Identifier for the model
            
        Returns:
            List of model versions
        """
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        
        return list(self.models[model_id].keys())
    
    def _get_latest_version(self, model_id: str) -> str:
        """
        Get the latest version of a model.
        
        Args:
            model_id: Identifier for the model
            
        Returns:
            Latest version
        """
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        
        versions = list(self.models[model_id].keys())
        if not versions:
            raise ValueError(f"No versions found for model: {model_id}")
        
        # Try semantic versioning first
        try:
            from packaging import version
            return max(versions, key=lambda v: version.parse(v))
        except (ImportError, ValueError):
            # Fall back to string comparison
            return max(versions)


class PromptRegistry:
    """Registry for versioned prompts."""
    
    def __init__(self, provider: CloudProvider, config: Dict[str, Any]):
        """
        Initialize the prompt registry.
        
        Args:
            provider: Cloud provider instance
            config: Registry configuration
        """
        self.provider = provider
        self.config = config
        self.storage_prefix = config.get("storage_prefix", "prompts")
        self.local_cache_dir = Path(config.get("local_cache_dir", "prompt_cache"))
        self.metadata_file = self.local_cache_dir / "prompt_metadata.json"
        self.prompts = {}  # Map of prompt_id to dict of version -> metadata
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.local_cache_dir, exist_ok=True)
        
        # Load existing metadata
        self._load_metadata()
        
        logger.info("Prompt registry initialized")
    
    def _load_metadata(self) -> None:
        """Load metadata from file."""
        if not self.metadata_file.exists():
            return
        
        try:
            with open(self.metadata_file, "r") as f:
                data = json.load(f)
            
            for prompt_id, versions in data.items():
                self.prompts[prompt_id] = {}
                for version, metadata in versions.items():
                    self.prompts[prompt_id][version] = ArtifactMetadata.from_dict(metadata)
            
            logger.info(f"Loaded metadata for {len(self.prompts)} prompts")
        except Exception as e:
            logger.error(f"Error loading prompt metadata: {str(e)}", exc_info=True)
    
    def _save_metadata(self) -> None:
        """Save metadata to file."""
        try:
            data = {}
            for prompt_id, versions in self.prompts.items():
                data[prompt_id] = {}
                for version, metadata in versions.items():
                    data[prompt_id][version] = metadata.to_dict()
            
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Saved prompt metadata")
        except Exception as e:
            logger.error(f"Error saving prompt metadata: {str(e)}", exc_info=True)
    
    def register_prompt(
        self,
        prompt_id: str,
        prompt_version: str,
        prompt_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a prompt with the registry.
        
        Args:
            prompt_id: Identifier for the prompt
            prompt_version: Version of the prompt
            prompt_text: The prompt text
            metadata: Additional metadata
            
        Returns:
            Storage URI of the registered prompt
        """
        # Create a temporary file for the prompt
        prompt_dir = self.local_cache_dir / "temp"
        os.makedirs(prompt_dir, exist_ok=True)
        prompt_file = prompt_dir / f"{prompt_id}_{prompt_version}.txt"
        
        with open(prompt_file, "w") as f:
            f.write(prompt_text)
        
        # Generate storage path
        storage_path = f"{self.storage_prefix}/{prompt_id}/{prompt_version}.txt"
        
        # Upload prompt to storage
        storage_uri = self.provider.store_artifact(
            artifact_path=prompt_file,
            destination_path=storage_path,
        )
        
        # Create metadata
        prompt_metadata = ArtifactMetadata(
            artifact_id=prompt_id,
            artifact_type="prompt",
            version=prompt_version,
            created_at=datetime.now(),
            created_by=self.config.get("user_id", "unknown"),
            storage_uri=storage_uri,
            metadata=metadata or {},
        )
        
        # Add content hash to metadata
        prompt_metadata.metadata["content_hash"] = hashlib.sha256(prompt_text.encode()).hexdigest()
        prompt_metadata.metadata["content_length"] = len(prompt_text)
        
        # Add to registry
        if prompt_id not in self.prompts:
            self.prompts[prompt_id] = {}
        
        self.prompts[prompt_id][prompt_version] = prompt_metadata
        
        # Save metadata
        self._save_metadata()
        
        # Clean up temporary file
        os.remove(prompt_file)
        
        logger.info(f"Registered prompt {prompt_id} version {prompt_version}")
        
        return storage_uri
    
    def get_prompt(
        self,
        prompt_id: str,
        prompt_version: Optional[str] = None,
    ) -> str:
        """
        Get a prompt from the registry.
        
        Args:
            prompt_id: Identifier for the prompt
            prompt_version: Version of the prompt (None for latest)
            
        Returns:
            Prompt text
        """
        if prompt_id not in self.prompts:
            raise ValueError(f"Prompt not found: {prompt_id}")
        
        # Get the requested version or latest
        if prompt_version is None:
            prompt_version = self._get_latest_version(prompt_id)
        
        if prompt_version not in self.prompts[prompt_id]:
            raise ValueError(f"Prompt version not found: {prompt_id} version {prompt_version}")
        
        prompt_metadata = self.prompts[prompt_id][prompt_version]
        
        # Create a temporary file for the prompt
        prompt_dir = self.local_cache_dir / prompt_id
        os.makedirs(prompt_dir, exist_ok=True)
        prompt_file = prompt_dir / f"{prompt_version}.txt"
        
        # Download prompt
        self.provider.retrieve_artifact(
            artifact_uri=prompt_metadata.storage_uri,
            destination_path=prompt_file,
        )
        
        # Read prompt text
        with open(prompt_file, "r") as f:
            prompt_text = f.read()
        
        logger.info(f"Retrieved prompt {prompt_id} version {prompt_version}")
        
        return prompt_text
    
    def get_prompt_metadata(
        self,
        prompt_id: str,
        prompt_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get metadata for a prompt.
        
        Args:
            prompt_id: Identifier for the prompt
            prompt_version: Version of the prompt (None for latest)
            
        Returns:
            Prompt metadata
        """
        if prompt_id not in self.prompts:
            raise ValueError(f"Prompt not found: {prompt_id}")
        
        # Get the requested version or latest
        if prompt_version is None:
            prompt_version = self._get_latest_version(prompt_id)
        
        if prompt_version not in self.prompts[prompt_id]:
            raise ValueError(f"Prompt version not found: {prompt_id} version {prompt_version}")
        
        return self.prompts[prompt_id][prompt_version].to_dict()
    
    def list_prompts(self) -> List[str]:
        """
        List all registered prompts.
        
        Returns:
            List of prompt IDs
        """
        return list(self.prompts.keys())
    
    def list_prompt_versions(self, prompt_id: str) -> List[str]:
        """
        List all versions of a prompt.
        
        Args:
            prompt_id: Identifier for the prompt
            
        Returns:
            List of prompt versions
        """
        if prompt_id not in self.prompts:
            raise ValueError(f"Prompt not found: {prompt_id}")
        
        return list(self.prompts[prompt_id].keys())
    
    def _get_latest_version(self, prompt_id: str) -> str:
        """
        Get the latest version of a prompt.
        
        Args:
            prompt_id: Identifier for the prompt
            
        Returns:
            Latest version
        """
        if prompt_id not in self.prompts:
            raise ValueError(f"Prompt not found: {prompt_id}")
        
        versions = list(self.prompts[prompt_id].keys())
        if not versions:
            raise ValueError(f"No versions found for prompt: {prompt_id}")
        
        # Try semantic versioning first
        try:
            from packaging import version
            return max(versions, key=lambda v: version.parse(v))
        except (ImportError, ValueError):
            # Fall back to string comparison
            return max(versions)


class VersionManager:
    """
    Version control system for model artifacts and prompts.
    """
    
    def __init__(self, provider: CloudProvider, config: Dict[str, Any]):
        """
        Initialize the version manager.
        
        Args:
            provider: Cloud provider instance
            config: Version manager configuration
        """
        self.provider = provider
        self.config = config
        
        # Initialize registries
        self.model_registry = ModelRegistry(
            provider=provider,
            config=config.get("model_registry", {}),
        )
        
        self.prompt_registry = PromptRegistry(
            provider=provider,
            config=config.get("prompt_registry", {}),
        )
        
        logger.info("Version manager initialized")
    
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
            metadata: Additional metadata
            
        Returns:
            Storage URI of the registered model
        """
        return self.model_registry.register_model(
            model_id=model_id,
            model_version=model_version,
            model_path=model_path,
            metadata=metadata,
        )
    
    def get_model(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        destination_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Get a model from the version manager.
        
        Args:
            model_id: Identifier for the model
            model_version: Version of the model (None for latest)
            destination_path: Path to download the model (None for cache)
            
        Returns:
            Path to the downloaded model
        """
        return self.model_registry.get_model(
            model_id=model_id,
            model_version=model_version,
            destination_path=destination_path,
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
            metadata: Additional metadata
            
        Returns:
            Storage URI of the registered prompt
        """
        return self.prompt_registry.register_prompt(
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            prompt_text=prompt_text,
            metadata=metadata,
        )
    
    def get_prompt(
        self,
        prompt_id: str,
        prompt_version: Optional[str] = None,
    ) -> str:
        """
        Get a prompt from the version manager.
        
        Args:
            prompt_id: Identifier for the prompt
            prompt_version: Version of the prompt (None for latest)
            
        Returns:
            Prompt text
        """
        return self.prompt_registry.get_prompt(
            prompt_id=prompt_id,
            prompt_version=prompt_version,
        )
    
    def list_models(self) -> List[str]:
        """
        List all registered models.
        
        Returns:
            List of model IDs
        """
        return self.model_registry.list_models()
    
    def list_model_versions(self, model_id: str) -> List[str]:
        """
        List all versions of a model.
        
        Args:
            model_id: Identifier for the model
            
        Returns:
            List of model versions
        """
        return self.model_registry.list_model_versions(model_id)
    
    def list_prompts(self) -> List[str]:
        """
        List all registered prompts.
        
        Returns:
            List of prompt IDs
        """
        return self.prompt_registry.list_prompts()
    
    def list_prompt_versions(self, prompt_id: str) -> List[str]:
        """
        List all versions of a prompt.
        
        Args:
            prompt_id: Identifier for the prompt
            
        Returns:
            List of prompt versions
        """
        return self.prompt_registry.list_prompt_versions(prompt_id)
    
    def get_model_metadata(
        self,
        model_id: str,
        model_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get metadata for a model.
        
        Args:
            model_id: Identifier for the model
            model_version: Version of the model (None for latest)
            
        Returns:
            Model metadata
        """
        return self.model_registry.get_model_metadata(
            model_id=model_id,
            model_version=model_version,
        )
    
    def get_prompt_metadata(
        self,
        prompt_id: str,
        prompt_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get metadata for a prompt.
        
        Args:
            prompt_id: Identifier for the prompt
            prompt_version: Version of the prompt (None for latest)
            
        Returns:
            Prompt metadata
        """
        return self.prompt_registry.get_prompt_metadata(
            prompt_id=prompt_id,
            prompt_version=prompt_version,
        )
    
    def register_deployment(
        self,
        model_id: str,
        model_version: str,
        deployment_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a deployment with the version manager.
        
        Args:
            model_id: Identifier for the model
            model_version: Version of the model
            deployment_id: ID of the deployment (optional)
            metadata: Additional metadata
        """
        # Get model metadata
        model_metadata = self.get_model_metadata(model_id, model_version)
        
        # Add deployment information to model metadata
        if "deployments" not in model_metadata["metadata"]:
            model_metadata["metadata"]["deployments"] = []
        
        deployment_info = {
            "deployment_id": deployment_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        model_metadata["metadata"]["deployments"].append(deployment_info)
        
        # Update model metadata
        self.model_registry.models[model_id][model_version].metadata = model_metadata["metadata"]
        
        # Save metadata
        self.model_registry._save_metadata()
        
        logger.info(f"Registered deployment for model {model_id} version {model_version}")
