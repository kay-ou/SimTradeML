"""Model loader with caching and async support.

Provides model loading from MLflow Model Registry with LRU caching.
"""

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient

from config.settings import get_settings
from simtrademl.models.base import BaseModel
from simtrademl.models.registry import MODEL_REGISTRY
from simtrademl.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Configure MLflow
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


class ModelLoader:
    """Model loader with caching support.

    Loads models from MLflow Model Registry and caches them in memory.
    """

    def __init__(self, cache_size: int = 10):
        """Initialize model loader.

        Args:
            cache_size: Maximum number of models to cache
        """
        self.cache_size = cache_size
        self._cache: Dict[str, BaseModel] = {}
        self._client = MlflowClient()

        logger.info("ModelLoader initialized", cache_size=cache_size)

    def _get_cache_key(self, model_name: str, model_version: str) -> str:
        """Generate cache key.

        Args:
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            Cache key string
        """
        return f"{model_name}:{model_version}"

    def _resolve_version(self, model_name: str, model_version: str) -> str:
        """Resolve model version alias to actual version number.

        Args:
            model_name: Name of the model
            model_version: Version alias (e.g., "latest", "production")

        Returns:
            Actual version number

        Raises:
            ValueError: If model or version not found
        """
        try:
            if model_version in ["latest", "production", "staging"]:
                # Get model version by alias
                versions = self._client.search_model_versions(
                    f"name='{model_name}'"
                )

                if not versions:
                    raise ValueError(f"Model '{model_name}' not found")

                if model_version == "latest":
                    # Get latest version
                    latest = max(versions, key=lambda v: int(v.version))
                    return latest.version
                else:
                    # Get version by stage
                    stage = model_version.capitalize()
                    for v in versions:
                        if v.current_stage == stage:
                            return v.version

                    raise ValueError(
                        f"No {model_version} version found for model '{model_name}'"
                    )
            else:
                # Assume it's already a version number
                return model_version

        except Exception as e:
            logger.error(
                "Failed to resolve model version",
                model_name=model_name,
                model_version=model_version,
                error=str(e),
            )
            raise

    def _load_from_mlflow(
        self,
        model_name: str,
        model_version: str,
    ) -> BaseModel:
        """Load model from MLflow Model Registry.

        Args:
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            Loaded model instance

        Raises:
            ValueError: If model not found or loading fails
        """
        try:
            # Resolve version
            version = self._resolve_version(model_name, model_version)

            logger.info(
                "Loading model from MLflow",
                model_name=model_name,
                version=version,
            )

            # Get model URI
            model_uri = f"models:/{model_name}/{version}"

            # Load model artifacts
            model_path = mlflow.artifacts.download_artifacts(model_uri)
            model_path = Path(model_path)

            # Get model metadata
            model_version_info = self._client.get_model_version(
                name=model_name,
                version=version,
            )

            # Extract model type from tags or run data
            model_type = None
            if model_version_info.tags:
                model_type = model_version_info.tags.get("model_type")

            if not model_type:
                # Try to infer from run data
                run = self._client.get_run(model_version_info.run_id)
                model_type = run.data.tags.get("model_type")

            if not model_type:
                raise ValueError(
                    f"Model type not found in tags for {model_name}:{version}"
                )

            # Get model class from registry
            if model_type not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model type: {model_type}")

            model_class = MODEL_REGISTRY[model_type]

            # Load model using the appropriate class
            # Look for model file (assumes .pkl extension)
            model_files = list(model_path.glob("*.pkl"))
            if not model_files:
                raise ValueError(f"No .pkl model file found in {model_path}")

            model_file = model_files[0]
            model = model_class.load(model_file)

            logger.info(
                "Model loaded successfully",
                model_name=model_name,
                version=version,
                model_type=model_type,
            )

            return model

        except Exception as e:
            logger.error(
                "Failed to load model from MLflow",
                model_name=model_name,
                model_version=model_version,
                error=str(e),
            )
            raise

    def _load_from_local(
        self,
        model_name: str,
        model_version: str,
    ) -> BaseModel:
        """Load model from local cache directory.

        Fallback method when MLflow is not available.

        Args:
            model_name: Name of the model
            model_version: Version of the model

        Returns:
            Loaded model instance

        Raises:
            ValueError: If model not found
        """
        model_dir = settings.model_cache_dir / model_name / model_version
        model_file = model_dir / "model.pkl"

        if not model_file.exists():
            raise ValueError(
                f"Model {model_name}:{model_version} not found in local cache"
            )

        logger.info(
            "Loading model from local cache",
            model_name=model_name,
            model_version=model_version,
            path=str(model_file),
        )

        # Try to load with each model type
        for model_type, model_class in MODEL_REGISTRY.items():
            try:
                model = model_class.load(model_file)
                logger.info(
                    "Model loaded from local cache",
                    model_name=model_name,
                    model_version=model_version,
                    model_type=model_type,
                )
                return model
            except Exception:
                continue

        raise ValueError(
            f"Failed to load model {model_name}:{model_version} from local cache"
        )

    def load_model(
        self,
        model_name: str,
        model_version: str = "latest",
        use_cache: bool = True,
    ) -> BaseModel:
        """Load model with caching.

        Args:
            model_name: Name of the model
            model_version: Version of the model (default: "latest")
            use_cache: Whether to use cache (default: True)

        Returns:
            Loaded model instance
        """
        cache_key = self._get_cache_key(model_name, model_version)

        # Check cache
        if use_cache and cache_key in self._cache:
            logger.debug(
                "Model loaded from cache",
                model_name=model_name,
                model_version=model_version,
            )
            return self._cache[cache_key]

        # Load model
        try:
            # Try MLflow first
            model = self._load_from_mlflow(model_name, model_version)
        except Exception as mlflow_error:
            logger.warning(
                "MLflow load failed, trying local cache",
                error=str(mlflow_error),
            )
            try:
                # Fallback to local cache
                model = self._load_from_local(model_name, model_version)
            except Exception as local_error:
                logger.error(
                    "Failed to load model from both MLflow and local cache",
                    mlflow_error=str(mlflow_error),
                    local_error=str(local_error),
                )
                raise ValueError(
                    f"Failed to load model {model_name}:{model_version}"
                ) from mlflow_error

        # Update cache
        if use_cache:
            # Implement LRU eviction
            if len(self._cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug("Evicted model from cache", cache_key=oldest_key)

            self._cache[cache_key] = model
            logger.debug("Model added to cache", cache_key=cache_key)

        return model

    async def load_model_async(
        self,
        model_name: str,
        model_version: str = "latest",
        use_cache: bool = True,
    ) -> BaseModel:
        """Load model asynchronously.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            use_cache: Whether to use cache

        Returns:
            Loaded model instance
        """
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(
            None,
            self.load_model,
            model_name,
            model_version,
            use_cache,
        )
        return model

    def preload_models(self, models: list[Tuple[str, str]]) -> None:
        """Preload commonly used models.

        Args:
            models: List of (model_name, model_version) tuples
        """
        logger.info("Preloading models", count=len(models))

        for model_name, model_version in models:
            try:
                self.load_model(model_name, model_version)
                logger.info(
                    "Model preloaded",
                    model_name=model_name,
                    model_version=model_version,
                )
            except Exception as e:
                logger.error(
                    "Failed to preload model",
                    model_name=model_name,
                    model_version=model_version,
                    error=str(e),
                )

    def clear_cache(self) -> None:
        """Clear model cache."""
        self._cache.clear()
        logger.info("Model cache cleared")

    def get_cache_info(self) -> Dict[str, int]:
        """Get cache information.

        Returns:
            Dictionary with cache stats
        """
        return {
            "cache_size": self.cache_size,
            "cached_models": len(self._cache),
            "cache_keys": list(self._cache.keys()),
        }


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get global model loader instance.

    Returns:
        ModelLoader instance
    """
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader
