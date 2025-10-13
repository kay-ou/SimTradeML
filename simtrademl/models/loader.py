"""Model loader with LRU caching for MLflow models.

This module provides efficient model loading from MLflow Registry with
LRU caching to minimize loading overhead.
"""

import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from config.settings import get_settings
from simtrademl.utils.logging import get_logger

logger = get_logger(__name__)


class ModelCache:
    """LRU cache for models with thread-safe operations."""

    def __init__(self, maxsize: int = 10):
        """Initialize model cache.

        Args:
            maxsize: Maximum number of models to cache
        """
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get model from cache.

        Args:
            key: Cache key (format: "model_name:version")

        Returns:
            Cached model or None if not found
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                logger.debug("Cache hit", cache_key=key, hit_rate=self.hit_rate)
                return self._cache[key]
            else:
                self._misses += 1
                logger.debug("Cache miss", cache_key=key, hit_rate=self.hit_rate)
                return None

    def put(self, key: str, model: Any) -> None:
        """Put model in cache with LRU eviction.

        Args:
            key: Cache key (format: "model_name:version")
            model: Model to cache
        """
        with self._lock:
            if key in self._cache:
                # Update existing entry and move to end
                self._cache.move_to_end(key)
                self._cache[key] = model
                logger.debug("Cache updated", cache_key=key, cache_size=len(self._cache))
            else:
                # Add new entry
                if len(self._cache) >= self._maxsize:
                    # Evict LRU (first item)
                    evicted_key, _ = self._cache.popitem(last=False)
                    logger.info(
                        "Cache eviction",
                        evicted_key=evicted_key,
                        new_key=key,
                        cache_size=len(self._cache),
                    )
                self._cache[key] = model
                logger.debug("Cache added", cache_key=key, cache_size=len(self._cache))

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            size_before = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Cache cleared", models_removed=size_before)

    @property
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as a float between 0 and 1
        """
        with self._lock:
            total = self._hits + self._misses
            if total == 0:
                return 0.0
            return self._hits / total

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.hit_rate,
            }


class ModelLoader:
    """Model loader with MLflow integration and LRU caching."""

    def __init__(
        self,
        mlflow_tracking_uri: Optional[str] = None,
        cache_size: Optional[int] = None,
        preload_models: Optional[List[str]] = None,
    ) -> None:
        """Initialize model loader.

        Args:
            mlflow_tracking_uri: MLflow tracking URI (from settings if None)
            cache_size: LRU cache size (from settings if None)
            preload_models: Models to preload on startup (format: ["model_name:version"], from settings if None)
        """
        settings = get_settings()

        # Set MLflow tracking URI
        self.mlflow_tracking_uri = mlflow_tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Initialize MLflow client
        self.client = MlflowClient(tracking_uri=self.mlflow_tracking_uri)

        # Initialize cache with size from settings if not provided
        cache_size = cache_size if cache_size is not None else settings.model_cache_size
        self.cache = ModelCache(maxsize=cache_size)

        # Use preload list from settings if not provided
        preload_models = preload_models if preload_models is not None else settings.model_preload_list

        logger.info(
            "ModelLoader initialized",
            mlflow_uri=self.mlflow_tracking_uri,
            cache_size=cache_size,
            preload_models=preload_models,
        )

        # Preload models if specified
        if preload_models:
            self.preload(preload_models)

    def load_model(
        self,
        model_name: str,
        version: str = "latest",
        stage: Optional[str] = None,
    ) -> Any:
        """Load model from MLflow Registry with caching.

        Args:
            model_name: Model name in registry
            version: Model version (e.g., '1', '2', 'latest')
            stage: Model stage (overrides version if specified)
                   Options: 'Production', 'Staging', 'Archived', 'None'

        Returns:
            Loaded model

        Raises:
            Exception: If model not found or loading fails
        """
        # Resolve version if stage is specified
        if stage:
            resolved_version = self._get_version_by_stage(model_name, stage)
            if resolved_version:
                version = resolved_version
                logger.info(
                    "Resolved version from stage",
                    model_name=model_name,
                    stage=stage,
                    version=version,
                )
            else:
                logger.warning(
                    "No model found for stage, using version",
                    model_name=model_name,
                    stage=stage,
                    version=version,
                )

        # Resolve 'latest' version
        if version == "latest":
            version = self._get_latest_version(model_name)
            logger.debug(
                "Resolved latest version", model_name=model_name, version=version
            )

        # Check cache
        cache_key = f"{model_name}:{version}"
        cached_model = self.cache.get(cache_key)
        if cached_model is not None:
            logger.info("Model loaded from cache", cache_key=cache_key)
            return cached_model

        # Load from MLflow
        try:
            logger.info("Loading model from MLflow", cache_key=cache_key)
            model = self._load_from_registry(model_name, version)

            # Cache the model
            self.cache.put(cache_key, model)

            logger.info(
                "Model loaded successfully",
                cache_key=cache_key,
                cache_size=self.cache.size,
            )
            return model

        except Exception as e:
            logger.error(
                "Failed to load model",
                model_name=model_name,
                version=version,
                error=str(e),
                exc_info=True,
            )
            raise

    def _load_from_registry(self, model_name: str, version: str) -> Any:
        """Load model from MLflow Registry.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            Loaded model
        """
        model_uri = f"models:/{model_name}/{version}"
        logger.debug("Loading from MLflow", model_uri=model_uri)

        # Use mlflow.pyfunc.load_model for maximum compatibility
        model = mlflow.pyfunc.load_model(model_uri)

        return model

    def _get_latest_version(self, model_name: str) -> str:
        """Get latest version number for a model.

        Args:
            model_name: Model name

        Returns:
            Latest version as string

        Raises:
            Exception: If no versions found
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")

            # Get the version with highest version number
            latest = max(versions, key=lambda v: int(v.version))
            return str(latest.version)

        except Exception as e:
            logger.error(
                "Failed to get latest version", model_name=model_name, error=str(e)
            )
            raise

    def _get_version_by_stage(self, model_name: str, stage: str) -> Optional[str]:
        """Get model version for a given stage.

        Args:
            model_name: Model name
            stage: Model stage

        Returns:
            Version string or None if not found
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return str(versions[0].version)
            return None

        except Exception as e:
            logger.warning(
                "Failed to get version by stage",
                model_name=model_name,
                stage=stage,
                error=str(e),
            )
            return None

    def preload(self, models: List[str]) -> None:
        """Preload models into cache.

        Args:
            models: List of models to preload (format: ["model_name:version"])
        """
        logger.info("Preloading models", models=models)

        for model_spec in models:
            try:
                if ":" in model_spec:
                    model_name, version = model_spec.split(":", 1)
                else:
                    model_name = model_spec
                    version = "latest"

                self.load_model(model_name, version)
                logger.info("Model preloaded", model_name=model_name, version=version)

            except Exception as e:
                logger.error(
                    "Failed to preload model",
                    model_spec=model_spec,
                    error=str(e),
                    exc_info=True,
                )
                # Continue with next model

    def clear_cache(self) -> None:
        """Clear model cache."""
        self.cache.clear()
        logger.info("Model cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return self.cache.stats

    def get_model_info(
        self, model_name: str, version: str = "latest"
    ) -> Dict[str, Any]:
        """Get model metadata without loading the model.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            Dictionary with model metadata

        Raises:
            Exception: If model not found
        """
        try:
            # Resolve latest version
            if version == "latest":
                version = self._get_latest_version(model_name)

            # Get model version details
            model_version = self.client.get_model_version(model_name, version)

            # Get run info for additional details
            run = self.client.get_run(model_version.run_id)

            info = {
                "name": model_version.name,
                "version": model_version.version,
                "stage": model_version.current_stage,
                "status": model_version.status,
                "creation_timestamp": model_version.creation_timestamp,
                "last_updated_timestamp": model_version.last_updated_timestamp,
                "run_id": model_version.run_id,
                "source": model_version.source,
                "description": model_version.description,
                "tags": model_version.tags,
                "metrics": run.data.metrics if run else {},
                "params": run.data.params if run else {},
            }

            logger.debug("Retrieved model info", model_name=model_name, version=version)
            return info

        except Exception as e:
            logger.error(
                "Failed to get model info",
                model_name=model_name,
                version=version,
                error=str(e),
                exc_info=True,
            )
            raise


# Example usage
if __name__ == "__main__":
    # Initialize logger
    from config.settings import get_settings

    settings = get_settings()

    # Create model loader
    loader = ModelLoader(
        mlflow_tracking_uri=settings.mlflow_tracking_uri,
        cache_size=5,
    )

    print(f"Model Loader initialized with MLflow URI: {loader.mlflow_tracking_uri}")
    print(f"Cache size: {loader.cache.size}/{loader.cache._maxsize}")
    print(f"Cache stats: {loader.get_cache_stats()}")
