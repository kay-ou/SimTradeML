"""Unit tests for ModelLoader and ModelCache.

This module tests the model loading and caching functionality with mocked MLflow.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from simtrademl.models.loader import ModelCache, ModelLoader


@pytest.mark.unit
class TestModelCache:
    """Test ModelCache LRU caching behavior."""

    def test_cache_initialization(self):
        """Test cache initializes with correct size."""
        cache = ModelCache(maxsize=5)

        assert cache.size == 0
        assert cache._maxsize == 5
        assert cache.hit_rate == 0.0

    def test_cache_put_and_get(self):
        """Test basic cache put and get operations."""
        cache = ModelCache(maxsize=3)

        # Put items
        cache.put("model1:v1", "model_obj_1")
        cache.put("model2:v1", "model_obj_2")

        # Get items
        result1 = cache.get("model1:v1")
        result2 = cache.get("model2:v1")

        assert result1 == "model_obj_1"
        assert result2 == "model_obj_2"
        assert cache.size == 2

    def test_cache_hit(self):
        """Test cache hit tracking."""
        cache = ModelCache(maxsize=3)

        cache.put("model1:v1", "model_obj_1")

        # First get - cache hit
        result = cache.get("model1:v1")
        assert result == "model_obj_1"
        assert cache._hits == 1
        assert cache._misses == 0
        assert cache.hit_rate == 1.0

    def test_cache_miss(self):
        """Test cache miss tracking."""
        cache = ModelCache(maxsize=3)

        # Get non-existent item - cache miss
        result = cache.get("nonexistent:v1")
        assert result is None
        assert cache._hits == 0
        assert cache._misses == 1
        assert cache.hit_rate == 0.0

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        cache = ModelCache(maxsize=3)

        cache.put("model1:v1", "model_obj_1")
        cache.put("model2:v1", "model_obj_2")

        # 2 hits
        cache.get("model1:v1")
        cache.get("model2:v1")

        # 1 miss
        cache.get("nonexistent:v1")

        # Hit rate should be 2/3
        assert cache._hits == 2
        assert cache._misses == 1
        assert cache.hit_rate == pytest.approx(2 / 3)

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ModelCache(maxsize=3)

        # Fill cache
        cache.put("model1:v1", "model_obj_1")
        cache.put("model2:v1", "model_obj_2")
        cache.put("model3:v1", "model_obj_3")

        assert cache.size == 3

        # Access model1 to make it more recently used
        cache.get("model1:v1")

        # Add 4th item - should evict model2 (least recently used)
        cache.put("model4:v1", "model_obj_4")

        assert cache.size == 3
        assert cache.get("model1:v1") == "model_obj_1"  # Still exists
        assert cache.get("model2:v1") is None  # Evicted
        assert cache.get("model3:v1") == "model_obj_3"  # Still exists
        assert cache.get("model4:v1") == "model_obj_4"  # Newly added

    def test_lru_eviction_order(self):
        """Test correct LRU eviction order."""
        cache = ModelCache(maxsize=2)

        cache.put("model1:v1", "model_obj_1")
        cache.put("model2:v1", "model_obj_2")

        # Access model1 to make it more recent
        cache.get("model1:v1")

        # Add model3 - should evict model2
        cache.put("model3:v1", "model_obj_3")

        assert cache.get("model1:v1") == "model_obj_1"
        assert cache.get("model2:v1") is None  # Evicted
        assert cache.get("model3:v1") == "model_obj_3"

    def test_cache_update_existing_key(self):
        """Test updating existing cache entry."""
        cache = ModelCache(maxsize=3)

        cache.put("model1:v1", "model_obj_1")
        cache.put("model1:v1", "model_obj_1_updated")

        assert cache.size == 1
        assert cache.get("model1:v1") == "model_obj_1_updated"

    def test_cache_clear(self):
        """Test clearing cache."""
        cache = ModelCache(maxsize=3)

        cache.put("model1:v1", "model_obj_1")
        cache.put("model2:v1", "model_obj_2")

        # Add some hits
        cache.get("model1:v1")

        assert cache.size == 2
        assert cache._hits == 1

        cache.clear()

        assert cache.size == 0
        assert cache._hits == 0
        assert cache._misses == 0
        assert cache.hit_rate == 0.0

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = ModelCache(maxsize=5)

        cache.put("model1:v1", "model_obj_1")
        cache.get("model1:v1")  # Hit
        cache.get("nonexistent:v1")  # Miss

        stats = cache.stats

        assert stats["size"] == 1
        assert stats["maxsize"] == 5
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


@pytest.mark.unit
class TestModelLoader:
    """Test ModelLoader with mocked MLflow."""

    @patch("simtrademl.models.loader.MlflowClient")
    @patch("simtrademl.models.loader.mlflow")
    def test_loader_initialization(self, mock_mlflow, mock_client):
        """Test ModelLoader initialization."""
        loader = ModelLoader(
            mlflow_tracking_uri="http://test:5000",
            cache_size=5,
            preload_models=[],
        )

        assert loader.mlflow_tracking_uri == "http://test:5000"
        assert loader.cache._maxsize == 5
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://test:5000")

    @patch("simtrademl.models.loader.MlflowClient")
    @patch("simtrademl.models.loader.mlflow")
    def test_loader_initialization_from_settings(self, mock_mlflow, mock_client):
        """Test ModelLoader reads from settings when no params provided."""
        with patch("simtrademl.models.loader.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                mlflow_tracking_uri="http://settings:5000",
                model_cache_size=15,
                model_preload_list=[],
            )

            loader = ModelLoader()

            assert loader.cache._maxsize == 15

    @patch("simtrademl.models.loader.MlflowClient")
    @patch("simtrademl.models.loader.mlflow")
    def test_load_model_cache_hit(self, mock_mlflow, mock_client):
        """Test loading model from cache (cache hit)."""
        mock_model = Mock()
        mock_mlflow.pyfunc.load_model.return_value = mock_model

        loader = ModelLoader(
            mlflow_tracking_uri="http://test:5000",
            cache_size=5,
            preload_models=[],
        )

        # First load - cache miss, loads from MLflow
        model1 = loader.load_model("test_model", "1")
        assert mock_mlflow.pyfunc.load_model.call_count == 1

        # Second load - cache hit, doesn't call MLflow
        model2 = loader.load_model("test_model", "1")
        assert mock_mlflow.pyfunc.load_model.call_count == 1  # Not called again
        assert model1 is model2  # Same object from cache

    @patch("simtrademl.models.loader.MlflowClient")
    @patch("simtrademl.models.loader.mlflow")
    def test_load_model_cache_miss(self, mock_mlflow, mock_client):
        """Test loading model not in cache (cache miss)."""
        mock_model = Mock()
        mock_mlflow.pyfunc.load_model.return_value = mock_model

        loader = ModelLoader(
            mlflow_tracking_uri="http://test:5000",
            cache_size=5,
            preload_models=[],
        )

        model = loader.load_model("test_model", "1")

        assert model is mock_model
        mock_mlflow.pyfunc.load_model.assert_called_once_with(
            "models:/test_model/1"
        )

    @patch("simtrademl.models.loader.MlflowClient")
    @patch("simtrademl.models.loader.mlflow")
    def test_load_model_resolves_latest_version(self, mock_mlflow, mock_client):
        """Test loading model with 'latest' version."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Mock version info
        mock_version = Mock(version="3")
        mock_client_instance.search_model_versions.return_value = [
            Mock(version="1"),
            Mock(version="2"),
            mock_version,
        ]

        mock_model = Mock()
        mock_mlflow.pyfunc.load_model.return_value = mock_model

        loader = ModelLoader(
            mlflow_tracking_uri="http://test:5000",
            cache_size=5,
            preload_models=[],
        )

        model = loader.load_model("test_model", "latest")

        # Should resolve to version 3 and load it
        mock_mlflow.pyfunc.load_model.assert_called_once_with(
            "models:/test_model/3"
        )

    @patch("simtrademl.models.loader.MlflowClient")
    @patch("simtrademl.models.loader.mlflow")
    def test_load_model_by_stage(self, mock_mlflow, mock_client):
        """Test loading model by stage (Production, Staging)."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Mock stage resolution
        mock_version = Mock(version="5")
        mock_client_instance.get_latest_versions.return_value = [mock_version]

        mock_model = Mock()
        mock_mlflow.pyfunc.load_model.return_value = mock_model

        loader = ModelLoader(
            mlflow_tracking_uri="http://test:5000",
            cache_size=5,
            preload_models=[],
        )

        model = loader.load_model("test_model", stage="Production")

        # Should resolve stage to version 5
        mock_client_instance.get_latest_versions.assert_called_once_with(
            "test_model", stages=["Production"]
        )
        mock_mlflow.pyfunc.load_model.assert_called_once_with(
            "models:/test_model/5"
        )

    @patch("simtrademl.models.loader.MlflowClient")
    @patch("simtrademl.models.loader.mlflow")
    def test_cache_eviction_with_multiple_models(self, mock_mlflow, mock_client):
        """Test cache eviction with multiple models."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_mlflow.pyfunc.load_model.side_effect = lambda uri: Mock(name=uri)

        loader = ModelLoader(
            mlflow_tracking_uri="http://test:5000",
            cache_size=2,  # Small cache
            preload_models=[],
        )

        # Load 3 models - should evict first one
        model1 = loader.load_model("model1", "1")
        model2 = loader.load_model("model2", "1")
        model3 = loader.load_model("model3", "1")

        # Check cache state
        assert loader.cache.size == 2
        assert loader.cache.get("model1:1") is None  # Evicted
        assert loader.cache.get("model2:1") is not None
        assert loader.cache.get("model3:1") is not None

    @patch("simtrademl.models.loader.MlflowClient")
    @patch("simtrademl.models.loader.mlflow")
    def test_preload_models(self, mock_mlflow, mock_client):
        """Test preloading models on initialization."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_mlflow.pyfunc.load_model.side_effect = lambda uri: Mock(name=uri)

        loader = ModelLoader(
            mlflow_tracking_uri="http://test:5000",
            cache_size=5,
            preload_models=["model1:1", "model2:2"],
        )

        # Models should be preloaded
        assert loader.cache.size == 2
        assert loader.cache.get("model1:1") is not None
        assert loader.cache.get("model2:2") is not None

    @patch("simtrademl.models.loader.MlflowClient")
    @patch("simtrademl.models.loader.mlflow")
    def test_preload_with_latest_version(self, mock_mlflow, mock_client):
        """Test preloading with 'latest' version."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_version = Mock(version="3")
        mock_client_instance.search_model_versions.return_value = [mock_version]

        mock_mlflow.pyfunc.load_model.side_effect = lambda uri: Mock(name=uri)

        loader = ModelLoader(
            mlflow_tracking_uri="http://test:5000",
            cache_size=5,
            preload_models=["model1"],  # No version = latest
        )

        # Should resolve latest and preload
        assert loader.cache.size == 1
        assert loader.cache.get("model1:3") is not None

    @patch("simtrademl.models.loader.MlflowClient")
    @patch("simtrademl.models.loader.mlflow")
    def test_get_model_info(self, mock_mlflow, mock_client):
        """Test getting model info without loading."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Mock model version - use MagicMock with spec-like attributes
        mock_model_version = MagicMock()
        mock_model_version.name = "test_model"
        mock_model_version.version = "1"
        mock_model_version.current_stage = "Production"
        mock_model_version.status = "READY"
        mock_model_version.creation_timestamp = 1234567890
        mock_model_version.last_updated_timestamp = 1234567900
        mock_model_version.run_id = "run123"
        mock_model_version.source = "s3://bucket/path"
        mock_model_version.description = "Test model"
        mock_model_version.tags = {"key": "value"}
        mock_client_instance.get_model_version.return_value = mock_model_version

        # Mock run
        mock_run = MagicMock()
        mock_run.data.metrics = {"mae": 0.05, "rmse": 0.08}
        mock_run.data.params = {"n_estimators": 100, "max_depth": 5}
        mock_client_instance.get_run.return_value = mock_run

        loader = ModelLoader(
            mlflow_tracking_uri="http://test:5000",
            cache_size=5,
            preload_models=[],
        )

        info = loader.get_model_info("test_model", "1")

        assert info["name"] == "test_model"
        assert info["version"] == "1"
        assert info["stage"] == "Production"
        assert info["metrics"]["mae"] == 0.05
        assert info["params"]["n_estimators"] == 100
        assert mock_mlflow.pyfunc.load_model.call_count == 0  # Not loaded

    @patch("simtrademl.models.loader.MlflowClient")
    @patch("simtrademl.models.loader.mlflow")
    def test_clear_cache(self, mock_mlflow, mock_client):
        """Test clearing cache."""
        mock_mlflow.pyfunc.load_model.side_effect = lambda uri: Mock(name=uri)

        loader = ModelLoader(
            mlflow_tracking_uri="http://test:5000",
            cache_size=5,
            preload_models=[],
        )

        # Load models
        loader.load_model("model1", "1")
        loader.load_model("model2", "1")

        assert loader.cache.size == 2

        # Clear cache
        loader.clear_cache()

        assert loader.cache.size == 0
        assert loader.cache.hit_rate == 0.0

    @patch("simtrademl.models.loader.MlflowClient")
    @patch("simtrademl.models.loader.mlflow")
    def test_get_cache_stats(self, mock_mlflow, mock_client):
        """Test getting cache statistics."""
        mock_mlflow.pyfunc.load_model.side_effect = lambda uri: Mock(name=uri)

        loader = ModelLoader(
            mlflow_tracking_uri="http://test:5000",
            cache_size=5,
            preload_models=[],
        )

        # Load model and access it
        loader.load_model("model1", "1")
        loader.load_model("model1", "1")  # Cache hit

        stats = loader.get_cache_stats()

        assert stats["size"] == 1
        assert stats["maxsize"] == 5
        assert stats["hits"] >= 1

    @patch("simtrademl.models.loader.MlflowClient")
    @patch("simtrademl.models.loader.mlflow")
    def test_load_model_error_handling(self, mock_mlflow, mock_client):
        """Test error handling when model loading fails."""
        mock_mlflow.pyfunc.load_model.side_effect = Exception("MLflow error")

        loader = ModelLoader(
            mlflow_tracking_uri="http://test:5000",
            cache_size=5,
            preload_models=[],
        )

        with pytest.raises(Exception) as exc_info:
            loader.load_model("nonexistent_model", "1")

        assert "MLflow error" in str(exc_info.value)

    @patch("simtrademl.models.loader.MlflowClient")
    @patch("simtrademl.models.loader.mlflow")
    def test_preload_continues_on_error(self, mock_mlflow, mock_client):
        """Test preload continues even if one model fails."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # First model fails, second succeeds
        def load_side_effect(uri):
            if "model1" in uri:
                raise Exception("Model1 not found")
            return Mock(name=uri)

        mock_mlflow.pyfunc.load_model.side_effect = load_side_effect

        loader = ModelLoader(
            mlflow_tracking_uri="http://test:5000",
            cache_size=5,
            preload_models=["model1:1", "model2:1"],
        )

        # Second model should be loaded despite first failing
        assert loader.cache.size == 1
        assert loader.cache.get("model2:1") is not None
