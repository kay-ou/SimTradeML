"""Tests for batch prediction endpoint."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from simtrademl.api.app import app
from simtrademl.api.schemas import BatchPredictionRequest


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create mock model."""
    model = MagicMock()
    model.predict = MagicMock(return_value=np.array([100.5, 101.2, 99.8]))
    return model


@pytest.fixture
def mock_model_loader(mock_model):
    """Create mock model loader."""
    loader = MagicMock()
    loader.load_model_async = AsyncMock(return_value=mock_model)
    return loader


class TestBatchPredictEndpoint:
    """Tests for /v1/batch_predict endpoint."""

    def test_batch_predict_success(
        self, client: TestClient, mock_model_loader: MagicMock
    ) -> None:
        """Test successful batch prediction."""
        with patch(
            "simtrademl.api.routers.predict.get_model_loader",
            return_value=mock_model_loader,
        ):
            request_data = {
                "model_name": "test_model",
                "model_version": "latest",
                "features_list": [
                    {"close": 100.0, "volume": 1000},
                    {"close": 101.0, "volume": 1100},
                    {"close": 99.0, "volume": 900},
                ],
            }

            response = client.post("/v1/batch_predict", json=request_data)

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 3
            assert data["model_name"] == "test_model"
            assert data["model_version"] == "latest"
            assert data["count"] == 3
            assert "timestamp" in data
            assert "trace_id" in data

    def test_batch_predict_with_large_batch(
        self, client: TestClient, mock_model_loader: MagicMock
    ) -> None:
        """Test batch prediction with large batch (100 items)."""
        mock_model_loader.load_model_async.return_value.predict.return_value = (
            np.random.randn(100) + 100
        )

        with patch(
            "simtrademl.api.routers.predict.get_model_loader",
            return_value=mock_model_loader,
        ):
            features_list = [
                {"close": 100.0 + i, "volume": 1000 + i * 10} for i in range(100)
            ]
            request_data = {
                "model_name": "test_model",
                "features_list": features_list,
            }

            response = client.post("/v1/batch_predict", json=request_data)

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data["predictions"]) == 100
            assert data["count"] == 100

    def test_batch_predict_exceeds_limit(self, client: TestClient) -> None:
        """Test batch prediction exceeds 1000 item limit."""
        features_list = [{"close": 100.0, "volume": 1000} for _ in range(1001)]
        request_data = {
            "model_name": "test_model",
            "features_list": features_list,
        }

        response = client.post("/v1/batch_predict", json=request_data)

        # Should fail validation
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_batch_predict_empty_list(self, client: TestClient) -> None:
        """Test batch prediction with empty features list."""
        request_data = {
            "model_name": "test_model",
            "features_list": [],
        }

        response = client.post("/v1/batch_predict", json=request_data)

        # Should fail validation
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_batch_predict_empty_features(self, client: TestClient) -> None:
        """Test batch prediction with empty feature dictionaries."""
        request_data = {
            "model_name": "test_model",
            "features_list": [{}, {"close": 100.0}],
        }

        response = client.post("/v1/batch_predict", json=request_data)

        # Should fail validation
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_batch_predict_model_not_found(
        self, client: TestClient, mock_model_loader: MagicMock
    ) -> None:
        """Test batch prediction with non-existent model."""
        mock_model_loader.load_model_async.side_effect = ValueError("Model not found")

        with patch(
            "simtrademl.api.routers.predict.get_model_loader",
            return_value=mock_model_loader,
        ):
            request_data = {
                "model_name": "nonexistent_model",
                "features_list": [{"close": 100.0}],
            }

            response = client.post("/v1/batch_predict", json=request_data)

            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "Model not found" in response.json()["detail"]

    def test_batch_predict_prediction_error(
        self, client: TestClient, mock_model_loader: MagicMock
    ) -> None:
        """Test batch prediction with prediction error."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Prediction failed")
        mock_model_loader.load_model_async.return_value = mock_model

        with patch(
            "simtrademl.api.routers.predict.get_model_loader",
            return_value=mock_model_loader,
        ):
            request_data = {
                "model_name": "test_model",
                "features_list": [{"close": 100.0}],
            }

            response = client.post("/v1/batch_predict", json=request_data)

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_batch_predict_at_max_limit(
        self, client: TestClient, mock_model_loader: MagicMock
    ) -> None:
        """Test batch prediction at exactly 1000 items (max limit)."""
        mock_model_loader.load_model_async.return_value.predict.return_value = (
            np.random.randn(1000) + 100
        )

        with patch(
            "simtrademl.api.routers.predict.get_model_loader",
            return_value=mock_model_loader,
        ):
            features_list = [{"close": 100.0, "volume": 1000} for _ in range(1000)]
            request_data = {
                "model_name": "test_model",
                "features_list": features_list,
            }

            response = client.post("/v1/batch_predict", json=request_data)

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["count"] == 1000
            assert len(data["predictions"]) == 1000


class TestBatchPredictionRequest:
    """Tests for BatchPredictionRequest schema."""

    def test_valid_batch_request(self) -> None:
        """Test valid batch prediction request."""
        request = BatchPredictionRequest(
            model_name="test_model",
            model_version="1.0",
            features_list=[{"close": 100.0}, {"close": 101.0}],
        )
        assert request.model_name == "test_model"
        assert len(request.features_list) == 2

    def test_batch_request_default_version(self) -> None:
        """Test batch request with default version."""
        request = BatchPredictionRequest(
            model_name="test_model",
            features_list=[{"close": 100.0}],
        )
        assert request.model_version == "latest"

    def test_batch_request_empty_list_raises_error(self) -> None:
        """Test empty features list raises validation error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            BatchPredictionRequest(
                model_name="test_model",
                features_list=[],
            )

    def test_batch_request_exceeds_limit_raises_error(self) -> None:
        """Test features list exceeding 1000 items raises error."""
        from pydantic import ValidationError

        features_list = [{"close": 100.0} for _ in range(1001)]
        with pytest.raises(ValidationError, match="at most 1000"):
            BatchPredictionRequest(
                model_name="test_model",
                features_list=features_list,
            )

    def test_batch_request_empty_features_raises_error(self) -> None:
        """Test empty feature dict raises validation error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            BatchPredictionRequest(
                model_name="test_model",
                features_list=[{}, {"close": 100.0}],
            )


@pytest.mark.integration
class TestBatchPredictIntegration:
    """Integration tests for batch prediction."""

    def test_batch_predict_response_format(
        self, client: TestClient, mock_model_loader: MagicMock
    ) -> None:
        """Test batch prediction response format."""
        with patch(
            "simtrademl.api.routers.predict.get_model_loader",
            return_value=mock_model_loader,
        ):
            request_data = {
                "model_name": "test_model",
                "features_list": [
                    {"close": 100.0, "volume": 1000},
                    {"close": 101.0, "volume": 1100},
                ],
            }

            response = client.post("/v1/batch_predict", json=request_data)

            assert response.status_code == status.HTTP_200_OK
            data = response.json()

            # Verify all required fields
            assert "predictions" in data
            assert "model_name" in data
            assert "model_version" in data
            assert "timestamp" in data
            assert "count" in data

            # Verify data types
            assert isinstance(data["predictions"], list)
            assert all(isinstance(p, (int, float)) for p in data["predictions"])
            assert isinstance(data["count"], int)
            assert isinstance(data["model_name"], str)

            # Verify trace_id in response headers
            assert "X-Trace-ID" in response.headers

    def test_batch_predict_handles_various_dtypes(
        self, client: TestClient, mock_model_loader: MagicMock
    ) -> None:
        """Test batch prediction with various data types."""
        with patch(
            "simtrademl.api.routers.predict.get_model_loader",
            return_value=mock_model_loader,
        ):
            request_data = {
                "model_name": "test_model",
                "features_list": [
                    {
                        "close": 100.5,
                        "volume": 1000,
                        "symbol": "AAPL",
                        "is_trading": True,
                    },
                    {
                        "close": 101.2,
                        "volume": 1100,
                        "symbol": "GOOGL",
                        "is_trading": False,
                    },
                ],
            }

            response = client.post("/v1/batch_predict", json=request_data)

            assert response.status_code == status.HTTP_200_OK
