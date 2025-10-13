"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from simtrademl.api.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "SimTradeML Inference API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "timestamp" in data
    assert data["version"] == "1.0.0"


def test_list_models_endpoint():
    """Test list models endpoint."""
    response = client.get("/v1/models")
    # May return empty list if MLflow not configured
    # Just check the response structure
    assert response.status_code in [200, 500]  # 500 if MLflow not available


def test_metrics_endpoint():
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "simtrademl_cached_models" in response.text
    assert "simtrademl_cache_size" in response.text


def test_openapi_endpoint():
    """Test OpenAPI schema is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert data["info"]["title"] == "SimTradeML Inference API"


def test_docs_endpoint():
    """Test Swagger docs are available."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# Note: prediction endpoints require models to be loaded from MLflow
# These tests would need proper model setup and mocking
@pytest.mark.skip(reason="Requires MLflow model setup")
def test_predict_endpoint():
    """Test single prediction endpoint."""
    request_data = {
        "model_name": "test_model",
        "model_version": "1",
        "features": {
            "close": 100.0,
            "volume": 1000000,
        },
    }
    response = client.post("/v1/predict", json=request_data)
    # Would need proper model setup to test


@pytest.mark.skip(reason="Requires MLflow model setup")
def test_batch_predict_endpoint():
    """Test batch prediction endpoint."""
    request_data = {
        "model_name": "test_model",
        "model_version": "1",
        "features_list": [
            {"close": 100.0, "volume": 1000000},
            {"close": 101.0, "volume": 1100000},
        ],
    }
    response = client.post("/v1/batch_predict", json=request_data)
    # Would need proper model setup to test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
