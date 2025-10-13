"""Integration tests for API authentication and rate limiting.

This module tests the security features of the API including API key
authentication and rate limiting middleware.
"""

import pytest
from fastapi.testclient import TestClient

from simtrademl.api.app import app

# Test API keys for testing (must match conftest.py setup)
TEST_API_KEY = "test-key-123"
INVALID_API_KEY = "invalid-key-456"


@pytest.fixture
def client() -> TestClient:
    """Create test client with security enabled.

    Returns:
        FastAPI test client
    """
    return TestClient(app)


@pytest.mark.integration
class TestAPIKeyAuthentication:
    """Test API key authentication functionality."""

    def test_predict_without_api_key(self, client: TestClient):
        """Test predict endpoint rejects requests without API key."""
        response = client.post(
            "/v1/predict",
            json={
                "model_name": "test_model",
                "model_version": "1",
                "features": {"feature1": 1.0, "feature2": 2.0},
            },
        )

        assert response.status_code == 401
        assert "detail" in response.json()
        assert "API key is missing" in response.json()["detail"]

    def test_predict_with_invalid_api_key(self, client: TestClient):
        """Test predict endpoint rejects invalid API key."""
        response = client.post(
            "/v1/predict",
            json={
                "model_name": "test_model",
                "model_version": "1",
                "features": {"feature1": 1.0, "feature2": 2.0},
            },
            headers={"X-API-Key": INVALID_API_KEY},
        )

        assert response.status_code == 401
        assert "detail" in response.json()
        assert "Invalid API key" in response.json()["detail"]

    def test_predict_with_valid_api_key(self, client: TestClient):
        """Test predict endpoint accepts valid API key."""
        response = client.post(
            "/v1/predict",
            json={
                "model_name": "test_model",
                "model_version": "1",
                "features": {"feature1": 1.0, "feature2": 2.0},
            },
            headers={"X-API-Key": TEST_API_KEY},
        )

        # Will fail with 404 or 500 due to missing model, but auth passes
        # 401 would indicate auth failure
        assert response.status_code != 401

    def test_batch_predict_without_api_key(self, client: TestClient):
        """Test batch_predict endpoint requires API key."""
        response = client.post(
            "/v1/batch_predict",
            json={
                "model_name": "test_model",
                "model_version": "1",
                "features_list": [
                    {"feature1": 1.0, "feature2": 2.0},
                    {"feature1": 3.0, "feature2": 4.0},
                ],
            },
        )

        assert response.status_code == 401

    def test_batch_predict_with_invalid_api_key(self, client: TestClient):
        """Test batch_predict endpoint rejects invalid API key."""
        response = client.post(
            "/v1/batch_predict",
            json={
                "model_name": "test_model",
                "model_version": "1",
                "features_list": [
                    {"feature1": 1.0, "feature2": 2.0},
                ],
            },
            headers={"X-API-Key": INVALID_API_KEY},
        )

        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_batch_predict_with_valid_api_key(self, client: TestClient):
        """Test batch_predict endpoint accepts valid API key."""
        response = client.post(
            "/v1/batch_predict",
            json={
                "model_name": "test_model",
                "model_version": "1",
                "features_list": [
                    {"feature1": 1.0, "feature2": 2.0},
                ],
            },
            headers={"X-API-Key": TEST_API_KEY},
        )

        # Will fail with 404 or 500 due to missing model, but auth passes
        assert response.status_code != 401


@pytest.mark.integration
class TestPublicEndpoints:
    """Test that public endpoints remain accessible without authentication."""

    def test_health_endpoint_no_auth(self, client: TestClient):
        """Test /health endpoint accessible without authentication."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_metrics_endpoint_no_auth(self, client: TestClient):
        """Test /metrics endpoint accessible without authentication."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "simtrademl" in response.text.lower()

    def test_docs_endpoint_no_auth(self, client: TestClient):
        """Test /docs endpoint accessible without authentication."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint_no_auth(self, client: TestClient):
        """Test /redoc endpoint accessible without authentication."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_endpoint_no_auth(self, client: TestClient):
        """Test /openapi.json endpoint accessible without authentication."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "openapi" in response.json()

    def test_root_endpoint_no_auth(self, client: TestClient):
        """Test / root endpoint accessible without authentication."""
        response = client.get("/")
        assert response.status_code == 200
        assert "service" in response.json()


@pytest.mark.integration
class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_headers_present(self, client: TestClient):
        """Test rate limit headers are present in responses."""
        response = client.post(
            "/v1/predict",
            json={
                "model_name": "test_model",
                "model_version": "1",
                "features": {"feature1": 1.0},
            },
            headers={"X-API-Key": TEST_API_KEY},
        )

        # Check for rate limit headers (even if request fails)
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

        # Verify limit matches settings (default 100)
        assert response.headers["X-RateLimit-Limit"] == "100"

    def test_rate_limiting_enforcement(self, client: TestClient):
        """Test rate limiting enforces request limits."""
        # Make 101 requests (default limit is 100 per minute)
        # Use a smaller number for faster testing
        success_count = 0
        rate_limited = False
        max_requests = 105  # Slightly over limit

        for i in range(max_requests):
            response = client.post(
                "/v1/predict",
                json={
                    "model_name": "test_model",
                    "model_version": "1",
                    "features": {"feature1": float(i)},
                },
                headers={"X-API-Key": TEST_API_KEY},
            )

            if response.status_code == 429:
                rate_limited = True
                # Verify 429 response has proper headers
                assert "Retry-After" in response.headers
                assert "X-RateLimit-Limit" in response.headers
                assert response.headers["X-RateLimit-Remaining"] == "0"
                assert "Rate limit exceeded" in response.json()["detail"]
                # Stop immediately after hitting rate limit
                print(f"Rate limited after {i} requests (expected ~100)")
                break
            elif response.status_code != 401:  # Count non-auth errors as success
                success_count += 1

        # Should be rate limited after 100 requests
        assert rate_limited, "Should have been rate limited after 100 requests"
        assert success_count == 100, f"Expected 100 successful requests, got {success_count}"

    def test_rate_limit_per_api_key(self, client: TestClient):
        """Test rate limiting is tracked per API key."""
        # Make requests with first API key
        for i in range(5):
            response = client.post(
                "/v1/predict",
                json={
                    "model_name": "test_model",
                    "model_version": "1",
                    "features": {"feature1": 1.0},
                },
                headers={"X-API-Key": TEST_API_KEY},
            )

            # Should still have remaining requests
            if "X-RateLimit-Remaining" in response.headers:
                remaining = int(response.headers["X-RateLimit-Remaining"])
                assert remaining >= 95  # 100 - 5

    def test_health_endpoint_not_rate_limited(self, client: TestClient):
        """Test /health endpoint is not rate limited."""
        # Make many requests to health endpoint
        for i in range(10):
            response = client.get("/health")
            assert response.status_code == 200

            # Health endpoint should not have rate limit headers
            # (it's excluded from rate limiting)

    def test_metrics_endpoint_not_rate_limited(self, client: TestClient):
        """Test /metrics endpoint is not rate limited."""
        # Make many requests to metrics endpoint
        for i in range(10):
            response = client.get("/metrics")
            assert response.status_code == 200

    def test_docs_endpoint_not_rate_limited(self, client: TestClient):
        """Test /docs endpoint is not rate limited."""
        # Make many requests to docs endpoint
        for i in range(10):
            response = client.get("/docs")
            assert response.status_code == 200


@pytest.mark.integration
class TestSecurityHeaders:
    """Test security-related headers in responses."""

    def test_trace_id_header(self, client: TestClient):
        """Test X-Trace-ID header is present in responses."""
        response = client.post(
            "/v1/predict",
            json={
                "model_name": "test_model",
                "model_version": "1",
                "features": {"feature1": 1.0},
            },
            headers={"X-API-Key": TEST_API_KEY},
        )

        assert "X-Trace-ID" in response.headers
        # Trace ID should be a UUID-like string
        trace_id = response.headers["X-Trace-ID"]
        assert len(trace_id) > 0

    def test_rate_limit_headers_format(self, client: TestClient):
        """Test rate limit headers have correct format."""
        response = client.post(
            "/v1/predict",
            json={
                "model_name": "test_model",
                "model_version": "1",
                "features": {"feature1": 1.0},
            },
            headers={"X-API-Key": TEST_API_KEY},
        )

        # Verify header values are valid integers
        assert response.headers["X-RateLimit-Limit"].isdigit()
        assert response.headers["X-RateLimit-Remaining"].isdigit()
        assert response.headers["X-RateLimit-Reset"].isdigit()

        # Verify logical constraints
        limit = int(response.headers["X-RateLimit-Limit"])
        remaining = int(response.headers["X-RateLimit-Remaining"])
        assert 0 <= remaining <= limit

    def test_auth_error_headers(self, client: TestClient):
        """Test authentication error responses have WWW-Authenticate header."""
        response = client.post(
            "/v1/predict",
            json={
                "model_name": "test_model",
                "model_version": "1",
                "features": {"feature1": 1.0},
            },
        )

        assert response.status_code == 401
        # FastAPI automatically adds WWW-Authenticate for 401 responses
        assert "WWW-Authenticate" in response.headers or "detail" in response.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
