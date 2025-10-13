"""Rate limiting middleware using sliding window algorithm.

This module implements rate limiting to prevent API abuse by tracking
requests per API key using a sliding window approach.
"""

import time
from collections import defaultdict, deque
from typing import Dict

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from simtrademl.utils.logging import get_logger

logger = get_logger(__name__)


class RateLimiter(BaseHTTPMiddleware):
    """Rate limiter middleware using sliding window algorithm.

    This middleware tracks requests per API key and enforces rate limits
    using a sliding window approach. It adds rate limit headers to all
    responses and returns 429 when limits are exceeded.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 100,
        burst_size: int = 20,
    ):
        """Initialize rate limiter middleware.

        Args:
            app: FastAPI application instance
            requests_per_minute: Maximum requests allowed per minute per API key
            burst_size: Maximum burst size (currently informational only)
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.request_history: Dict[str, deque] = defaultdict(deque)

        logger.info(
            "RateLimiter initialized",
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
        )

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting.

        This method is called for every request. It checks the rate limit,
        adds appropriate headers, and either allows the request or returns 429.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in the chain

        Returns:
            HTTP response with rate limit headers

        Raises:
            HTTPException: 429 if rate limit is exceeded
        """
        # Skip rate limiting for health check, docs, and metrics endpoints
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json", "/metrics"]:
            return await call_next(request)

        # Get API key from header (use "anonymous" if not provided)
        api_key = request.headers.get("X-API-Key", "anonymous")

        # Get request history for this API key
        current_time = time.time()
        history = self.request_history[api_key]

        # Remove old entries (older than 60 seconds) - sliding window
        while history and current_time - history[0] > 60:
            history.popleft()

        # Check if rate limit is exceeded
        if len(history) >= self.requests_per_minute:
            logger.warning(
                "Rate limit exceeded",
                api_key=api_key[:8] + "***" if len(api_key) >= 8 else "***",
                requests_in_window=len(history),
                limit=self.requests_per_minute,
            )

            # Return 429 response directly instead of raising HTTPException
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute."
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + 60)),
                },
            )

        # Add current request timestamp to history
        history.append(current_time)

        # Process the request
        response = await call_next(request)

        # Add rate limit headers to response
        remaining = self.requests_per_minute - len(history)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))

        # Log rate limit info (only for first few requests or when getting close to limit)
        if len(history) <= 5 or remaining <= 10:
            logger.debug(
                "Request processed",
                path=request.url.path,
                requests_in_window=len(history),
                remaining=remaining,
            )

        return response

    def get_stats(self) -> Dict[str, any]:
        """Get rate limiter statistics.

        Returns:
            Dictionary with current rate limiter stats
        """
        total_tracked_keys = len(self.request_history)
        current_time = time.time()

        # Count active keys (keys with requests in last minute)
        active_keys = sum(
            1
            for history in self.request_history.values()
            if history and current_time - history[-1] < 60
        )

        return {
            "total_tracked_keys": total_tracked_keys,
            "active_keys": active_keys,
            "requests_per_minute_limit": self.requests_per_minute,
            "burst_size": self.burst_size,
        }

    def clear_old_entries(self, max_age_seconds: int = 3600):
        """Clear old request history entries to prevent memory growth.

        This method should be called periodically to clean up old entries
        for API keys that are no longer active.

        Args:
            max_age_seconds: Maximum age in seconds before clearing history
        """
        current_time = time.time()
        keys_to_remove = []

        for api_key, history in self.request_history.items():
            # If the last request for this key is older than max_age, remove it
            if not history or current_time - history[-1] > max_age_seconds:
                keys_to_remove.append(api_key)

        for api_key in keys_to_remove:
            del self.request_history[api_key]

        if keys_to_remove:
            logger.info(
                "Cleared old rate limiter entries",
                keys_removed=len(keys_to_remove),
            )


# Example usage and testing
if __name__ == "__main__":
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    print("Testing RateLimiter middleware...")

    # Create test app
    app = FastAPI()

    # Add rate limiter (low limit for testing)
    app.add_middleware(RateLimiter, requests_per_minute=5, burst_size=2)

    @app.get("/test")
    async def test_endpoint():
        return {"message": "Success"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # Test with TestClient
    client = TestClient(app)

    # Test 1: First few requests should succeed
    print("\nTest 1: Making 5 requests (should all succeed)...")
    for i in range(5):
        response = client.get("/test", headers={"X-API-Key": "test-key"})
        print(f"  Request {i+1}: {response.status_code}")
        assert response.status_code == 200

    # Test 2: 6th request should be rate limited
    print("\nTest 2: Making 6th request (should be rate limited)...")
    response = client.get("/test", headers={"X-API-Key": "test-key"})
    print(f"  Request 6: {response.status_code}")
    assert response.status_code == 429

    # Test 3: Health endpoint should not be rate limited
    print("\nTest 3: Accessing /health (should not be rate limited)...")
    response = client.get("/health")
    print(f"  Health check: {response.status_code}")
    assert response.status_code == 200

    print("\nAll rate limiter tests passed!")
