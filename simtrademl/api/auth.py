"""API Key authentication for FastAPI endpoints.

This module provides API key-based authentication using FastAPI Security.
"""

from typing import List, Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from config.settings import get_settings
from simtrademl.utils.logging import get_logger

logger = get_logger(__name__)

# Define API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKeyAuth:
    """API Key authentication dependency.

    This class validates API keys from the X-API-Key header against
    a configured list of valid keys from settings.
    """

    def __init__(self, valid_api_keys: Optional[List[str]] = None):
        """Initialize API key authenticator.

        Args:
            valid_api_keys: List of valid API keys. If None, uses settings.
        """
        self.valid_api_keys = valid_api_keys or []
        logger.debug(
            "APIKeyAuth initialized",
            num_valid_keys=len(self.valid_api_keys),
        )

    async def __call__(self, api_key: Optional[str] = Security(api_key_header)) -> str:
        """Verify API key from request header.

        This method is called automatically by FastAPI when used as a dependency.

        Args:
            api_key: API key extracted from X-API-Key header

        Returns:
            Validated API key string

        Raises:
            HTTPException: 401 if API key is invalid or missing
        """
        # Check if API key is provided
        if not api_key:
            logger.warning("API key missing from request")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is missing. Please provide X-API-Key header.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        # Check if API key is valid
        if api_key not in self.valid_api_keys:
            logger.warning(
                "Invalid API key attempted",
                api_key_prefix=api_key[:8] if len(api_key) >= 8 else "***",
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        logger.debug("API key authenticated successfully")
        return api_key


def get_api_key_dependency() -> APIKeyAuth:
    """Get API key dependency for FastAPI routes.

    This factory function creates an APIKeyAuth instance with keys
    loaded from settings. Use it in FastAPI route dependencies.

    Returns:
        APIKeyAuth instance configured with valid API keys from settings

    Example:
        ```python
        from fastapi import Depends
        from simtrademl.api.auth import get_api_key_dependency

        @router.post("/predict", dependencies=[Depends(get_api_key_dependency())])
        async def predict(data: PredictRequest):
            # This endpoint requires valid API key
            ...
        ```
    """
    settings = get_settings()
    return APIKeyAuth(valid_api_keys=settings.api_keys)


# Example usage and testing
if __name__ == "__main__":
    from config.settings import reload_settings

    settings = reload_settings()
    print(f"Loaded API keys: {len(settings.api_keys)} keys configured")

    auth = get_api_key_dependency()
    print(f"APIKeyAuth created with {len(auth.valid_api_keys)} valid keys")
