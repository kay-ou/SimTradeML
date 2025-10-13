"""Model management and health check endpoints.

Provides endpoints for model information and service health checks.
"""

from datetime import datetime
from typing import Dict, Optional

import mlflow
from fastapi import APIRouter, HTTPException, status
from mlflow.tracking import MlflowClient

from config.settings import get_settings
from simtrademl.api.model_loader import get_model_loader
from simtrademl.api.schemas import HealthResponse, ModelInfo
from simtrademl.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter(tags=["management"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check service health and dependencies status",
)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Health status of the service and its dependencies
    """
    dependencies = {}

    # Check MLflow connection
    try:
        client = MlflowClient()
        client.search_registered_models(max_results=1)
        dependencies["mlflow"] = "healthy"
    except Exception as e:
        logger.warning("MLflow health check failed", error=str(e))
        dependencies["mlflow"] = "unhealthy"

    # Check Redis connection (if configured)
    # TODO: Add Redis health check when implemented
    # dependencies["redis"] = "healthy"

    # Determine overall status
    unhealthy_deps = [k for k, v in dependencies.items() if v == "unhealthy"]
    if unhealthy_deps:
        overall_status = "degraded" if len(unhealthy_deps) < len(dependencies) else "unhealthy"
    else:
        overall_status = "healthy"

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        timestamp=datetime.utcnow(),
        dependencies=dependencies,
    )


@router.get(
    "/v1/models/{model_name}/info",
    response_model=ModelInfo,
    status_code=status.HTTP_200_OK,
    summary="Get model information",
    description="Retrieve metadata and information about a specific model",
)
async def get_model_info(
    model_name: str,
    model_version: str = "latest",
) -> ModelInfo:
    """Get model information.

    Args:
        model_name: Name of the model
        model_version: Version of the model (default: "latest")

    Returns:
        Model metadata and information

    Raises:
        HTTPException: If model not found or error occurs
    """
    logger.info(
        "Model info request",
        model_name=model_name,
        model_version=model_version,
    )

    try:
        client = MlflowClient()

        # Resolve version
        if model_version == "latest":
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"Model '{model_name}' not found")
            model_version_info = max(versions, key=lambda v: int(v.version))
        else:
            model_version_info = client.get_model_version(
                name=model_name,
                version=model_version,
            )

        # Get run information for metrics
        run = client.get_run(model_version_info.run_id)

        # Extract metrics
        metrics = {}
        if run.data.metrics:
            metrics = {k: round(v, 4) for k, v in run.data.metrics.items()}

        # Extract model type from tags
        model_type = model_version_info.tags.get("model_type")
        if not model_type:
            model_type = run.data.tags.get("model_type", "unknown")

        # Extract features list from tags
        features_str = run.data.tags.get("features")
        features = features_str.split(",") if features_str else None

        # Get creation timestamp
        created_at = datetime.fromtimestamp(
            model_version_info.creation_timestamp / 1000
        )

        # Create response
        response = ModelInfo(
            name=model_name,
            version=model_version_info.version,
            model_type=model_type,
            created_at=created_at,
            metrics=metrics if metrics else None,
            features=features,
            description=model_version_info.description,
        )

        logger.info(
            "Model info retrieved",
            model_name=model_name,
            version=model_version_info.version,
            model_type=model_type,
        )

        return response

    except Exception as e:
        logger.error(
            "Failed to get model info",
            model_name=model_name,
            model_version=model_version,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {str(e)}",
        )


@router.get(
    "/v1/models",
    status_code=status.HTTP_200_OK,
    summary="List all models",
    description="List all registered models in MLflow Model Registry",
)
async def list_models() -> dict:
    """List all registered models.

    Returns:
        Dictionary with list of models and metadata
    """
    logger.info("List models request")

    try:
        client = MlflowClient()
        registered_models = client.search_registered_models()

        models_info = []
        for rm in registered_models:
            # Get latest version
            versions = client.search_model_versions(f"name='{rm.name}'")
            if versions:
                latest_version = max(versions, key=lambda v: int(v.version))

                models_info.append({
                    "name": rm.name,
                    "latest_version": latest_version.version,
                    "current_stage": latest_version.current_stage,
                    "description": rm.description,
                })

        return {
            "models": models_info,
            "count": len(models_info),
            "timestamp": datetime.utcnow(),
        }

    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Prometheus-compatible metrics endpoint",
)
async def metrics():
    """Prometheus metrics endpoint.

    Returns:
        Prometheus-formatted metrics
    """
    # TODO: Implement Prometheus metrics collection
    # For now, return basic metrics
    model_loader = get_model_loader()
    cache_info = model_loader.get_cache_info()

    metrics_text = f"""# HELP simtrademl_cached_models Number of cached models
# TYPE simtrademl_cached_models gauge
simtrademl_cached_models {cache_info['cached_models']}

# HELP simtrademl_cache_size Maximum cache size
# TYPE simtrademl_cache_size gauge
simtrademl_cache_size {cache_info['cache_size']}
"""

    return metrics_text
