"""Model registry for dynamic model loading.

Provides a central registry of all available model types.
"""

from typing import Dict, Type

from simtrademl.models.arima_model import ARIMAModel
from simtrademl.models.base import BaseModel
from simtrademl.models.prophet_model import ProphetModel

# Model registry mapping model type to model class
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "arima": ARIMAModel,
    "prophet": ProphetModel,
}


def register_model(model_type: str, model_class: Type[BaseModel]) -> None:
    """Register a new model type.

    Args:
        model_type: Unique identifier for the model type
        model_class: Model class that inherits from BaseModel

    Raises:
        ValueError: If model type already registered
    """
    if model_type in MODEL_REGISTRY:
        raise ValueError(f"Model type '{model_type}' already registered")

    if not issubclass(model_class, BaseModel):
        raise ValueError(
            f"Model class must inherit from BaseModel, got {model_class}"
        )

    MODEL_REGISTRY[model_type] = model_class


def get_model_class(model_type: str) -> Type[BaseModel]:
    """Get model class by type.

    Args:
        model_type: Model type identifier

    Returns:
        Model class

    Raises:
        ValueError: If model type not found
    """
    if model_type not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available types: {available}"
        )

    return MODEL_REGISTRY[model_type]


def list_model_types() -> list[str]:
    """List all registered model types.

    Returns:
        List of model type identifiers
    """
    return list(MODEL_REGISTRY.keys())
