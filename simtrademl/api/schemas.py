"""Pydantic schemas for API request and response models.

Defines data models for inference API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Request model for single prediction.

    Attributes:
        model_name: Name of the model to use
        model_version: Version of the model (default: "latest")
        features: Dictionary of feature values
    """

    model_name: str = Field(
        ...,
        description="Name of the model to use for prediction",
        examples=["xgboost_price_predictor"],
    )
    model_version: str = Field(
        default="latest",
        description="Version of the model to use",
        examples=["latest", "1", "production"],
    )
    features: Dict[str, Any] = Field(
        ...,
        description="Dictionary of feature names to values",
        examples=[
            {
                "close": 100.5,
                "volume": 1000000,
                "rsi_14": 65.2,
                "macd": 0.5,
            }
        ],
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate features dictionary is not empty.

        Args:
            v: Features dictionary

        Returns:
            Validated features

        Raises:
            ValueError: If features is empty
        """
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        return v


class PredictionResponse(BaseModel):
    """Response model for single prediction.

    Attributes:
        prediction: Predicted value
        confidence: Confidence score (if available)
        model_name: Name of the model used
        model_version: Version of the model used
        timestamp: Prediction timestamp
        trace_id: Request trace ID
    """

    prediction: float = Field(
        ...,
        description="Predicted value",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Prediction confidence score",
        ge=0.0,
        le=1.0,
    )
    model_name: str = Field(
        ...,
        description="Name of the model used",
    )
    model_version: str = Field(
        ...,
        description="Version of the model used",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction timestamp (UTC)",
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Request trace ID for debugging",
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions.

    Attributes:
        model_name: Name of the model to use
        model_version: Version of the model (default: "latest")
        features_list: List of feature dictionaries
    """

    model_name: str = Field(
        ...,
        description="Name of the model to use for predictions",
    )
    model_version: str = Field(
        default="latest",
        description="Version of the model to use",
    )
    features_list: List[Dict[str, Any]] = Field(
        ...,
        description="List of feature dictionaries",
        max_length=1000,
    )

    @field_validator("features_list")
    @classmethod
    def validate_features_list(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate features list.

        Args:
            v: List of feature dictionaries

        Returns:
            Validated features list

        Raises:
            ValueError: If list is empty or too large
        """
        if not v:
            raise ValueError("Features list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Features list cannot exceed 1000 items")

        # Check each item is not empty
        for i, features in enumerate(v):
            if not features:
                raise ValueError(f"Features at index {i} cannot be empty")

        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions.

    Attributes:
        predictions: List of predicted values
        model_name: Name of the model used
        model_version: Version of the model used
        timestamp: Prediction timestamp
        trace_id: Request trace ID
        count: Number of predictions
    """

    predictions: List[float] = Field(
        ...,
        description="List of predicted values",
    )
    model_name: str = Field(
        ...,
        description="Name of the model used",
    )
    model_version: str = Field(
        ...,
        description="Version of the model used",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction timestamp (UTC)",
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Request trace ID for debugging",
    )
    count: int = Field(
        ...,
        description="Number of predictions returned",
    )


class ModelInfo(BaseModel):
    """Model metadata information.

    Attributes:
        name: Model name
        version: Model version
        model_type: Type of model (e.g., "xgboost", "lightgbm")
        created_at: Model creation timestamp
        metrics: Model performance metrics
        features: List of expected feature names
        description: Model description
    """

    name: str = Field(
        ...,
        description="Model name",
    )
    version: str = Field(
        ...,
        description="Model version",
    )
    model_type: str = Field(
        ...,
        description="Type of model",
        examples=["xgboost", "lightgbm", "lstm", "arima"],
    )
    created_at: datetime = Field(
        ...,
        description="Model creation timestamp",
    )
    metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Model performance metrics",
        examples=[{"rmse": 0.05, "mae": 0.03, "r2": 0.95}],
    )
    features: Optional[List[str]] = Field(
        default=None,
        description="List of expected feature names",
    )
    description: Optional[str] = Field(
        default=None,
        description="Model description",
    )


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Service status
        version: API version
        timestamp: Health check timestamp
        dependencies: Status of dependencies
    """

    status: str = Field(
        ...,
        description="Service status",
        examples=["healthy", "degraded", "unhealthy"],
    )
    version: str = Field(
        ...,
        description="API version",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp (UTC)",
    )
    dependencies: Optional[Dict[str, str]] = Field(
        default=None,
        description="Status of dependencies (e.g., MLflow, Redis)",
        examples=[{"mlflow": "healthy", "redis": "healthy"}],
    )


class ErrorResponse(BaseModel):
    """Error response model.

    Attributes:
        error: Error message
        detail: Detailed error information
        trace_id: Request trace ID
        timestamp: Error timestamp
    """

    error: str = Field(
        ...,
        description="Error message",
    )
    detail: Optional[str] = Field(
        default=None,
        description="Detailed error information",
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Request trace ID for debugging",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp (UTC)",
    )
