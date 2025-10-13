"""Prediction endpoints for model inference.

Provides single and batch prediction endpoints.
"""

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from simtrademl.api.auth import get_api_key_dependency
from simtrademl.api.model_loader import get_model_loader
from simtrademl.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from simtrademl.utils.logging import get_logger, trace_id_var

logger = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["predictions"])


def _prepare_features(features: Dict[str, any]) -> pd.DataFrame:
    """Prepare features for prediction.

    Args:
        features: Dictionary of feature values

    Returns:
        DataFrame with features
    """
    # Convert dict to DataFrame with single row
    df = pd.DataFrame([features])
    return df


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Make a single prediction",
    description="Make a prediction using a specified model and feature values",
    dependencies=[Depends(get_api_key_dependency())],
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Make a single prediction.

    Args:
        request: Prediction request with model name and features

    Returns:
        Prediction response with predicted value

    Raises:
        HTTPException: If model loading or prediction fails
    """
    trace_id = trace_id_var.get(None)

    logger.info(
        "Prediction request received",
        model_name=request.model_name,
        model_version=request.model_version,
        trace_id=trace_id,
    )

    try:
        # Load model
        model_loader = get_model_loader()
        model = await model_loader.load_model_async(
            model_name=request.model_name,
            model_version=request.model_version,
        )

        # Prepare features
        X = _prepare_features(request.features)

        # Make prediction
        predictions = model.predict(X)

        # Extract single prediction value
        prediction_value = float(predictions[0])

        logger.info(
            "Prediction successful",
            model_name=request.model_name,
            model_version=request.model_version,
            prediction=prediction_value,
            trace_id=trace_id,
        )

        # Create response
        response = PredictionResponse(
            prediction=prediction_value,
            confidence=None,  # TODO: Add confidence scoring
            model_name=request.model_name,
            model_version=request.model_version,
            timestamp=datetime.utcnow(),
            trace_id=trace_id,
        )

        return response

    except ValueError as e:
        logger.error(
            "Prediction failed - invalid input",
            model_name=request.model_name,
            error=str(e),
            trace_id=trace_id,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        logger.error(
            "Prediction failed - internal error",
            model_name=request.model_name,
            error=str(e),
            trace_id=trace_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/batch_predict",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Make batch predictions",
    description="Make predictions for multiple feature sets (max 1000)",
    dependencies=[Depends(get_api_key_dependency())],
)
async def batch_predict(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Make batch predictions.

    Args:
        request: Batch prediction request with model name and features list

    Returns:
        Batch prediction response with list of predictions

    Raises:
        HTTPException: If model loading or prediction fails
    """
    trace_id = trace_id_var.get(None)

    logger.info(
        "Batch prediction request received",
        model_name=request.model_name,
        model_version=request.model_version,
        batch_size=len(request.features_list),
        trace_id=trace_id,
    )

    try:
        # Load model
        model_loader = get_model_loader()
        model = await model_loader.load_model_async(
            model_name=request.model_name,
            model_version=request.model_version,
        )

        # Prepare features as DataFrame
        X = pd.DataFrame(request.features_list)

        # Make predictions
        predictions = model.predict(X)

        # Convert to list of floats
        predictions_list = [float(p) for p in predictions]

        logger.info(
            "Batch prediction successful",
            model_name=request.model_name,
            model_version=request.model_version,
            batch_size=len(predictions_list),
            trace_id=trace_id,
        )

        # Create response
        response = BatchPredictionResponse(
            predictions=predictions_list,
            model_name=request.model_name,
            model_version=request.model_version,
            timestamp=datetime.utcnow(),
            trace_id=trace_id,
            count=len(predictions_list),
        )

        return response

    except ValueError as e:
        logger.error(
            "Batch prediction failed - invalid input",
            model_name=request.model_name,
            error=str(e),
            trace_id=trace_id,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        logger.error(
            "Batch prediction failed - internal error",
            model_name=request.model_name,
            error=str(e),
            trace_id=trace_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )
