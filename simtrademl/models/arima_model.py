"""ARIMA time series model implementation using statsmodels.

Provides a wrapper around statsmodels ARIMA for time series forecasting.
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from simtrademl.models.base import BaseModel
from simtrademl.utils.logging import get_logger

logger = get_logger(__name__)


class ARIMAModel(BaseModel):
    """ARIMA (AutoRegressive Integrated Moving Average) model.

    Suitable for univariate time series forecasting.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        trend: Optional[str] = None,
    ):
        """Initialize ARIMA model.

        Args:
            order: The (p,d,q) order of the model for the AR, I, MA parameters
            seasonal_order: The (P,D,Q,s) order of the seasonal component
            trend: Parameter controlling the deterministic trend.
                   Options: 'n', 'c', 't', 'ct' for no trend, constant, linear,
                   constant with linear trend.
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model = None
        self.fitted_model = None

        logger.info(
            "ARIMAModel initialized",
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, pd.Series],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> None:
        """Fit ARIMA model to time series data.

        Args:
            X: Time series data (if y is None, X is used as the series)
            y: Target series (if provided, X is ignored)
        """
        # ARIMA is univariate, use y if provided, otherwise X
        if y is not None:
            if isinstance(y, pd.Series):
                series = y
            else:
                series = pd.Series(y)
        else:
            if isinstance(X, pd.DataFrame):
                if X.shape[1] != 1:
                    raise ValueError(
                        "ARIMA requires univariate data. "
                        f"Got {X.shape[1]} columns."
                    )
                series = X.iloc[:, 0]
            elif isinstance(X, pd.Series):
                series = X
            else:
                series = pd.Series(X)

        logger.info(
            "Fitting ARIMA model",
            n_observations=len(series),
            order=self.order,
        )

        # Create and fit ARIMA model
        self.model = ARIMA(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
        )
        self.fitted_model = self.model.fit()

        logger.info(
            "ARIMA model fitted",
            aic=round(self.fitted_model.aic, 2),
            bic=round(self.fitted_model.bic, 2),
        )

    def predict(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        steps: Optional[int] = None,
    ) -> np.ndarray:
        """Make predictions using the fitted ARIMA model.

        Args:
            X: If provided, determines number of steps (len(X)) to forecast
            steps: Number of steps ahead to forecast (overrides X if provided)

        Returns:
            Array of predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        # Determine number of steps
        if steps is None:
            if X is not None:
                steps = len(X)
            else:
                steps = 1

        logger.debug("Making ARIMA predictions", steps=steps)

        # Forecast future values
        forecast = self.fitted_model.forecast(steps=steps)

        return forecast.values if isinstance(forecast, pd.Series) else forecast

    def save(self, path: Union[str, Path]) -> None:
        """Save ARIMA model to disk.

        Args:
            path: Path to save the model
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save fitted model using pickle
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "fitted_model": self.fitted_model,
                    "order": self.order,
                    "seasonal_order": self.seasonal_order,
                    "trend": self.trend,
                },
                f,
            )

        logger.info("ARIMA model saved", path=str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ARIMAModel":
        """Load ARIMA model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded ARIMAModel instance
        """
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        model = cls(
            order=data["order"],
            seasonal_order=data["seasonal_order"],
            trend=data["trend"],
        )
        model.fitted_model = data["fitted_model"]

        logger.info("ARIMA model loaded", path=str(path))

        return model


# Example usage
if __name__ == "__main__":
    from simtrademl.utils.logging import configure_logging

    configure_logging(log_level="INFO", log_format="console")

    # Create sample time series data
    np.random.seed(42)
    n = 100
    t = np.arange(n)
    trend = 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(n) * 3
    series = trend + seasonal + noise + 100

    print("=== ARIMA Model Example ===")
    print(f"Time series length: {len(series)}")

    # Initialize and fit model
    model = ARIMAModel(order=(2, 1, 2))
    model.fit(pd.Series(series))

    # Make predictions
    forecast = model.predict(steps=10)
    print(f"\n10-step forecast: {forecast}")

    # Save and load model
    model.save("/tmp/arima_model.pkl")
    loaded_model = ARIMAModel.load("/tmp/arima_model.pkl")
    forecast2 = loaded_model.predict(steps=10)
    print(f"Forecast from loaded model: {forecast2}")
    print(f"Forecasts match: {np.allclose(forecast, forecast2)}")
