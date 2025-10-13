"""Prophet time series forecasting model.

Provides a wrapper around Facebook Prophet for time series forecasting.
Note: Prophet requires specific data format with 'ds' and 'y' columns.
"""

import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from simtrademl.models.base import BaseModel
from simtrademl.utils.logging import get_logger

logger = get_logger(__name__)


class ProphetModel(BaseModel):
    """Prophet model for time series forecasting.

    Prophet is designed for forecasting time series data with strong
    seasonal patterns and several seasons of historical data.
    """

    def __init__(
        self,
        growth: str = "linear",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        seasonality_mode: str = "additive",
        daily_seasonality: bool = False,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
    ):
        """Initialize Prophet model.

        Args:
            growth: 'linear' or 'logistic' growth
            changepoint_prior_scale: Flexibility of trend changes
            seasonality_prior_scale: Strength of seasonality
            seasonality_mode: 'additive' or 'multiplicative'
            daily_seasonality: Fit daily seasonality
            weekly_seasonality: Fit weekly seasonality
            yearly_seasonality: Fit yearly seasonality
        """
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.model = None

        logger.info(
            "ProphetModel initialized",
            growth=growth,
            seasonality_mode=seasonality_mode,
        )

    def _prepare_dataframe(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Prepare dataframe in Prophet format (ds, y columns).

        Args:
            X: Features (should contain datetime index or 'ds' column)
            y: Target values

        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        if y is not None:
            # Use y as target
            if isinstance(X, pd.DataFrame):
                if "ds" in X.columns:
                    ds = X["ds"]
                elif isinstance(X.index, pd.DatetimeIndex):
                    ds = X.index
                else:
                    raise ValueError(
                        "X must have datetime index or 'ds' column"
                    )
            else:
                raise ValueError(
                    "When y is provided, X must be DataFrame with dates"
                )

            df = pd.DataFrame({"ds": ds, "y": y})
        else:
            # X should already be in Prophet format
            if isinstance(X, pd.DataFrame):
                if "ds" in X.columns and "y" in X.columns:
                    df = X[["ds", "y"]].copy()
                elif isinstance(X.index, pd.DatetimeIndex) and X.shape[1] == 1:
                    df = pd.DataFrame({
                        "ds": X.index,
                        "y": X.iloc[:, 0],
                    })
                else:
                    raise ValueError(
                        "X must have 'ds' and 'y' columns or "
                        "datetime index with single value column"
                    )
            else:
                raise ValueError("X must be DataFrame for Prophet")

        return df

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> None:
        """Fit Prophet model.

        Args:
            X: Training data (with datetime index/column)
            y: Target values (optional if X has 'y' column)
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError(
                "Prophet is not installed. Install it with: "
                "pip install prophet"
            )

        df = self._prepare_dataframe(X, y)

        logger.info(
            "Fitting Prophet model",
            n_observations=len(df),
            date_range=f"{df['ds'].min()} to {df['ds'].max()}",
        )

        # Create and fit Prophet model
        self.model = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
        )

        self.model.fit(df)

        logger.info("Prophet model fitted successfully")

    def predict(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        periods: int = 30,
        freq: str = "D",
    ) -> np.ndarray:
        """Make predictions using Prophet.

        Args:
            X: Future dates (optional, will create if not provided)
            periods: Number of periods to forecast (if X not provided)
            freq: Frequency of periods ('D', 'W', 'M', etc.)

        Returns:
            Array of predictions (yhat values)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        # Create future dataframe if not provided
        if X is None:
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
        else:
            if isinstance(X, pd.DataFrame):
                if "ds" not in X.columns:
                    if isinstance(X.index, pd.DatetimeIndex):
                        future = pd.DataFrame({"ds": X.index})
                    else:
                        raise ValueError("X must have 'ds' column or datetime index")
                else:
                    future = X[["ds"]].copy()
            else:
                raise ValueError("X must be DataFrame with dates")

        logger.debug("Making Prophet predictions", n_periods=len(future))

        # Make predictions
        forecast = self.model.predict(future)

        return forecast["yhat"].values

    def save(self, path: Union[str, Path]) -> None:
        """Save Prophet model to disk.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be fitted before saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model configuration and fitted model
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "growth": self.growth,
                    "changepoint_prior_scale": self.changepoint_prior_scale,
                    "seasonality_prior_scale": self.seasonality_prior_scale,
                    "seasonality_mode": self.seasonality_mode,
                    "daily_seasonality": self.daily_seasonality,
                    "weekly_seasonality": self.weekly_seasonality,
                    "yearly_seasonality": self.yearly_seasonality,
                },
                f,
            )

        logger.info("Prophet model saved", path=str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ProphetModel":
        """Load Prophet model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded ProphetModel instance
        """
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        model = cls(
            growth=data["growth"],
            changepoint_prior_scale=data["changepoint_prior_scale"],
            seasonality_prior_scale=data["seasonality_prior_scale"],
            seasonality_mode=data["seasonality_mode"],
            daily_seasonality=data["daily_seasonality"],
            weekly_seasonality=data["weekly_seasonality"],
            yearly_seasonality=data["yearly_seasonality"],
        )
        model.model = data["model"]

        logger.info("Prophet model loaded", path=str(path))

        return model


# Example usage
if __name__ == "__main__":
    from simtrademl.utils.logging import configure_logging

    configure_logging(log_level="INFO", log_format="console")

    print("=== Prophet Model Example ===")
    print("\nNote: Prophet requires installation: pip install prophet")
    print("Example code demonstrates the API, but requires prophet package.")

    # # Create sample time series data with dates
    # dates = pd.date_range("2023-01-01", periods=365, freq="D")
    # values = (
    #     np.sin(2 * np.pi * np.arange(365) / 365) * 10
    #     + np.random.randn(365) * 2
    #     + 100
    # )
    # df = pd.DataFrame({"ds": dates, "y": values})
    #
    # print(f"Time series length: {len(df)}")
    #
    # # Initialize and fit model
    # model = ProphetModel()
    # model.fit(df)
    #
    # # Make predictions
    # forecast = model.predict(periods=30)
    # print(f"\n30-day forecast (first 5): {forecast[:5]}")
    #
    # # Save and load model
    # model.save("/tmp/prophet_model.pkl")
    # loaded_model = ProphetModel.load("/tmp/prophet_model.pkl")
    # print("Model saved and loaded successfully")
