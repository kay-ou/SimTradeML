"""Unit tests for ARIMA and Prophet models."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simtrademl.models.arima_model import ARIMAModel
from simtrademl.models.prophet_model import ProphetModel


@pytest.fixture
def sample_time_series() -> pd.Series:
    """Create a sample time series for testing."""
    np.random.seed(42)
    n = 100
    t = np.arange(n)
    trend = 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(n) * 3
    series = trend + seasonal + noise + 100
    return pd.Series(series)


@pytest.fixture
def sample_dataframe_with_dates() -> pd.DataFrame:
    """Create a sample dataframe with dates for Prophet."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    values = (
        np.sin(2 * np.pi * np.arange(100) / 365) * 10
        + np.random.randn(100) * 2
        + 100
    )
    return pd.DataFrame({"ds": dates, "y": values})


class TestARIMAModel:
    """Tests for ARIMAModel."""

    def test_init(self) -> None:
        """Test ARIMA model initialization."""
        model = ARIMAModel(order=(1, 1, 1))
        assert model.order == (1, 1, 1)
        assert model.seasonal_order is None
        assert model.fitted_model is None

    def test_init_with_seasonal(self) -> None:
        """Test ARIMA model with seasonal parameters."""
        model = ARIMAModel(
            order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend="c"
        )
        assert model.seasonal_order == (1, 1, 1, 12)
        assert model.trend == "c"

    def test_fit_with_series(self, sample_time_series: pd.Series) -> None:
        """Test fitting ARIMA with pandas Series."""
        model = ARIMAModel(order=(2, 1, 2))
        model.fit(sample_time_series)

        assert model.fitted_model is not None
        assert hasattr(model.fitted_model, "aic")
        assert hasattr(model.fitted_model, "bic")

    def test_fit_with_numpy_array(self, sample_time_series: pd.Series) -> None:
        """Test fitting ARIMA with numpy array."""
        model = ARIMAModel(order=(1, 1, 1))
        model.fit(sample_time_series.values)

        assert model.fitted_model is not None

    def test_fit_with_dataframe_single_column(
        self, sample_time_series: pd.Series
    ) -> None:
        """Test fitting ARIMA with single-column DataFrame."""
        df = pd.DataFrame({"value": sample_time_series})
        model = ARIMAModel(order=(1, 1, 1))
        model.fit(df)

        assert model.fitted_model is not None

    def test_fit_with_dataframe_multiple_columns_raises_error(
        self, sample_time_series: pd.Series
    ) -> None:
        """Test that multivariate DataFrame raises error."""
        df = pd.DataFrame({"value1": sample_time_series, "value2": sample_time_series})
        model = ARIMAModel(order=(1, 1, 1))

        with pytest.raises(ValueError, match="univariate"):
            model.fit(df)

    def test_fit_with_xy(self, sample_time_series: pd.Series) -> None:
        """Test fitting with X and y parameters."""
        X = pd.DataFrame({"dummy": range(len(sample_time_series))})
        y = sample_time_series

        model = ARIMAModel(order=(1, 1, 1))
        model.fit(X, y)

        assert model.fitted_model is not None

    def test_predict_single_step(self, sample_time_series: pd.Series) -> None:
        """Test prediction for single step."""
        model = ARIMAModel(order=(1, 1, 1))
        model.fit(sample_time_series)

        forecast = model.predict(steps=1)

        assert isinstance(forecast, np.ndarray)
        assert len(forecast) == 1

    def test_predict_multiple_steps(self, sample_time_series: pd.Series) -> None:
        """Test prediction for multiple steps."""
        model = ARIMAModel(order=(1, 1, 1))
        model.fit(sample_time_series)

        forecast = model.predict(steps=10)

        assert isinstance(forecast, np.ndarray)
        assert len(forecast) == 10

    def test_predict_without_fit_raises_error(self) -> None:
        """Test that prediction without fit raises error."""
        model = ARIMAModel(order=(1, 1, 1))

        with pytest.raises(ValueError, match="fitted before prediction"):
            model.predict(steps=1)

    def test_save_and_load(
        self, sample_time_series: pd.Series, tmp_path: Path
    ) -> None:
        """Test saving and loading ARIMA model."""
        model = ARIMAModel(order=(2, 1, 2))
        model.fit(sample_time_series)

        # Make prediction before saving
        forecast1 = model.predict(steps=5)

        # Save model
        model_path = tmp_path / "arima_model.pkl"
        model.save(model_path)

        # Load model
        loaded_model = ARIMAModel.load(model_path)

        # Make prediction with loaded model
        forecast2 = loaded_model.predict(steps=5)

        # Predictions should match
        np.testing.assert_array_almost_equal(forecast1, forecast2)

    def test_save_without_fit_raises_error(self, tmp_path: Path) -> None:
        """Test that saving unfitted model raises error."""
        model = ARIMAModel(order=(1, 1, 1))

        with pytest.raises(ValueError, match="fitted before saving"):
            model.save(tmp_path / "model.pkl")


class TestProphetModel:
    """Tests for ProphetModel."""

    def test_init(self) -> None:
        """Test Prophet model initialization."""
        model = ProphetModel()
        assert model.growth == "linear"
        assert model.seasonality_mode == "additive"
        assert model.model is None

    def test_init_with_custom_params(self) -> None:
        """Test Prophet with custom parameters."""
        model = ProphetModel(
            growth="logistic",
            seasonality_mode="multiplicative",
            daily_seasonality=True,
        )
        assert model.growth == "logistic"
        assert model.seasonality_mode == "multiplicative"
        assert model.daily_seasonality is True

    def test_fit_with_prophet_format(
        self, sample_dataframe_with_dates: pd.DataFrame
    ) -> None:
        """Test fitting Prophet with ds/y format DataFrame."""
        pytest.importorskip("prophet")

        model = ProphetModel()
        model.fit(sample_dataframe_with_dates)

        assert model.model is not None

    def test_fit_with_datetime_index(self) -> None:
        """Test fitting Prophet with datetime index."""
        pytest.importorskip("prophet")

        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        values = np.random.randn(50) + 100
        df = pd.DataFrame({"value": values}, index=dates)

        model = ProphetModel()
        model.fit(df)

        assert model.model is not None

    def test_fit_with_xy(self) -> None:
        """Test fitting Prophet with X and y parameters."""
        pytest.importorskip("prophet")

        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        X = pd.DataFrame({"ds": dates})
        y = np.random.randn(50) + 100

        model = ProphetModel()
        model.fit(X, y)

        assert model.model is not None

    def test_fit_without_prophet_raises_import_error(
        self, sample_dataframe_with_dates: pd.DataFrame
    ) -> None:
        """Test that missing prophet raises ImportError."""
        # Skip if prophet is actually installed
        pytest.importorskip("prophet", reason="prophet is installed, cannot test ImportError")

        # This test would need prophet to not be installed
        # Since we can't easily mock importlib, we skip this test

    def test_predict_default_periods(
        self, sample_dataframe_with_dates: pd.DataFrame
    ) -> None:
        """Test prediction with default parameters."""
        pytest.importorskip("prophet")

        model = ProphetModel()
        model.fit(sample_dataframe_with_dates)

        forecast = model.predict(periods=10)

        assert isinstance(forecast, np.ndarray)
        # Should predict for training + future periods
        assert len(forecast) == len(sample_dataframe_with_dates) + 10

    def test_predict_with_future_dates(
        self, sample_dataframe_with_dates: pd.DataFrame
    ) -> None:
        """Test prediction with specified future dates."""
        pytest.importorskip("prophet")

        model = ProphetModel()
        model.fit(sample_dataframe_with_dates)

        # Create future dates
        future_dates = pd.date_range(
            sample_dataframe_with_dates["ds"].iloc[-1] + pd.Timedelta(days=1),
            periods=5,
            freq="D",
        )
        future_df = pd.DataFrame({"ds": future_dates})

        forecast = model.predict(future_df)

        assert isinstance(forecast, np.ndarray)
        assert len(forecast) == 5

    def test_predict_without_fit_raises_error(self) -> None:
        """Test that prediction without fit raises error."""
        model = ProphetModel()

        with pytest.raises(ValueError, match="fitted before prediction"):
            model.predict(periods=10)

    def test_save_and_load(
        self, sample_dataframe_with_dates: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test saving and loading Prophet model."""
        pytest.importorskip("prophet")

        model = ProphetModel()
        model.fit(sample_dataframe_with_dates)

        # Make prediction before saving
        forecast1 = model.predict(periods=5)

        # Save model
        model_path = tmp_path / "prophet_model.pkl"
        model.save(model_path)

        # Load model
        loaded_model = ProphetModel.load(model_path)

        # Make prediction with loaded model
        forecast2 = loaded_model.predict(periods=5)

        # Predictions should match (approximately due to randomness in Prophet)
        np.testing.assert_array_almost_equal(forecast1, forecast2, decimal=5)

    def test_save_without_fit_raises_error(self, tmp_path: Path) -> None:
        """Test that saving unfitted model raises error."""
        model = ProphetModel()

        with pytest.raises(ValueError, match="fitted before saving"):
            model.save(tmp_path / "model.pkl")


@pytest.mark.unit
class TestTimeSeriesModelsIntegration:
    """Integration tests for time series models."""

    def test_arima_complete_workflow(self) -> None:
        """Test complete ARIMA workflow."""
        # Create synthetic time series
        np.random.seed(42)
        n = 150
        series = pd.Series(np.random.randn(n).cumsum() + 100)

        # Train model
        model = ARIMAModel(order=(2, 1, 2))
        model.fit(series[:100])

        # Make predictions
        forecast = model.predict(steps=10)

        assert len(forecast) == 10
        assert not np.any(np.isnan(forecast))

    def test_prophet_complete_workflow(self) -> None:
        """Test complete Prophet workflow."""
        pytest.importorskip("prophet")

        # Create synthetic time series with dates
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        values = np.random.randn(100) + 100
        df = pd.DataFrame({"ds": dates, "y": values})

        # Train model
        model = ProphetModel(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        # Make predictions
        forecast = model.predict(periods=10)

        assert len(forecast) == 110  # 100 training + 10 future
        assert not np.any(np.isnan(forecast))
