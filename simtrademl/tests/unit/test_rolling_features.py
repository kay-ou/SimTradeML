"""Unit tests for feature engineering transformers."""

import numpy as np
import pandas as pd
import pytest

from simtrademl.features.engineer import (
    FeaturePipeline,
    FeatureTransformer,
    LagFeatureTransformer,
    RollingFeatureTransformer,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample dataframe for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            "date": pd.date_range("2024-01-01", periods=10),
        }
    )


class TestRollingFeatureTransformer:
    """Tests for RollingFeatureTransformer."""

    def test_init(self) -> None:
        """Test initialization of RollingFeatureTransformer."""
        transformer = RollingFeatureTransformer(
            columns=["close"], windows=[5], stats=["mean"]
        )
        assert transformer.columns == ["close"]
        assert transformer.windows == [5]
        assert transformer.stats == ["mean"]

    def test_init_with_invalid_stats(self) -> None:
        """Test initialization with invalid stats raises error."""
        with pytest.raises(ValueError, match="Invalid stats"):
            RollingFeatureTransformer(
                columns=["close"], windows=[5], stats=["invalid_stat"]
            )

    def test_transform_rolling_mean(self, sample_dataframe: pd.DataFrame) -> None:
        """Test rolling mean calculation."""
        transformer = RollingFeatureTransformer(
            columns=["close"], windows=[3], stats=["mean"]
        )
        result = transformer.transform(sample_dataframe)

        assert "close_rolling_mean_3" in result.columns
        # First 2 values should be NaN
        assert pd.isna(result["close_rolling_mean_3"].iloc[0])
        assert pd.isna(result["close_rolling_mean_3"].iloc[1])
        # Third value should be mean of first 3
        assert result["close_rolling_mean_3"].iloc[2] == pytest.approx(101.0)

    def test_transform_rolling_std(self, sample_dataframe: pd.DataFrame) -> None:
        """Test rolling standard deviation calculation."""
        transformer = RollingFeatureTransformer(
            columns=["close"], windows=[3], stats=["std"]
        )
        result = transformer.transform(sample_dataframe)

        assert "close_rolling_std_3" in result.columns
        # Check that std is calculated
        assert not pd.isna(result["close_rolling_std_3"].iloc[2])

    def test_transform_rolling_max_min(self, sample_dataframe: pd.DataFrame) -> None:
        """Test rolling max and min calculation."""
        transformer = RollingFeatureTransformer(
            columns=["close"], windows=[3], stats=["max", "min"]
        )
        result = transformer.transform(sample_dataframe)

        assert "close_rolling_max_3" in result.columns
        assert "close_rolling_min_3" in result.columns
        # Check values
        assert result["close_rolling_max_3"].iloc[2] == 102.0
        assert result["close_rolling_min_3"].iloc[2] == 100.0

    def test_transform_rolling_sum(self, sample_dataframe: pd.DataFrame) -> None:
        """Test rolling sum calculation."""
        transformer = RollingFeatureTransformer(
            columns=["close"], windows=[3], stats=["sum"]
        )
        result = transformer.transform(sample_dataframe)

        assert "close_rolling_sum_3" in result.columns
        assert result["close_rolling_sum_3"].iloc[2] == pytest.approx(303.0)

    def test_transform_rolling_median(self, sample_dataframe: pd.DataFrame) -> None:
        """Test rolling median calculation."""
        transformer = RollingFeatureTransformer(
            columns=["close"], windows=[3], stats=["median"]
        )
        result = transformer.transform(sample_dataframe)

        assert "close_rolling_median_3" in result.columns
        assert result["close_rolling_median_3"].iloc[2] == 101.0

    def test_transform_multiple_windows(self, sample_dataframe: pd.DataFrame) -> None:
        """Test rolling features with multiple windows."""
        transformer = RollingFeatureTransformer(
            columns=["close"], windows=[3, 5], stats=["mean"]
        )
        result = transformer.transform(sample_dataframe)

        assert "close_rolling_mean_3" in result.columns
        assert "close_rolling_mean_5" in result.columns

    def test_transform_multiple_columns(self, sample_dataframe: pd.DataFrame) -> None:
        """Test rolling features for multiple columns."""
        transformer = RollingFeatureTransformer(
            columns=["close", "volume"], windows=[3], stats=["mean"]
        )
        result = transformer.transform(sample_dataframe)

        assert "close_rolling_mean_3" in result.columns
        assert "volume_rolling_mean_3" in result.columns

    def test_transform_multiple_stats(self, sample_dataframe: pd.DataFrame) -> None:
        """Test rolling features with multiple statistics."""
        transformer = RollingFeatureTransformer(
            columns=["close"], windows=[3], stats=["mean", "std", "max", "min"]
        )
        result = transformer.transform(sample_dataframe)

        assert "close_rolling_mean_3" in result.columns
        assert "close_rolling_std_3" in result.columns
        assert "close_rolling_max_3" in result.columns
        assert "close_rolling_min_3" in result.columns

    def test_transform_missing_column(self, sample_dataframe: pd.DataFrame) -> None:
        """Test transformation with missing column."""
        transformer = RollingFeatureTransformer(
            columns=["nonexistent"], windows=[3], stats=["mean"]
        )
        result = transformer.transform(sample_dataframe)

        # Should not create feature for missing column
        assert "nonexistent_rolling_mean_3" not in result.columns
        # Original columns should remain
        assert len(result.columns) == len(sample_dataframe.columns)

    def test_transform_does_not_modify_original(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test that transformation doesn't modify original dataframe."""
        original_cols = sample_dataframe.columns.tolist()
        transformer = RollingFeatureTransformer(
            columns=["close"], windows=[3], stats=["mean"]
        )
        result = transformer.transform(sample_dataframe)

        assert sample_dataframe.columns.tolist() == original_cols
        assert len(result.columns) > len(sample_dataframe.columns)


class TestLagFeatureTransformer:
    """Tests for LagFeatureTransformer."""

    def test_init(self) -> None:
        """Test initialization of LagFeatureTransformer."""
        transformer = LagFeatureTransformer(columns=["close"], lags=[1, 5])
        assert transformer.columns == ["close"]
        assert transformer.lags == [1, 5]

    def test_transform_single_lag(self, sample_dataframe: pd.DataFrame) -> None:
        """Test lag feature with single lag period."""
        transformer = LagFeatureTransformer(columns=["close"], lags=[1])
        result = transformer.transform(sample_dataframe)

        assert "close_lag_1" in result.columns
        # First value should be NaN
        assert pd.isna(result["close_lag_1"].iloc[0])
        # Second value should be first value of close
        assert result["close_lag_1"].iloc[1] == 100.0
        assert result["close_lag_1"].iloc[2] == 101.0

    def test_transform_multiple_lags(self, sample_dataframe: pd.DataFrame) -> None:
        """Test lag features with multiple lag periods."""
        transformer = LagFeatureTransformer(columns=["close"], lags=[1, 2, 5])
        result = transformer.transform(sample_dataframe)

        assert "close_lag_1" in result.columns
        assert "close_lag_2" in result.columns
        assert "close_lag_5" in result.columns

        # Check lag_2
        assert pd.isna(result["close_lag_2"].iloc[1])
        assert result["close_lag_2"].iloc[2] == 100.0

        # Check lag_5
        assert pd.isna(result["close_lag_5"].iloc[4])
        assert result["close_lag_5"].iloc[5] == 100.0

    def test_transform_multiple_columns(self, sample_dataframe: pd.DataFrame) -> None:
        """Test lag features for multiple columns."""
        transformer = LagFeatureTransformer(columns=["close", "volume"], lags=[1])
        result = transformer.transform(sample_dataframe)

        assert "close_lag_1" in result.columns
        assert "volume_lag_1" in result.columns

    def test_transform_missing_column(self, sample_dataframe: pd.DataFrame) -> None:
        """Test transformation with missing column."""
        transformer = LagFeatureTransformer(columns=["nonexistent"], lags=[1])
        result = transformer.transform(sample_dataframe)

        # Should not create feature for missing column
        assert "nonexistent_lag_1" not in result.columns

    def test_transform_preserves_original_data(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test that transformation preserves original data."""
        transformer = LagFeatureTransformer(columns=["close"], lags=[1])
        result = transformer.transform(sample_dataframe)

        # Original column should be unchanged
        pd.testing.assert_series_equal(result["close"], sample_dataframe["close"])


class TestFeaturePipeline:
    """Tests for FeaturePipeline."""

    def test_init(self) -> None:
        """Test initialization of FeaturePipeline."""
        transformers = [
            RollingFeatureTransformer(columns=["close"], windows=[3], stats=["mean"]),
            LagFeatureTransformer(columns=["close"], lags=[1]),
        ]
        pipeline = FeaturePipeline(transformers=transformers)

        assert len(pipeline.transformers) == 2
        assert pipeline.cache_enabled is False

    def test_fit_transform_single_transformer(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test pipeline with single transformer."""
        pipeline = FeaturePipeline(
            transformers=[
                RollingFeatureTransformer(
                    columns=["close"], windows=[3], stats=["mean"]
                )
            ]
        )
        result = pipeline.fit_transform(sample_dataframe)

        assert "close_rolling_mean_3" in result.columns

    def test_fit_transform_multiple_transformers(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test pipeline with multiple transformers."""
        pipeline = FeaturePipeline(
            transformers=[
                RollingFeatureTransformer(
                    columns=["close"], windows=[3], stats=["mean"]
                ),
                LagFeatureTransformer(columns=["close"], lags=[1]),
            ]
        )
        result = pipeline.fit_transform(sample_dataframe)

        # Both transformers should have been applied
        assert "close_rolling_mean_3" in result.columns
        assert "close_lag_1" in result.columns

    def test_fit_transform_with_cache(self, sample_dataframe: pd.DataFrame) -> None:
        """Test pipeline with caching enabled."""
        pipeline = FeaturePipeline(
            transformers=[
                RollingFeatureTransformer(
                    columns=["close"], windows=[3], stats=["mean"]
                )
            ],
            cache_enabled=True,
        )

        # First call should compute features
        result1 = pipeline.fit_transform(sample_dataframe, cache_key="test")

        # Second call should return cached result
        result2 = pipeline.fit_transform(sample_dataframe, cache_key="test")

        pd.testing.assert_frame_equal(result1, result2)

    def test_clear_cache(self, sample_dataframe: pd.DataFrame) -> None:
        """Test clearing the cache."""
        pipeline = FeaturePipeline(
            transformers=[
                RollingFeatureTransformer(
                    columns=["close"], windows=[3], stats=["mean"]
                )
            ],
            cache_enabled=True,
        )

        # Cache some results
        pipeline.fit_transform(sample_dataframe, cache_key="test")
        assert len(pipeline._cache) == 1

        # Clear cache
        pipeline.clear_cache()
        assert len(pipeline._cache) == 0

    def test_fit_transform_without_cache_key(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test pipeline without cache key doesn't cache."""
        pipeline = FeaturePipeline(
            transformers=[
                RollingFeatureTransformer(
                    columns=["close"], windows=[3], stats=["mean"]
                )
            ],
            cache_enabled=True,
        )

        # Call without cache_key
        pipeline.fit_transform(sample_dataframe)

        # Cache should be empty
        assert len(pipeline._cache) == 0


@pytest.mark.unit
class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering."""

    def test_complete_feature_engineering_workflow(self) -> None:
        """Test complete feature engineering workflow."""
        # Create larger sample data
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 100),
                "date": pd.date_range("2024-01-01", periods=100),
            }
        )

        # Create pipeline with multiple transformers
        pipeline = FeaturePipeline(
            transformers=[
                RollingFeatureTransformer(
                    columns=["close", "volume"],
                    windows=[5, 20],
                    stats=["mean", "std", "max", "min"],
                ),
                LagFeatureTransformer(columns=["close"], lags=[1, 5, 10]),
            ],
            cache_enabled=True,
        )

        # Transform data
        result = pipeline.fit_transform(df, cache_key="full_features")

        # Verify features were created
        # Original: 3 columns
        # Rolling: 2 columns * 2 windows * 4 stats = 16 features
        # Lag: 1 column * 3 lags = 3 features
        # Total: 3 + 16 + 3 = 22 columns
        assert len(result.columns) == 22

        # Check specific features exist
        assert "close_rolling_mean_5" in result.columns
        assert "close_rolling_std_20" in result.columns
        assert "volume_rolling_max_5" in result.columns
        assert "close_lag_1" in result.columns
        assert "close_lag_10" in result.columns

    def test_feature_transformer_is_abstract(self) -> None:
        """Test that FeatureTransformer cannot be instantiated."""
        with pytest.raises(TypeError):
            FeatureTransformer()  # type: ignore
