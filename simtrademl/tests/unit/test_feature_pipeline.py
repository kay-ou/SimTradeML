"""Test feature engineering transformers."""
import numpy as np
import pandas as pd

from simtrademl.features.engineer import (
    FeaturePipeline,
    LagFeatureTransformer,
    RollingFeatureTransformer,
)


def test_feature_pipeline() -> None:
    """Test FeaturePipeline with cache."""
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        "close": np.random.randn(100).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 100),
    })

    # Create pipeline
    pipeline = FeaturePipeline(
        transformers=[
            RollingFeatureTransformer(
                columns=["close"],
                windows=[5, 20],
                stats=["mean", "std"],
            ),
            LagFeatureTransformer(
                columns=["close"],
                lags=[1, 5],
            ),
        ],
        cache_enabled=True,
    )

    # Transform with caching
    df_result = pipeline.fit_transform(df, cache_key="test_features")

    # Verify features were added
    assert df_result.shape[1] > df.shape[1], "Features should be added"
    assert "close_rolling_mean_5" in df_result.columns
    assert "close_rolling_std_20" in df_result.columns
    assert "close_lag_1" in df_result.columns
    assert "close_lag_5" in df_result.columns

    # Test cache
    df_cached = pipeline.fit_transform(df, cache_key="test_features")
    assert df_result.equals(df_cached), "Cached result should be identical"

    # Clear cache
    pipeline.clear_cache()

    print("✓ FeaturePipeline test passed")
    print(f"  Original columns: {df.shape[1]}")
    print(f"  After pipeline: {df_result.shape[1]}")
    print(f"  Features added: {df_result.shape[1] - df.shape[1]}")


if __name__ == "__main__":
    test_feature_pipeline()
