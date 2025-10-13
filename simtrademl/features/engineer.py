"""Feature engineering module for transforming raw data into ML features.

Provides various transformers for creating features from time series data.
"""

from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from simtrademl.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureTransformer(ABC):
    """Abstract base class for feature transformers.

    All feature transformers must implement the transform method.
    """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe to add features.

        Args:
            df: Input dataframe

        Returns:
            Transformed dataframe with additional features
        """
        pass


class RollingFeatureTransformer(FeatureTransformer):
    """Compute rolling window statistics.

    Calculates rolling mean, std, max, min, sum for specified columns and windows.
    """

    def __init__(
        self,
        columns: List[str],
        windows: List[int],
        stats: List[str] = ["mean", "std"],
    ):
        """Initialize rolling feature transformer.

        Args:
            columns: List of column names to compute rolling features for
            windows: List of window sizes
            stats: List of statistics to compute.
                   Options: 'mean', 'std', 'max', 'min', 'sum', 'median'
        """
        self.columns = columns
        self.windows = windows
        self.stats = stats

        valid_stats = {"mean", "std", "max", "min", "sum", "median"}
        invalid_stats = set(stats) - valid_stats
        if invalid_stats:
            raise ValueError(
                f"Invalid stats: {invalid_stats}. "
                f"Valid options: {valid_stats}"
            )

        logger.info(
            "RollingFeatureTransformer initialized",
            columns=columns,
            windows=windows,
            stats=stats,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling features.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with rolling features added
        """
        df = df.copy()

        for col in self.columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in dataframe, skipping")
                continue

            for window in self.windows:
                rolling = df[col].rolling(window=window)

                if "mean" in self.stats:
                    feature_name = f"{col}_rolling_mean_{window}"
                    df[feature_name] = rolling.mean()

                if "std" in self.stats:
                    feature_name = f"{col}_rolling_std_{window}"
                    df[feature_name] = rolling.std()

                if "max" in self.stats:
                    feature_name = f"{col}_rolling_max_{window}"
                    df[feature_name] = rolling.max()

                if "min" in self.stats:
                    feature_name = f"{col}_rolling_min_{window}"
                    df[feature_name] = rolling.min()

                if "sum" in self.stats:
                    feature_name = f"{col}_rolling_sum_{window}"
                    df[feature_name] = rolling.sum()

                if "median" in self.stats:
                    feature_name = f"{col}_rolling_median_{window}"
                    df[feature_name] = rolling.median()

        # Count added features
        original_cols = len([c for c in df.columns if not c.startswith(tuple(self.columns))])
        added_features = len(df.columns) - original_cols

        logger.info(
            "Rolling features computed",
            added_features=added_features,
            total_columns=len(df.columns),
        )

        return df


class LagFeatureTransformer(FeatureTransformer):
    """Create lag features for time series data.

    Generates lagged versions of specified columns.
    """

    def __init__(self, columns: List[str], lags: List[int]):
        """Initialize lag feature transformer.

        Args:
            columns: List of column names to create lags for
            lags: List of lag periods
        """
        self.columns = columns
        self.lags = lags

        logger.info(
            "LagFeatureTransformer initialized",
            columns=columns,
            lags=lags,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with lag features added
        """
        df = df.copy()

        for col in self.columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in dataframe, skipping")
                continue

            for lag in self.lags:
                feature_name = f"{col}_lag_{lag}"
                df[feature_name] = df[col].shift(lag)

        added_features = len(self.columns) * len(self.lags)
        logger.info(
            "Lag features created",
            added_features=added_features,
            total_columns=len(df.columns),
        )

        return df


class FeaturePipeline:
    """Chain multiple feature transformers.

    Executes transformers in sequence and optionally caches results.
    """

    def __init__(
        self,
        transformers: List[FeatureTransformer],
        cache_enabled: bool = False,
    ):
        """Initialize feature pipeline.

        Args:
            transformers: List of feature transformers to apply
            cache_enabled: Whether to enable feature caching
        """
        self.transformers = transformers
        self.cache_enabled = cache_enabled
        self._cache: dict = {}

        logger.info(
            "FeaturePipeline initialized",
            num_transformers=len(transformers),
            cache_enabled=cache_enabled,
        )

    def fit_transform(self, df: pd.DataFrame, cache_key: str = None) -> pd.DataFrame:
        """Apply all transformers in sequence.

        Args:
            df: Input dataframe
            cache_key: Optional key for caching results

        Returns:
            Transformed dataframe
        """
        # Check cache if enabled
        if self.cache_enabled and cache_key and cache_key in self._cache:
            logger.info("Returning cached features", cache_key=cache_key)
            return self._cache[cache_key].copy()

        result = df.copy()

        for i, transformer in enumerate(self.transformers):
            transformer_name = transformer.__class__.__name__
            logger.debug(
                f"Applying transformer {i+1}/{len(self.transformers)}",
                transformer=transformer_name,
            )
            result = transformer.transform(result)

        # Cache results if enabled
        if self.cache_enabled and cache_key:
            self._cache[cache_key] = result.copy()
            logger.info("Features cached", cache_key=cache_key)

        logger.info(
            "Feature pipeline completed",
            input_columns=len(df.columns),
            output_columns=len(result.columns),
            added_features=len(result.columns) - len(df.columns),
        )

        return result

    def clear_cache(self) -> None:
        """Clear feature cache."""
        self._cache.clear()
        logger.info("Feature cache cleared")


# Example usage
if __name__ == "__main__":
    import numpy as np
    from simtrademl.utils.logging import configure_logging

    configure_logging(log_level="INFO", log_format="console")

    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        "close": np.random.randn(100).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 100),
        "date": pd.date_range("2024-01-01", periods=100),
    })

    print("=== Original Data ===")
    print(df.head())
    print(f"Shape: {df.shape}")

    # Example 1: Rolling features
    print("\n=== Rolling Features ===")
    rolling_transformer = RollingFeatureTransformer(
        columns=["close", "volume"],
        windows=[5, 20],
        stats=["mean", "std", "max", "min"],
    )
    df_rolling = rolling_transformer.transform(df)
    print(df_rolling[["close", "close_rolling_mean_5", "close_rolling_std_20"]].head(25))
    print(f"Shape: {df_rolling.shape}")

    # Example 2: Lag features
    print("\n=== Lag Features ===")
    lag_transformer = LagFeatureTransformer(
        columns=["close"],
        lags=[1, 5, 10],
    )
    df_lag = lag_transformer.transform(df)
    print(df_lag[["close", "close_lag_1", "close_lag_5"]].head(15))

    # Example 3: Feature pipeline
    print("\n=== Feature Pipeline ===")
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

    df_pipeline = pipeline.fit_transform(df, cache_key="example_features")
    print(f"Features created: {df_pipeline.shape[1] - df.shape[1]}")
    print(df_pipeline.columns.tolist())

    # Test cache
    df_cached = pipeline.fit_transform(df, cache_key="example_features")
    print("\nCache working:", df_pipeline.equals(df_cached))
