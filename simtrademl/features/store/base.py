"""Feature store abstract base class.

This module defines the abstract interface for feature stores, providing
a contract for batch and online feature storage and retrieval.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

import pandas as pd


class FeatureStore(ABC):
    """Abstract feature store interface.

    This abstract base class defines the contract for feature store implementations.
    Feature stores provide:
    - Batch feature storage for training data
    - Batch feature retrieval with time-range filtering
    - Online feature retrieval for low-latency inference

    Implementations must provide concrete implementations of all abstract methods.
    """

    @abstractmethod
    def write_batch(
        self,
        feature_view: str,
        features: pd.DataFrame,
        entity_column: str = "symbol",
        timestamp_column: str = "timestamp",
    ) -> None:
        """Write batch features to store.

        Stores features for a specific feature view. Features are organized by
        entity (e.g., stock symbol) and timestamp for efficient time-series queries.

        Args:
            feature_view: Feature view name (e.g., 'momentum_features', 'volatility_features').
                         This groups related features together.
            features: DataFrame containing features to store. Must include:
                     - entity_column: Entity identifier (e.g., stock symbol)
                     - timestamp_column: Timestamp for each feature row
                     - Feature columns: The actual feature values
            entity_column: Name of the entity identifier column in the DataFrame.
                          Defaults to 'symbol'.
            timestamp_column: Name of the timestamp column in the DataFrame.
                             Defaults to 'timestamp'.

        Returns:
            None

        Raises:
            ValueError: If required columns are missing or invalid
            FeatureStoreError: If write operation fails

        Example:
            >>> store = ConcreteFeatureStore(...)
            >>> features = pd.DataFrame({
            ...     'symbol': ['AAPL', 'AAPL', 'MSFT'],
            ...     'timestamp': [
            ...         datetime(2024, 1, 1),
            ...         datetime(2024, 1, 2),
            ...         datetime(2024, 1, 1)
            ...     ],
            ...     'rsi_14': [65.0, 70.0, 55.0],
            ...     'macd': [0.5, 0.8, -0.2]
            ... })
            >>> store.write_batch(
            ...     feature_view='momentum_features',
            ...     features=features
            ... )
        """
        pass

    @abstractmethod
    def read_batch(
        self,
        feature_view: str,
        entities: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Read batch features from store for a time range.

        Retrieves historical features for training or batch scoring. This method
        ensures point-in-time correctness: only features that existed at or before
        the requested timestamp are returned.

        Args:
            feature_view: Feature view name to read from
            entities: List of entity IDs to retrieve features for (e.g., ['AAPL', 'MSFT'])
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)

        Returns:
            DataFrame containing requested features with columns:
            - entity column (e.g., 'symbol')
            - timestamp column
            - Feature columns from the feature view

        Raises:
            FeatureNotFoundError: If feature view doesn't exist
            FeatureStoreError: If read operation fails

        Example:
            >>> store = ConcreteFeatureStore(...)
            >>> features = store.read_batch(
            ...     feature_view='momentum_features',
            ...     entities=['AAPL', 'MSFT'],
            ...     start_time=datetime(2024, 1, 1),
            ...     end_time=datetime(2024, 1, 31)
            ... )
            >>> print(features.shape)
            (60, 5)  # 30 days * 2 symbols, with symbol, timestamp, and 3 features
        """
        pass

    @abstractmethod
    def read_online(
        self,
        feature_view: str,
        entities: List[str],
    ) -> pd.DataFrame:
        """Read latest features for online inference.

        Retrieves the most recent features for each entity for low-latency inference.
        Implementations should use caching (e.g., Redis) to minimize latency.

        Args:
            feature_view: Feature view name to read from
            entities: List of entity IDs to retrieve latest features for

        Returns:
            DataFrame containing latest features with columns:
            - entity column (e.g., 'symbol')
            - timestamp column (timestamp of latest feature)
            - Feature columns from the feature view

        Raises:
            FeatureNotFoundError: If feature view doesn't exist or no features found
            FeatureStoreError: If read operation fails

        Example:
            >>> store = ConcreteFeatureStore(...)
            >>> # Get latest features for real-time prediction
            >>> features = store.read_online(
            ...     feature_view='momentum_features',
            ...     entities=['AAPL']
            ... )
            >>> print(features)
               symbol           timestamp  rsi_14  macd  macd_signal
            0   AAPL 2024-01-31 16:00:00    72.5   1.2          0.8

        Notes:
            - This method should prioritize low latency over freshness
            - Implementations should cache results (e.g., in Redis) with appropriate TTL
            - If cache misses, fall back to database and update cache
        """
        pass
