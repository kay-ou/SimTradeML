"""TimescaleDB + Redis feature store implementation.

This module provides a production-ready feature store implementation using:
- TimescaleDB for persistent feature storage with time-series optimization
- Redis for low-latency online feature caching
- Connection pooling for efficient database access
- Automatic retry logic for transient failures
"""

import json
from datetime import datetime
from typing import List, Optional

import pandas as pd
import psycopg2
from psycopg2 import pool
from psycopg2.extras import execute_values
import redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import get_settings
from simtrademl.features.store.base import FeatureStore
from simtrademl.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureStoreError(Exception):
    """Base exception for feature store errors."""
    pass


class FeatureNotFoundError(FeatureStoreError):
    """Exception raised when requested features are not found."""
    pass


class TimescaleFeatureStore(FeatureStore):
    """Feature store implementation using TimescaleDB and Redis.

    This implementation provides:
    - Batch feature storage in TimescaleDB with JSONB format
    - Time-range queries with point-in-time correctness
    - Low-latency online feature retrieval with Redis caching
    - Connection pooling for PostgreSQL
    - Automatic retry logic for transient failures

    Attributes:
        timescaledb_url: PostgreSQL connection string
        redis_url: Redis connection string
        cache_ttl: Cache TTL in seconds for online features
    """

    def __init__(
        self,
        timescaledb_url: str,
        redis_url: str,
        cache_ttl: int = 300,
        pool_minconn: int = 1,
        pool_maxconn: int = 10,
    ) -> None:
        """Initialize TimescaleDB feature store with connection pools.

        Args:
            timescaledb_url: TimescaleDB connection URL (postgresql://...)
            redis_url: Redis connection URL (redis://...)
            cache_ttl: Online feature cache TTL in seconds (default: 300)
            pool_minconn: Minimum number of connections in pool (default: 1)
            pool_maxconn: Maximum number of connections in pool (default: 10)

        Raises:
            FeatureStoreError: If connection initialization fails
        """
        self.timescaledb_url = timescaledb_url
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl

        logger.info(
            "Initializing TimescaleFeatureStore",
            cache_ttl=cache_ttl,
            pool_minconn=pool_minconn,
            pool_maxconn=pool_maxconn,
        )

        try:
            # Initialize PostgreSQL connection pool
            self.pg_pool = pool.ThreadedConnectionPool(
                minconn=pool_minconn,
                maxconn=pool_maxconn,
                dsn=timescaledb_url,
            )

            # Initialize Redis client
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=False,  # We'll handle JSON encoding manually
            )

            # Test connections
            self._test_connections()

            logger.info("TimescaleFeatureStore initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize TimescaleFeatureStore", error=str(e))
            raise FeatureStoreError(f"Initialization failed: {e}") from e

    def _test_connections(self) -> None:
        """Test database and Redis connections.

        Raises:
            FeatureStoreError: If connection tests fail
        """
        # Test PostgreSQL
        conn = None
        try:
            conn = self.pg_pool.getconn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            logger.debug("PostgreSQL connection test successful")
        except Exception as e:
            raise FeatureStoreError(f"PostgreSQL connection test failed: {e}") from e
        finally:
            if conn:
                self.pg_pool.putconn(conn)

        # Test Redis
        try:
            self.redis_client.ping()
            logger.debug("Redis connection test successful")
        except Exception as e:
            raise FeatureStoreError(f"Redis connection test failed: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((psycopg2.OperationalError, psycopg2.InterfaceError)),
    )
    def write_batch(
        self,
        feature_view: str,
        features: pd.DataFrame,
        entity_column: str = "symbol",
        timestamp_column: str = "timestamp",
    ) -> None:
        """Write batch features to TimescaleDB.

        Features are stored as JSONB for flexible schema. Each row in the DataFrame
        is stored as a separate record in the features table.

        Args:
            feature_view: Feature view name (e.g., 'momentum_features')
            features: DataFrame with entity, timestamp, and feature columns
            entity_column: Name of entity identifier column (default: 'symbol')
            timestamp_column: Name of timestamp column (default: 'timestamp')

        Raises:
            ValueError: If required columns are missing
            FeatureStoreError: If write operation fails
        """
        # Validate inputs
        if entity_column not in features.columns:
            raise ValueError(f"Entity column '{entity_column}' not found in DataFrame")
        if timestamp_column not in features.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")

        if features.empty:
            logger.warning("Empty DataFrame provided, skipping write", feature_view=feature_view)
            return

        logger.info(
            "Writing batch features",
            feature_view=feature_view,
            num_rows=len(features),
            num_features=len(features.columns) - 2,  # Exclude entity and timestamp
        )

        conn = None
        try:
            conn = self.pg_pool.getconn()

            # Prepare data for insertion
            records = []
            for _, row in features.iterrows():
                entity = row[entity_column]
                timestamp = row[timestamp_column]

                # Convert timestamp to datetime if needed
                if isinstance(timestamp, pd.Timestamp):
                    timestamp = timestamp.to_pydatetime()

                # Extract feature columns (exclude entity and timestamp)
                feature_cols = [col for col in features.columns
                               if col not in [entity_column, timestamp_column]]
                feature_dict = {col: row[col] for col in feature_cols}

                # Convert numpy types to Python types for JSON serialization
                feature_dict = {k: float(v) if pd.notna(v) else None
                               for k, v in feature_dict.items()}

                records.append((entity, timestamp, feature_view, json.dumps(feature_dict)))

            # Bulk insert with execute_values for efficiency
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO features (entity, timestamp, feature_view, features)
                    VALUES %s
                    ON CONFLICT (entity, timestamp, feature_view)
                    DO UPDATE SET features = EXCLUDED.features
                    """,
                    records,
                )

            conn.commit()

            logger.info(
                "Successfully wrote batch features",
                feature_view=feature_view,
                rows_written=len(records),
            )

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(
                "Failed to write batch features",
                feature_view=feature_view,
                error=str(e),
            )
            raise FeatureStoreError(f"Write batch failed: {e}") from e
        finally:
            if conn:
                self.pg_pool.putconn(conn)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((psycopg2.OperationalError, psycopg2.InterfaceError)),
    )
    def read_batch(
        self,
        feature_view: str,
        entities: List[str],
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Read batch features from TimescaleDB with time-range filtering.

        This method ensures point-in-time correctness: only features that existed
        at or before the requested timestamp are returned.

        Args:
            feature_view: Feature view name
            entities: List of entity IDs (e.g., ['AAPL', 'MSFT'])
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)

        Returns:
            DataFrame with entity, timestamp, and feature columns

        Raises:
            FeatureNotFoundError: If no features found for the criteria
            FeatureStoreError: If read operation fails
        """
        if not entities:
            raise ValueError("Entities list cannot be empty")

        logger.info(
            "Reading batch features",
            feature_view=feature_view,
            num_entities=len(entities),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
        )

        conn = None
        try:
            conn = self.pg_pool.getconn()

            with conn.cursor() as cur:
                # Query with point-in-time correctness
                cur.execute(
                    """
                    SELECT entity, timestamp, features
                    FROM features
                    WHERE feature_view = %s
                      AND entity = ANY(%s)
                      AND timestamp >= %s
                      AND timestamp <= %s
                    ORDER BY entity, timestamp
                    """,
                    (feature_view, entities, start_time, end_time),
                )

                rows = cur.fetchall()

            if not rows:
                logger.warning(
                    "No features found",
                    feature_view=feature_view,
                    entities=entities,
                )
                raise FeatureNotFoundError(
                    f"No features found for {feature_view} "
                    f"with entities {entities} in time range"
                )

            # Parse results into DataFrame
            records = []
            for entity, timestamp, features_json in rows:
                feature_dict = json.loads(features_json)
                record = {
                    "symbol": entity,
                    "timestamp": timestamp,
                    **feature_dict,
                }
                records.append(record)

            df = pd.DataFrame(records)

            logger.info(
                "Successfully read batch features",
                feature_view=feature_view,
                rows_returned=len(df),
            )

            return df

        except FeatureNotFoundError:
            raise
        except Exception as e:
            logger.error(
                "Failed to read batch features",
                feature_view=feature_view,
                error=str(e),
            )
            raise FeatureStoreError(f"Read batch failed: {e}") from e
        finally:
            if conn:
                self.pg_pool.putconn(conn)

    def read_online(
        self,
        feature_view: str,
        entities: List[str],
    ) -> pd.DataFrame:
        """Read latest features for online inference with Redis caching.

        This method first checks Redis cache for cached features. On cache miss,
        it queries TimescaleDB and updates the cache with the results.

        Args:
            feature_view: Feature view name
            entities: List of entity IDs

        Returns:
            DataFrame with latest features for each entity

        Raises:
            FeatureNotFoundError: If no features found
            FeatureStoreError: If read operation fails
        """
        if not entities:
            raise ValueError("Entities list cannot be empty")

        logger.info(
            "Reading online features",
            feature_view=feature_view,
            num_entities=len(entities),
        )

        # Try to get from Redis cache
        cached_results = []
        cache_misses = []

        for entity in entities:
            cache_key = f"feature:{feature_view}:{entity}"
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    cached_results.append(json.loads(cached_data))
                    logger.debug("Cache hit", entity=entity, feature_view=feature_view)
                else:
                    cache_misses.append(entity)
            except Exception as e:
                logger.warning("Redis cache read failed", entity=entity, error=str(e))
                cache_misses.append(entity)

        # Fetch missing entities from TimescaleDB
        db_results = []
        if cache_misses:
            logger.info(
                "Cache misses, querying database",
                num_misses=len(cache_misses),
                feature_view=feature_view,
            )
            db_results = self._fetch_latest_from_db(feature_view, cache_misses)

            # Update Redis cache with fetched results
            self._update_cache(feature_view, db_results)

        # Combine cached and DB results
        all_results = cached_results + db_results

        if not all_results:
            raise FeatureNotFoundError(
                f"No features found for {feature_view} with entities {entities}"
            )

        df = pd.DataFrame(all_results)

        logger.info(
            "Successfully read online features",
            feature_view=feature_view,
            rows_returned=len(df),
            cache_hits=len(cached_results),
            cache_misses=len(cache_misses),
        )

        return df

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((psycopg2.OperationalError, psycopg2.InterfaceError)),
    )
    def _fetch_latest_from_db(
        self,
        feature_view: str,
        entities: List[str],
    ) -> List[dict]:
        """Fetch latest features for entities from TimescaleDB.

        Args:
            feature_view: Feature view name
            entities: List of entity IDs

        Returns:
            List of feature dictionaries

        Raises:
            FeatureStoreError: If query fails
        """
        conn = None
        try:
            conn = self.pg_pool.getconn()

            with conn.cursor() as cur:
                # Query for latest features per entity
                cur.execute(
                    """
                    SELECT DISTINCT ON (entity) entity, timestamp, features
                    FROM features
                    WHERE feature_view = %s
                      AND entity = ANY(%s)
                    ORDER BY entity, timestamp DESC
                    """,
                    (feature_view, entities),
                )

                rows = cur.fetchall()

            # Parse results
            results = []
            for entity, timestamp, features_json in rows:
                feature_dict = json.loads(features_json)
                record = {
                    "symbol": entity,
                    "timestamp": timestamp,
                    **feature_dict,
                }
                results.append(record)

            return results

        except Exception as e:
            logger.error(
                "Failed to fetch latest from database",
                feature_view=feature_view,
                error=str(e),
            )
            raise FeatureStoreError(f"Database query failed: {e}") from e
        finally:
            if conn:
                self.pg_pool.putconn(conn)

    def _update_cache(self, feature_view: str, results: List[dict]) -> None:
        """Update Redis cache with feature results.

        Args:
            feature_view: Feature view name
            results: List of feature dictionaries to cache
        """
        for result in results:
            entity = result.get("symbol")
            if not entity:
                continue

            cache_key = f"feature:{feature_view}:{entity}"
            try:
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(result, default=str),  # default=str handles datetime
                )
                logger.debug("Updated cache", entity=entity, feature_view=feature_view)
            except Exception as e:
                logger.warning(
                    "Failed to update cache",
                    entity=entity,
                    feature_view=feature_view,
                    error=str(e),
                )

    def close(self) -> None:
        """Close all connections and cleanup resources.

        This method should be called when the feature store is no longer needed.
        """
        logger.info("Closing TimescaleFeatureStore connections")

        try:
            if hasattr(self, 'pg_pool'):
                self.pg_pool.closeall()
                logger.debug("Closed PostgreSQL connection pool")
        except Exception as e:
            logger.warning("Error closing PostgreSQL pool", error=str(e))

        try:
            if hasattr(self, 'redis_client'):
                self.redis_client.close()
                logger.debug("Closed Redis connection")
        except Exception as e:
            logger.warning("Error closing Redis connection", error=str(e))

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
