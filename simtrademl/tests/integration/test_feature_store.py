"""Integration tests for TimescaleDB + Redis feature store.

This module provides comprehensive integration tests for the feature store,
testing end-to-end functionality with real PostgreSQL and Redis instances.
"""

import os
from datetime import datetime, timedelta
from typing import Generator

import pandas as pd
import psycopg2
import pytest
import redis

from simtrademl.features.store.timescale import TimescaleFeatureStore, FeatureNotFoundError


# Test database configuration
TEST_TIMESCALEDB_URL = os.getenv(
    "TEST_TIMESCALEDB_URL",
    "postgresql://postgres:postgres@localhost:5432/simtrademl_test"
)
TEST_REDIS_URL = os.getenv(
    "TEST_REDIS_URL",
    "redis://localhost:6379/15"  # Use DB 15 for tests to avoid conflicts
)


@pytest.fixture(scope="module")
def test_db_connection() -> Generator:
    """Create test database connection and ensure tables exist.

    This fixture:
    1. Connects to PostgreSQL
    2. Creates the features table if it doesn't exist
    3. Yields the connection
    4. Cleans up after all tests in the module
    """
    conn = psycopg2.connect(TEST_TIMESCALEDB_URL)
    conn.autocommit = True

    try:
        with conn.cursor() as cur:
            # Create features table with TimescaleDB hypertable
            cur.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    entity TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    feature_view TEXT NOT NULL,
                    features JSONB NOT NULL,
                    PRIMARY KEY (entity, timestamp, feature_view)
                )
            """)

            # Try to create hypertable (will fail if already exists, which is ok)
            try:
                cur.execute("""
                    SELECT create_hypertable('features', 'timestamp',
                                            if_not_exists => TRUE)
                """)
            except psycopg2.Error:
                # Hypertable might already exist or TimescaleDB extension not available
                pass

        yield conn

    finally:
        conn.close()


@pytest.fixture(scope="function")
def clean_test_db(test_db_connection) -> Generator:
    """Clean test database before and after each test.

    This ensures test isolation by removing all data before each test runs.
    """
    conn = test_db_connection

    # Clean before test
    with conn.cursor() as cur:
        cur.execute("DELETE FROM features")

    yield

    # Clean after test
    with conn.cursor() as cur:
        cur.execute("DELETE FROM features")


@pytest.fixture(scope="function")
def clean_test_redis() -> Generator:
    """Clean test Redis database before and after each test."""
    redis_client = redis.from_url(TEST_REDIS_URL)

    # Clean before test
    redis_client.flushdb()

    yield

    # Clean after test
    redis_client.flushdb()
    redis_client.close()


@pytest.fixture(scope="function")
def feature_store(clean_test_db, clean_test_redis) -> Generator[TimescaleFeatureStore, None, None]:
    """Create feature store instance for testing.

    This fixture provides a fully configured feature store connected to
    test databases (PostgreSQL and Redis).
    """
    store = TimescaleFeatureStore(
        timescaledb_url=TEST_TIMESCALEDB_URL,
        redis_url=TEST_REDIS_URL,
        cache_ttl=60,  # Short TTL for testing
        pool_minconn=1,
        pool_maxconn=5,
    )

    yield store

    # Cleanup
    store.close()


@pytest.fixture(scope="function")
def sample_features() -> pd.DataFrame:
    """Generate sample feature data for testing.

    Creates a DataFrame with multiple entities (symbols) and timestamps,
    suitable for testing batch write/read operations.
    """
    timestamps = [
        datetime(2024, 1, 1, 10, 0, 0),
        datetime(2024, 1, 1, 11, 0, 0),
        datetime(2024, 1, 1, 12, 0, 0),
        datetime(2024, 1, 2, 10, 0, 0),
        datetime(2024, 1, 2, 11, 0, 0),
    ]

    data = {
        "symbol": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
        "timestamp": timestamps,
        "rsi_14": [65.0, 70.0, 68.5, 55.0, 58.2],
        "macd": [0.5, 0.8, 0.7, -0.2, 0.1],
        "macd_signal": [0.4, 0.6, 0.65, -0.1, 0.05],
    }

    return pd.DataFrame(data)


@pytest.mark.integration
def test_write_batch_and_read_batch(feature_store: TimescaleFeatureStore, sample_features: pd.DataFrame):
    """Test writing features in batch and reading them back.

    This test verifies:
    1. Batch write stores all features correctly
    2. Batch read retrieves features with time-range filtering
    3. Returned data matches original data
    """
    # Write batch features
    feature_store.write_batch(
        feature_view="momentum_features",
        features=sample_features,
        entity_column="symbol",
        timestamp_column="timestamp",
    )

    # Read batch features for all entities in time range
    result = feature_store.read_batch(
        feature_view="momentum_features",
        entities=["AAPL", "MSFT"],
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 3),
    )

    # Verify result structure
    assert len(result) == 5  # All 5 rows
    assert "symbol" in result.columns
    assert "timestamp" in result.columns
    assert "rsi_14" in result.columns
    assert "macd" in result.columns
    assert "macd_signal" in result.columns

    # Verify data integrity
    assert set(result["symbol"].unique()) == {"AAPL", "MSFT"}
    assert len(result[result["symbol"] == "AAPL"]) == 3
    assert len(result[result["symbol"] == "MSFT"]) == 2

    # Verify specific values
    aapl_first = result[(result["symbol"] == "AAPL") &
                        (result["timestamp"] == datetime(2024, 1, 1, 10, 0, 0))].iloc[0]
    assert aapl_first["rsi_14"] == 65.0
    assert aapl_first["macd"] == 0.5


@pytest.mark.integration
def test_read_batch_time_range_filtering(feature_store: TimescaleFeatureStore, sample_features: pd.DataFrame):
    """Test time-range filtering in batch read.

    Verifies that only features within the specified time range are returned.
    """
    # Write features
    feature_store.write_batch(
        feature_view="momentum_features",
        features=sample_features,
    )

    # Read only features from Jan 1
    result = feature_store.read_batch(
        feature_view="momentum_features",
        entities=["AAPL", "MSFT"],
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        end_time=datetime(2024, 1, 1, 23, 59, 59),
    )

    # Should only get Jan 1 data (3 rows for AAPL)
    assert len(result) == 3
    assert all(result["timestamp"].dt.date == datetime(2024, 1, 1).date())


@pytest.mark.integration
def test_read_batch_entity_filtering(feature_store: TimescaleFeatureStore, sample_features: pd.DataFrame):
    """Test entity filtering in batch read.

    Verifies that only features for specified entities are returned.
    """
    # Write features
    feature_store.write_batch(
        feature_view="momentum_features",
        features=sample_features,
    )

    # Read only AAPL features
    result = feature_store.read_batch(
        feature_view="momentum_features",
        entities=["AAPL"],
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 3),
    )

    # Should only get AAPL data (3 rows)
    assert len(result) == 3
    assert all(result["symbol"] == "AAPL")


@pytest.mark.integration
def test_read_batch_not_found(feature_store: TimescaleFeatureStore):
    """Test reading features when none exist.

    Verifies that FeatureNotFoundError is raised when no features match criteria.
    """
    with pytest.raises(FeatureNotFoundError) as exc_info:
        feature_store.read_batch(
            feature_view="nonexistent_features",
            entities=["AAPL"],
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
        )

    assert "No features found" in str(exc_info.value)


@pytest.mark.integration
def test_read_online_cache_miss(feature_store: TimescaleFeatureStore, sample_features: pd.DataFrame):
    """Test online read with cache miss (queries database).

    This test verifies:
    1. First read queries database (cache miss)
    2. Latest features for each entity are returned
    3. Cache is populated after database query
    """
    # Write features
    feature_store.write_batch(
        feature_view="momentum_features",
        features=sample_features,
    )

    # First read - should query database (cache miss)
    result = feature_store.read_online(
        feature_view="momentum_features",
        entities=["AAPL", "MSFT"],
    )

    # Verify result contains latest features
    assert len(result) == 2  # One row per entity
    assert set(result["symbol"].unique()) == {"AAPL", "MSFT"}

    # Verify latest timestamps
    aapl_row = result[result["symbol"] == "AAPL"].iloc[0]
    msft_row = result[result["symbol"] == "MSFT"].iloc[0]

    assert aapl_row["timestamp"] == datetime(2024, 1, 1, 12, 0, 0)  # Latest AAPL
    assert aapl_row["rsi_14"] == 68.5

    assert msft_row["timestamp"] == datetime(2024, 1, 2, 11, 0, 0)  # Latest MSFT
    assert msft_row["rsi_14"] == 58.2


@pytest.mark.integration
def test_read_online_cache_hit(feature_store: TimescaleFeatureStore, sample_features: pd.DataFrame):
    """Test online read with cache hit.

    This test verifies:
    1. First read populates cache
    2. Second read retrieves from cache (cache hit)
    3. Cached data matches original data
    """
    # Write features
    feature_store.write_batch(
        feature_view="momentum_features",
        features=sample_features,
    )

    # First read - populates cache
    result1 = feature_store.read_online(
        feature_view="momentum_features",
        entities=["AAPL"],
    )

    # Second read - should hit cache
    result2 = feature_store.read_online(
        feature_view="momentum_features",
        entities=["AAPL"],
    )

    # Results should be identical
    assert len(result1) == len(result2) == 1
    assert result1["symbol"].iloc[0] == result2["symbol"].iloc[0] == "AAPL"
    assert result1["rsi_14"].iloc[0] == result2["rsi_14"].iloc[0]

    # Verify cache was used by checking Redis directly
    redis_client = redis.from_url(TEST_REDIS_URL)
    cache_key = "feature:momentum_features:AAPL"
    cached_data = redis_client.get(cache_key)
    assert cached_data is not None
    redis_client.close()


@pytest.mark.integration
def test_read_online_cache_ttl(feature_store: TimescaleFeatureStore, sample_features: pd.DataFrame):
    """Test that cache entries have correct TTL.

    Verifies that cached features expire after cache_ttl seconds.
    """
    # Write features
    feature_store.write_batch(
        feature_view="momentum_features",
        features=sample_features,
    )

    # Read to populate cache
    feature_store.read_online(
        feature_view="momentum_features",
        entities=["AAPL"],
    )

    # Check TTL on cached entry
    redis_client = redis.from_url(TEST_REDIS_URL)
    cache_key = "feature:momentum_features:AAPL"
    ttl = redis_client.ttl(cache_key)

    # TTL should be set and > 0 (feature_store has cache_ttl=60)
    assert ttl > 0
    assert ttl <= 60

    redis_client.close()


@pytest.mark.integration
def test_point_in_time_correctness(feature_store: TimescaleFeatureStore):
    """Test point-in-time correctness: queries don't return future data.

    This is critical for training data to prevent data leakage.
    Verifies that querying at a specific time only returns features
    that existed at or before that time.
    """
    # Create features at different times
    past_time = datetime(2024, 1, 1, 10, 0, 0)
    current_time = datetime(2024, 1, 1, 12, 0, 0)
    future_time = datetime(2024, 1, 1, 14, 0, 0)

    features_df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "timestamp": [past_time, current_time, future_time],
        "rsi_14": [60.0, 65.0, 70.0],
        "macd": [0.3, 0.5, 0.7],
        "macd_signal": [0.2, 0.4, 0.6],
    })

    # Write features
    feature_store.write_batch(
        feature_view="momentum_features",
        features=features_df,
    )

    # Query up to current_time - should NOT include future data
    result = feature_store.read_batch(
        feature_view="momentum_features",
        entities=["AAPL"],
        start_time=past_time,
        end_time=current_time,
    )

    # Should only get 2 rows (past and current, NOT future)
    assert len(result) == 2
    assert all(result["timestamp"] <= current_time)
    assert future_time not in result["timestamp"].values

    # Verify data integrity
    timestamps = sorted(result["timestamp"].tolist())
    assert timestamps == [past_time, current_time]


@pytest.mark.integration
def test_write_batch_upsert(feature_store: TimescaleFeatureStore):
    """Test that write_batch performs upsert (update on conflict).

    Verifies that writing the same entity+timestamp+feature_view
    updates existing features rather than failing.
    """
    # Write initial features
    initial_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "timestamp": [datetime(2024, 1, 1, 10, 0, 0)],
        "rsi_14": [60.0],
        "macd": [0.5],
    })

    feature_store.write_batch(
        feature_view="momentum_features",
        features=initial_df,
    )

    # Write updated features (same entity + timestamp)
    updated_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "timestamp": [datetime(2024, 1, 1, 10, 0, 0)],
        "rsi_14": [65.0],  # Updated value
        "macd": [0.8],     # Updated value
    })

    feature_store.write_batch(
        feature_view="momentum_features",
        features=updated_df,
    )

    # Read back - should have updated values
    result = feature_store.read_batch(
        feature_view="momentum_features",
        entities=["AAPL"],
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 2),
    )

    # Should only have 1 row (updated, not duplicate)
    assert len(result) == 1
    assert result["rsi_14"].iloc[0] == 65.0
    assert result["macd"].iloc[0] == 0.8


@pytest.mark.integration
def test_multiple_feature_views(feature_store: TimescaleFeatureStore):
    """Test storing and retrieving different feature views.

    Verifies that different feature views are isolated from each other.
    """
    # Write momentum features
    momentum_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "timestamp": [datetime(2024, 1, 1, 10, 0, 0)],
        "rsi_14": [65.0],
        "macd": [0.5],
    })

    feature_store.write_batch(
        feature_view="momentum_features",
        features=momentum_df,
    )

    # Write volatility features
    volatility_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "timestamp": [datetime(2024, 1, 1, 10, 0, 0)],
        "atr_14": [2.5],
        "bb_upper": [152.0],
        "bb_lower": [148.0],
    })

    feature_store.write_batch(
        feature_view="volatility_features",
        features=volatility_df,
    )

    # Read momentum features
    momentum_result = feature_store.read_batch(
        feature_view="momentum_features",
        entities=["AAPL"],
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 2),
    )

    # Read volatility features
    volatility_result = feature_store.read_batch(
        feature_view="volatility_features",
        entities=["AAPL"],
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 2),
    )

    # Verify isolation
    assert "rsi_14" in momentum_result.columns
    assert "macd" in momentum_result.columns
    assert "atr_14" not in momentum_result.columns

    assert "atr_14" in volatility_result.columns
    assert "bb_upper" in volatility_result.columns
    assert "rsi_14" not in volatility_result.columns


@pytest.mark.integration
def test_empty_dataframe_handling(feature_store: TimescaleFeatureStore):
    """Test that writing empty DataFrame is handled gracefully."""
    empty_df = pd.DataFrame(columns=["symbol", "timestamp", "rsi_14"])

    # Should not raise error
    feature_store.write_batch(
        feature_view="momentum_features",
        features=empty_df,
    )


@pytest.mark.integration
def test_large_batch_write_read(feature_store: TimescaleFeatureStore):
    """Test writing and reading large batches of features.

    Verifies that the feature store can handle realistic data volumes.
    """
    # Generate 1000 rows of feature data
    num_rows = 1000
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(num_rows)]

    large_df = pd.DataFrame({
        "symbol": ["AAPL"] * num_rows,
        "timestamp": timestamps,
        "rsi_14": [65.0 + i * 0.01 for i in range(num_rows)],
        "macd": [0.5 + i * 0.001 for i in range(num_rows)],
        "macd_signal": [0.4 + i * 0.001 for i in range(num_rows)],
    })

    # Write large batch
    feature_store.write_batch(
        feature_view="momentum_features",
        features=large_df,
    )

    # Read back all features
    result = feature_store.read_batch(
        feature_view="momentum_features",
        entities=["AAPL"],
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 12, 31),
    )

    # Verify all rows were stored and retrieved
    assert len(result) == num_rows
    assert result["symbol"].unique()[0] == "AAPL"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
