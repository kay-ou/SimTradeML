# Integration Tests

This directory contains integration tests that require external services (PostgreSQL, Redis, MLflow, etc.).

## Feature Store Integration Tests

The feature store tests (`test_feature_store.py`) require:
- PostgreSQL with TimescaleDB extension
- Redis

### Running with Docker Compose

The easiest way to run integration tests is using Docker Compose:

```bash
# Start test databases
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
poetry run pytest simtrademl/tests/integration/test_feature_store.py -v -m integration

# Stop test databases
docker-compose -f docker-compose.test.yml down
```

### Running with Local Services

If you have PostgreSQL and Redis running locally:

```bash
# Set test database URLs
export TEST_TIMESCALEDB_URL="postgresql://postgres:postgres@localhost:5432/simtrademl_test"
export TEST_REDIS_URL="redis://localhost:6379/15"

# Create test database
createdb simtrademl_test
psql simtrademl_test -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"

# Run tests
poetry run pytest simtrademl/tests/integration/test_feature_store.py -v -m integration
```

## Docker Compose Configuration

Create `docker-compose.test.yml` in the project root:

```yaml
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg14
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: simtrademl_test
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
```

## Test Coverage

The feature store integration tests cover:

1. **Batch Write and Read**
   - Write features in batch to TimescaleDB
   - Read features back with time-range filtering
   - Verify data integrity

2. **Time-Range Filtering**
   - Test that only features within specified time ranges are returned
   - Validate start_time and end_time boundaries

3. **Entity Filtering**
   - Test that only features for specified entities are returned
   - Verify entity-based query isolation

4. **Error Handling**
   - Test FeatureNotFoundError when no features exist
   - Validate error messages and exception handling

5. **Redis Caching**
   - Test cache miss (first read queries database)
   - Test cache hit (second read uses cache)
   - Verify cache TTL configuration
   - Check cache key format

6. **Point-in-Time Correctness**
   - Critical for preventing data leakage in training
   - Verify that queries don't return future data
   - Test time-travel queries

7. **Upsert Behavior**
   - Test that duplicate writes update existing features
   - Verify conflict resolution strategy

8. **Feature View Isolation**
   - Test that different feature views are independent
   - Verify feature view namespacing

9. **Edge Cases**
   - Empty DataFrame handling
   - Large batch operations (1000+ rows)
   - Missing entities

## CI/CD Integration

In CI environments (GitHub Actions), use service containers:

```yaml
jobs:
  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: timescale/timescaledb:latest-pg14
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: simtrademl_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      - name: Run integration tests
        run: poetry run pytest -m integration
        env:
          TEST_TIMESCALEDB_URL: postgresql://postgres:postgres@localhost:5432/simtrademl_test
          TEST_REDIS_URL: redis://localhost:6379/15
```

## Troubleshooting

### Connection Refused Errors

If you see `connection refused` errors:
1. Verify PostgreSQL and Redis are running: `docker-compose ps`
2. Check ports are not already in use: `lsof -i :5432` and `lsof -i :6379`
3. Verify environment variables are set correctly

### TimescaleDB Extension Missing

If you see errors about TimescaleDB:
```bash
psql simtrademl_test -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
```

### Permission Errors

If you see permission errors:
```bash
psql simtrademl_test -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;"
```

## Test Data Cleanup

Tests automatically clean up data using fixtures:
- `clean_test_db`: Removes all data from PostgreSQL before/after each test
- `clean_test_redis`: Flushes test Redis database (DB 15) before/after each test

This ensures test isolation and reproducibility.
