"""Pytest configuration for integration tests."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment before running any tests."""
    # Set test API keys for authentication tests
    original_api_keys = os.environ.get("API_KEYS")
    os.environ["API_KEYS"] = '["test-key-123"]'

    yield

    # Restore original environment
    if original_api_keys:
        os.environ["API_KEYS"] = original_api_keys
    else:
        os.environ.pop("API_KEYS", None)
