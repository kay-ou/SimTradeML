"""Feature store module.

This module provides abstract interfaces and implementations for feature storage
and retrieval, supporting both batch (training) and online (inference) workflows.
"""

from simtrademl.features.store.base import FeatureStore
from simtrademl.features.store.timescale import (
    TimescaleFeatureStore,
    FeatureStoreError,
    FeatureNotFoundError,
)

__all__ = [
    "FeatureStore",
    "TimescaleFeatureStore",
    "FeatureStoreError",
    "FeatureNotFoundError",
]
