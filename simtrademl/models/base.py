"""Base model interface for all ML models.

Defines the abstract base class that all models must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all ML models.

    All models must implement fit, predict, save, and load methods.
    """

    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the model on the given data.

        Args:
            X: Training features
            y: Training target
        """
        pass

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions on the given data.

        Args:
            X: Features to predict on

        Returns:
            Array of predictions
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> "BaseModel":
        """Load a model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded model instance
        """
        pass
