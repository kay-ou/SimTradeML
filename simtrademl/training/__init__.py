"""Training module for model training and hyperparameter tuning.

This module provides:
- Trainer class for model training with MLflow tracking
- HyperparameterTuner for Optuna-based hyperparameter optimization
- Training configuration and result data structures
"""

from simtrademl.training.trainer import (
    TrainingConfig,
    TrainingResult,
    Trainer,
)
from simtrademl.training.tuner import HyperparameterTuner

__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "Trainer",
    "HyperparameterTuner",
]
