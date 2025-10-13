"""Training configuration and result data structures.

This module provides data classes for configuring model training and
capturing training results with MLflow integration.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrainingConfig:
    """Training configuration for model training.

    This dataclass encapsulates all configuration parameters needed for
    training a model, including model type, hyperparameters, data split ratios,
    and training options like early stopping.

    Attributes:
        model_type: Model type key from MODEL_REGISTRY (e.g., 'xgboost', 'lightgbm').
                   This determines which model class will be instantiated for training.
        hyperparameters: Dictionary of model-specific hyperparameters.
                        These are passed directly to the model's constructor.
        train_ratio: Proportion of data to use for training (default: 0.7).
                    Must be in range (0, 1).
        val_ratio: Proportion of data to use for validation (default: 0.15).
                  Used for early stopping and hyperparameter tuning.
        test_ratio: Proportion of data to use for final testing (default: 0.15).
                   Used for final model evaluation.
        random_seed: Random seed for reproducibility (default: 42).
                    Ensures consistent data splits across runs.
        early_stopping_rounds: Number of rounds without improvement before stopping
                              training (optional). Only applicable to tree-based models
                              like XGBoost and LightGBM. If None, no early stopping.
        eval_metric: Evaluation metric for model selection (default: 'mae').
                    Supported values: 'mae', 'rmse', 'r2', 'mape'.
                    Used to determine best model and for early stopping.

    Example:
        >>> config = TrainingConfig(
        ...     model_type='xgboost',
        ...     hyperparameters={
        ...         'n_estimators': 100,
        ...         'max_depth': 5,
        ...         'learning_rate': 0.1,
        ...     },
        ...     train_ratio=0.7,
        ...     val_ratio=0.15,
        ...     test_ratio=0.15,
        ...     random_seed=42,
        ...     early_stopping_rounds=10,
        ...     eval_metric='mae',
        ... )

    Notes:
        - train_ratio + val_ratio + test_ratio should equal 1.0
        - For time-series data, splits maintain temporal order (no shuffling)
        - hyperparameters dict is model-specific; refer to model documentation
    """

    model_type: str
    hyperparameters: Dict[str, Any]
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    early_stopping_rounds: Optional[int] = None
    eval_metric: str = "mae"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate ratios sum to 1.0 (with small tolerance for floating point)
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must equal 1.0, "
                f"got {total_ratio}"
            )

        # Validate individual ratios are in valid range
        for ratio_name, ratio_value in [
            ("train_ratio", self.train_ratio),
            ("val_ratio", self.val_ratio),
            ("test_ratio", self.test_ratio),
        ]:
            if not (0.0 < ratio_value < 1.0):
                raise ValueError(
                    f"{ratio_name} must be in range (0, 1), got {ratio_value}"
                )

        # Validate eval_metric
        valid_metrics = {"mae", "rmse", "r2", "mape", "mse"}
        if self.eval_metric not in valid_metrics:
            raise ValueError(
                f"eval_metric must be one of {valid_metrics}, got {self.eval_metric}"
            )

        # Validate early_stopping_rounds if provided
        if self.early_stopping_rounds is not None and self.early_stopping_rounds <= 0:
            raise ValueError(
                f"early_stopping_rounds must be positive, got {self.early_stopping_rounds}"
            )


@dataclass
class TrainingResult:
    """Training result containing model metadata and metrics.

    This dataclass captures the output of a training run, including MLflow
    run information, model location, evaluation metrics, and training metadata.

    Attributes:
        run_id: MLflow run ID for tracking this training run.
               Can be used to query MLflow for detailed run information.
        model_name: Human-readable name for the trained model.
                   Used for model registration and identification.
        model_uri: MLflow model URI for loading the trained model.
                  Format: 'runs:/<run_id>/model' or 'models:/<model_name>/<version>'.
        metrics: Dictionary of evaluation metrics computed on test set.
                Keys are metric names (e.g., 'mae', 'rmse', 'r2'),
                values are metric scores.
        best_iteration: Best iteration/epoch determined by early stopping (optional).
                       Only populated for tree-based models with early stopping enabled.
                       None if early stopping was not used or not applicable.

    Example:
        >>> result = TrainingResult(
        ...     run_id='abc123def456',
        ...     model_name='xgboost_price_predictor',
        ...     model_uri='runs:/abc123def456/model',
        ...     metrics={
        ...         'mae': 0.0523,
        ...         'rmse': 0.0847,
        ...         'r2': 0.9234,
        ...         'mape': 0.0612,
        ...     },
        ...     best_iteration=87,
        ... )

    Notes:
        - All metrics are computed on the test set (held-out data)
        - Lower is better for mae, rmse, mape; higher is better for r2
        - best_iteration can be used to retrain with optimal number of iterations
    """

    run_id: str
    model_name: str
    model_uri: str
    metrics: Dict[str, float]
    best_iteration: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate result after initialization."""
        # Validate run_id is not empty
        if not self.run_id or not self.run_id.strip():
            raise ValueError("run_id cannot be empty")

        # Validate model_name is not empty
        if not self.model_name or not self.model_name.strip():
            raise ValueError("model_name cannot be empty")

        # Validate model_uri is not empty and has valid format
        if not self.model_uri or not self.model_uri.strip():
            raise ValueError("model_uri cannot be empty")

        # Validate metrics dict is not empty
        if not self.metrics:
            raise ValueError("metrics dictionary cannot be empty")

        # Validate all metric values are finite numbers
        for metric_name, metric_value in self.metrics.items():
            if not isinstance(metric_value, (int, float)):
                raise ValueError(
                    f"Metric '{metric_name}' must be numeric, got {type(metric_value)}"
                )
            if not (-1e10 < metric_value < 1e10):  # Check for inf/nan
                raise ValueError(
                    f"Metric '{metric_name}' has invalid value: {metric_value}"
                )

        # Validate best_iteration if provided
        if self.best_iteration is not None and self.best_iteration <= 0:
            raise ValueError(
                f"best_iteration must be positive, got {self.best_iteration}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert training result to dictionary.

        Returns:
            Dictionary representation of the training result.

        Example:
            >>> result = TrainingResult(...)
            >>> result_dict = result.to_dict()
            >>> print(result_dict)
            {
                'run_id': 'abc123',
                'model_name': 'xgboost_predictor',
                'model_uri': 'runs:/abc123/model',
                'metrics': {'mae': 0.05, 'rmse': 0.08},
                'best_iteration': 87
            }
        """
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "model_uri": self.model_uri,
            "metrics": self.metrics,
            "best_iteration": self.best_iteration,
        }

    def get_primary_metric(self, metric_name: str = "mae") -> float:
        """Get primary evaluation metric value.

        Args:
            metric_name: Name of the metric to retrieve (default: 'mae')

        Returns:
            Metric value

        Raises:
            KeyError: If metric_name not found in metrics

        Example:
            >>> result = TrainingResult(...)
            >>> mae = result.get_primary_metric('mae')
            >>> print(f"Test MAE: {mae:.4f}")
        """
        if metric_name not in self.metrics:
            raise KeyError(
                f"Metric '{metric_name}' not found. Available metrics: {list(self.metrics.keys())}"
            )
        return self.metrics[metric_name]


# Trainer class implementation
import pandas as pd
from pathlib import Path
from typing import Any, Optional, Tuple


class Trainer:
    """Model trainer with MLflow integration.

    Orchestrates model training with MLflow experiment tracking, time-series aware
    data splitting, and comprehensive evaluation metrics.

    Attributes:
        config: Training configuration containing model type, hyperparameters, and split ratios
        mlflow_tracking_uri: MLflow tracking server URI (optional, uses settings if not provided)
        mlflow_experiment_name: MLflow experiment name for grouping runs
    """

    def __init__(
        self,
        config: TrainingConfig,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: str = "default",
    ) -> None:
        """Initialize trainer.

        Args:
            config: Training configuration with model type and hyperparameters
            mlflow_tracking_uri: MLflow tracking server URI (defaults to settings)
            mlflow_experiment_name: Experiment name for MLflow tracking

        Raises:
            ValueError: If model_type not found in MODEL_REGISTRY
        """
        from simtrademl.models.registry import get_model_class
        from config.settings import get_settings
        from simtrademl.utils.logging import get_logger

        self.config = config
        self.logger = get_logger(__name__)

        # Validate model type exists in registry
        try:
            self.model_class = get_model_class(config.model_type)
        except ValueError as e:
            self.logger.error(
                "Invalid model type",
                model_type=config.model_type,
                error=str(e),
            )
            raise

        # Configure MLflow
        import mlflow

        settings = get_settings()
        self.mlflow_tracking_uri = mlflow_tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Set experiment
        self.mlflow_experiment_name = mlflow_experiment_name
        mlflow.set_experiment(mlflow_experiment_name)

        self.logger.info(
            "Trainer initialized",
            model_type=config.model_type,
            experiment=mlflow_experiment_name,
            tracking_uri=self.mlflow_tracking_uri,
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> TrainingResult:
        """Train model with MLflow tracking.

        Performs complete training workflow:
        1. Split data (if validation not provided)
        2. Instantiate model from registry
        3. Train model with optional eval_set
        4. Log params, metrics, and model to MLflow
        5. Return training result with run metadata

        Args:
            X: Training features (DataFrame)
            y: Training target (Series)
            X_val: Validation features (optional, will split if not provided)
            y_val: Validation target (optional)

        Returns:
            TrainingResult with run_id, model_uri, metrics, and best_iteration

        Raises:
            TrainingError: If training fails
        """
        import mlflow
        import numpy as np

        self.logger.info(
            "Starting training",
            model_type=self.config.model_type,
            n_samples=len(X),
            n_features=X.shape[1],
        )

        # Split data if validation set not provided
        if X_val is None or y_val is None:
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)
        else:
            # If validation provided, split remaining into train/test
            train_size = int(len(X) * (self.config.train_ratio / (self.config.train_ratio + self.config.test_ratio)))
            X_train = X.iloc[:train_size]
            y_train = y.iloc[:train_size]
            X_test = X.iloc[train_size:]
            y_test = y.iloc[train_size:]

        self.logger.info(
            "Data split complete",
            train_size=len(X_train),
            val_size=len(X_val),
            test_size=len(X_test),
        )

        # Start MLflow run
        with mlflow.start_run() as run:
            # Log configuration
            mlflow.log_params(self.config.hyperparameters)
            mlflow.log_param("model_type", self.config.model_type)
            mlflow.log_param("train_ratio", self.config.train_ratio)
            mlflow.log_param("val_ratio", self.config.val_ratio)
            mlflow.log_param("test_ratio", self.config.test_ratio)
            mlflow.log_param("random_seed", self.config.random_seed)
            mlflow.log_param("eval_metric", self.config.eval_metric)

            if self.config.early_stopping_rounds:
                mlflow.log_param("early_stopping_rounds", self.config.early_stopping_rounds)

            # Instantiate model from registry
            model = self.model_class(**self.config.hyperparameters)

            self.logger.info(
                "Model instantiated",
                model_class=self.model_class.__name__,
                hyperparameters=self.config.hyperparameters,
            )

            # Train model - use simple fit since BaseModel interface doesn't support eval_set
            try:
                self.logger.info("Training model")
                model.fit(X_train, y_train)
                self.logger.info("Model training complete")

            except Exception as e:
                self.logger.error(
                    "Training failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

            # Get best iteration if available (for some model types)
            best_iteration = None
            if hasattr(model, "best_iteration"):
                best_iteration = model.best_iteration
                mlflow.log_metric("best_iteration", best_iteration)
                self.logger.info("Best iteration", iteration=best_iteration)
            elif hasattr(model, "best_iteration_"):
                best_iteration = model.best_iteration_
                mlflow.log_metric("best_iteration", best_iteration)
                self.logger.info("Best iteration", iteration=best_iteration)

            # Evaluate model on test set
            metrics = self._evaluate(model, X_test, y_test)

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            self.logger.info("Evaluation metrics", metrics=metrics)

            # Log model to MLflow using BaseModel save interface
            model_name = f"{self.config.model_type}_model"

            try:
                # Try sklearn logging first (works for many models)
                mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
                self.logger.info("Model logged to MLflow", model_name=model_name)
            except Exception as e:
                # Fallback: use BaseModel save interface
                self.logger.warning(
                    "Failed to log with mlflow.sklearn, using BaseModel save interface",
                    error=str(e),
                )
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_path = Path(tmpdir) / "model.pkl"
                    model.save(model_path)
                    mlflow.log_artifact(str(model_path), "model")

            # Get model URI and run info
            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"

            self.logger.info(
                "Training complete",
                run_id=run_id,
                model_uri=model_uri,
                metrics=metrics,
            )

            # Return training result
            return TrainingResult(
                run_id=run_id,
                model_name=model_name,
                model_uri=model_uri,
                metrics=metrics,
                best_iteration=best_iteration,
            )

    def _split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Time-series aware data split.

        Splits data maintaining temporal order (no shuffling) to prevent data leakage
        in time-series models. Uses configured ratios for train/val/test split.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n_samples = len(X)

        # Calculate split indices (maintain temporal order)
        train_end = int(n_samples * self.config.train_ratio)
        val_end = train_end + int(n_samples * self.config.val_ratio)

        # Split without shuffling (time-series aware)
        X_train = X.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]

        y_train = y.iloc[:train_end]
        y_val = y.iloc[train_end:val_end]
        y_test = y.iloc[val_end:]

        self.logger.debug(
            "Data split indices",
            train_end=train_end,
            val_end=val_end,
            total=n_samples,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _evaluate(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate model and compute metrics.

        Computes comprehensive evaluation metrics including MAE, RMSE, R2, MAPE, and MSE.
        All metrics are calculated on the test set.

        Args:
            model: Trained model with predict method
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of metric names to values

        Raises:
            ValueError: If predictions contain invalid values
        """
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )
        import numpy as np

        # Make predictions
        y_pred = model.predict(X_test)

        # Validate predictions
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            self.logger.error("Predictions contain invalid values (NaN or Inf)")
            raise ValueError("Predictions contain NaN or Inf values")

        # Compute metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Compute MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        y_test_np = y_test.values if isinstance(y_test, pd.Series) else y_test
        mask = y_test_np != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_test_np[mask] - y_pred[mask]) / y_test_np[mask])) * 100
        else:
            mape = np.nan

        metrics = {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape) if not np.isnan(mape) else None,
        }

        # Remove None values
        metrics = {k: v for k, v in metrics.items() if v is not None}

        return metrics
