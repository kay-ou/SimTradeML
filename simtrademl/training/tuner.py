"""Hyperparameter tuning module using Optuna."""

from typing import Any, Dict, Optional

import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from simtrademl.models.registry import MODEL_REGISTRY, get_model_class
from simtrademl.training.search_spaces import SEARCH_SPACES, get_search_space
from simtrademl.utils.logging import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """Hyperparameter tuner using Optuna.

    This class automates hyperparameter optimization for models registered in
    MODEL_REGISTRY. It uses Optuna's Tree-structured Parzen Estimator (TPE)
    algorithm to efficiently search the hyperparameter space defined in
    SEARCH_SPACES.

    Example:
        >>> tuner = HyperparameterTuner(
        ...     model_type="xgboost",
        ...     n_trials=50,
        ...     n_jobs=4
        ... )
        >>> best_params = tuner.optimize(X_train, y_train, X_val, y_val)
        >>> print(best_params)
        {'max_depth': 7, 'learning_rate': 0.05, ...}
    """

    def __init__(
        self,
        model_type: str,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
    ) -> None:
        """Initialize hyperparameter tuner.

        Args:
            model_type: Model type from MODEL_REGISTRY (e.g., 'xgboost', 'lightgbm')
            n_trials: Number of optimization trials to run
            timeout: Timeout in seconds (None = no timeout)
            n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = sequential)

        Raises:
            ValueError: If model_type not found in MODEL_REGISTRY
            KeyError: If search space not defined for model_type
        """
        if model_type not in MODEL_REGISTRY:
            available = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Available types: {available}"
            )

        if model_type not in SEARCH_SPACES:
            raise KeyError(
                f"No search space defined for model type '{model_type}'. "
                f"Please add search space to simtrademl/training/search_spaces.py"
            )

        self.model_type = model_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs

        logger.info(
            "Initialized HyperparameterTuner",
            model_type=model_type,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
        )

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = "mae",
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna.

        Runs Optuna optimization to find the best hyperparameters by training
        models on X_train/y_train and evaluating on X_val/y_val. Supports
        early stopping through Optuna's pruning mechanism.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            metric: Metric to optimize ('mae', 'rmse', or 'r2')
                   For 'mae' and 'rmse': minimize (lower is better)
                   For 'r2': maximize (higher is better)

        Returns:
            Dictionary containing best hyperparameters found

        Example:
            >>> best_params = tuner.optimize(X_train, y_train, X_val, y_val, metric="mae")
            >>> best_params
            {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 500}
        """
        logger.info(
            "Starting hyperparameter optimization",
            model_type=self.model_type,
            n_trials=self.n_trials,
            metric=metric,
            train_samples=len(X_train),
            val_samples=len(X_val),
        )

        # Create Optuna study
        # Use minimize direction for mae/rmse, maximize for r2
        direction = "maximize" if metric == "r2" else "minimize"
        study = optuna.create_study(
            direction=direction,
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        # Define objective function with access to training data
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function.

            Args:
                trial: Optuna trial object

            Returns:
                Metric value to optimize (minimize or maximize)
            """
            try:
                # Get hyperparameters from search space
                params = self._suggest_params(trial)

                logger.debug(
                    "Trial parameters",
                    trial_number=trial.number,
                    params=params,
                )

                # Create model instance with suggested hyperparameters
                model_class = get_model_class(self.model_type)
                model = model_class(**params)

                # Train model on training data
                model.fit(X_train, y_train)

                # Predict on validation data
                y_pred = model.predict(X_val)

                # Calculate metric
                if metric == "mae":
                    score = mean_absolute_error(y_val, y_pred)
                elif metric == "rmse":
                    score = mean_squared_error(y_val, y_pred, squared=False)
                elif metric == "r2":
                    score = r2_score(y_val, y_pred)
                else:
                    raise ValueError(
                        f"Unsupported metric '{metric}'. "
                        f"Supported: 'mae', 'rmse', 'r2'"
                    )

                logger.debug(
                    "Trial completed",
                    trial_number=trial.number,
                    metric=metric,
                    score=score,
                )

                return score

            except Exception as e:
                logger.error(
                    "Trial failed",
                    trial_number=trial.number,
                    error=str(e),
                )
                # Return worst possible value to prune this trial
                return float('inf') if direction == "minimize" else float('-inf')

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )

        # Log results
        logger.info(
            "Optimization completed",
            best_value=study.best_value,
            best_params=study.best_params,
            n_trials=len(study.trials),
        )

        return study.best_params

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters from search space.

        Reads the search space for the model type and uses Optuna's trial
        object to suggest parameter values. Handles both uniform and
        log-scale distributions.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters

        Example:
            For xgboost with search space:
            {
                "max_depth": (3, 10),
                "learning_rate": (0.01, 0.3, "log"),
            }

            Returns:
            {
                "max_depth": 7,  # Uniform int between 3-10
                "learning_rate": 0.05,  # Log-scale float between 0.01-0.3
            }
        """
        search_space = get_search_space(self.model_type)
        params = {}

        for param_name, param_range in search_space.items():
            # Check if log-scale distribution
            is_log = len(param_range) == 3 and param_range[2] == "log"

            # Determine if parameter is int or float based on range values
            min_val, max_val = param_range[0], param_range[1]
            is_int = isinstance(min_val, int) and isinstance(max_val, int)

            if is_int:
                # Integer parameter (uniform distribution)
                params[param_name] = trial.suggest_int(
                    param_name,
                    min_val,
                    max_val,
                )
            else:
                # Float parameter
                if is_log:
                    # Log-scale distribution
                    params[param_name] = trial.suggest_float(
                        param_name,
                        min_val,
                        max_val,
                        log=True,
                    )
                else:
                    # Uniform distribution
                    params[param_name] = trial.suggest_float(
                        param_name,
                        min_val,
                        max_val,
                    )

        return params
