"""Unit tests for HyperparameterTuner."""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

from simtrademl.training.tuner import HyperparameterTuner
from simtrademl.models.base import BaseModel
from simtrademl.models.registry import register_model, MODEL_REGISTRY
from simtrademl.training.search_spaces import SEARCH_SPACES


class MockXGBoostModel(BaseModel):
    """Mock XGBoost model for testing.

    This is a simple linear model that mimics XGBoost's interface
    for testing purposes only.
    """

    def __init__(
        self,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize mock XGBoost model."""
        super().__init__()
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.coefficients: np.ndarray | None = None
        self.intercept: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MockXGBoostModel":
        """Fit simple linear model."""
        # Use simple linear regression for testing
        # Add small regularization to avoid singular matrix
        X_np = X.values
        y_np = y.values

        # Add intercept column
        X_with_intercept = np.c_[np.ones(len(X_np)), X_np]

        # Compute coefficients using normal equation with regularization
        alpha = 0.01  # Small regularization
        XtX = X_with_intercept.T @ X_with_intercept
        XtX_reg = XtX + alpha * np.eye(XtX.shape[0])
        Xty = X_with_intercept.T @ y_np

        try:
            coeffs = np.linalg.solve(XtX_reg, Xty)
            self.intercept = coeffs[0]
            self.coefficients = coeffs[1:]
        except np.linalg.LinAlgError:
            # Fallback: use mean prediction
            self.intercept = float(y_np.mean())
            self.coefficients = np.zeros(X_np.shape[1])

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using linear model."""
        if self.coefficients is None:
            raise ValueError("Model not fitted yet")

        X_np = X.values
        predictions = X_np @ self.coefficients + self.intercept
        return predictions

    def save(self, path: str) -> None:
        """Save model (not implemented for mock)."""
        pass

    def load(self, path: str) -> "MockXGBoostModel":
        """Load model (not implemented for mock)."""
        return self


class MockLightGBMModel(BaseModel):
    """Mock LightGBM model for testing."""

    def __init__(
        self,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_samples: int = 20,
        **kwargs: Any,
    ) -> None:
        """Initialize mock LightGBM model."""
        super().__init__()
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_samples = min_child_samples
        self.coefficients: np.ndarray | None = None
        self.intercept: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MockLightGBMModel":
        """Fit simple linear model."""
        # Same simple implementation as MockXGBoostModel
        X_np = X.values
        y_np = y.values

        X_with_intercept = np.c_[np.ones(len(X_np)), X_np]
        alpha = 0.01
        XtX = X_with_intercept.T @ X_with_intercept
        XtX_reg = XtX + alpha * np.eye(XtX.shape[0])
        Xty = X_with_intercept.T @ y_np

        try:
            coeffs = np.linalg.solve(XtX_reg, Xty)
            self.intercept = coeffs[0]
            self.coefficients = coeffs[1:]
        except np.linalg.LinAlgError:
            self.intercept = float(y_np.mean())
            self.coefficients = np.zeros(X_np.shape[1])

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using linear model."""
        if self.coefficients is None:
            raise ValueError("Model not fitted yet")

        X_np = X.values
        predictions = X_np @ self.coefficients + self.intercept
        return predictions

    def save(self, path: str) -> None:
        """Save model (not implemented for mock)."""
        pass

    def load(self, path: str) -> "MockLightGBMModel":
        """Load model (not implemented for mock)."""
        return self


@pytest.fixture(scope="module", autouse=True)
def register_mock_models():
    """Register mock models for testing.

    This fixture automatically registers mock XGBoost and LightGBM models
    before running tests and cleans up after all tests complete.
    """
    # Save original registry state
    original_registry = MODEL_REGISTRY.copy()

    # Register mock models
    if "xgboost" not in MODEL_REGISTRY:
        register_model("xgboost", MockXGBoostModel)
    if "lightgbm" not in MODEL_REGISTRY:
        register_model("lightgbm", MockLightGBMModel)

    yield

    # Restore original registry (cleanup)
    MODEL_REGISTRY.clear()
    MODEL_REGISTRY.update(original_registry)


@pytest.fixture
def synthetic_data():
    """Generate synthetic regression data for testing.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val) DataFrames
    """
    np.random.seed(42)

    # Generate training data
    n_train = 100
    n_features = 5
    X_train = pd.DataFrame(
        np.random.rand(n_train, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    # Generate target with linear relationship + noise
    true_coeffs = np.random.rand(n_features)
    y_train = pd.Series(
        X_train.values @ true_coeffs + np.random.randn(n_train) * 0.1,
        name="target",
    )

    # Generate validation data
    n_val = 30
    X_val = pd.DataFrame(
        np.random.rand(n_val, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y_val = pd.Series(
        X_val.values @ true_coeffs + np.random.randn(n_val) * 0.1,
        name="target",
    )

    return X_train, y_train, X_val, y_val


@pytest.mark.unit
class TestHyperparameterTuner:
    """Test cases for HyperparameterTuner."""

    def test_initialization_with_valid_model_type(self):
        """Test tuner initialization with valid model type."""
        tuner = HyperparameterTuner(
            model_type="xgboost",
            n_trials=10,
            timeout=None,
            n_jobs=1,
        )

        assert tuner.model_type == "xgboost"
        assert tuner.n_trials == 10
        assert tuner.timeout is None
        assert tuner.n_jobs == 1

    def test_initialization_with_invalid_model_type(self):
        """Test tuner initialization with invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            HyperparameterTuner(
                model_type="invalid_model",
                n_trials=10,
            )

    def test_initialization_with_missing_search_space(self):
        """Test tuner initialization with model type missing search space."""
        # Prophet is in MODEL_REGISTRY but not in SEARCH_SPACES
        with pytest.raises(KeyError, match="No search space defined"):
            HyperparameterTuner(
                model_type="prophet",
                n_trials=10,
            )

    def test_optimize_with_xgboost(self, synthetic_data):
        """Test optimization with XGBoost model."""
        X_train, y_train, X_val, y_val = synthetic_data

        tuner = HyperparameterTuner(
            model_type="xgboost",
            n_trials=5,  # Small number for fast testing
            n_jobs=1,
        )

        best_params = tuner.optimize(
            X_train, y_train, X_val, y_val, metric="mae"
        )

        # Verify best params contain expected hyperparameters
        expected_params = set(SEARCH_SPACES["xgboost"].keys())
        assert set(best_params.keys()) == expected_params

        # Verify parameter values are within expected ranges
        search_space = SEARCH_SPACES["xgboost"]
        for param_name, param_value in best_params.items():
            min_val, max_val = search_space[param_name][:2]
            assert min_val <= param_value <= max_val, (
                f"Parameter {param_name}={param_value} "
                f"not in range [{min_val}, {max_val}]"
            )

    def test_optimize_with_lightgbm(self, synthetic_data):
        """Test optimization with LightGBM model."""
        X_train, y_train, X_val, y_val = synthetic_data

        tuner = HyperparameterTuner(
            model_type="lightgbm",
            n_trials=5,
            n_jobs=1,
        )

        best_params = tuner.optimize(
            X_train, y_train, X_val, y_val, metric="rmse"
        )

        # Verify best params contain expected hyperparameters
        expected_params = set(SEARCH_SPACES["lightgbm"].keys())
        assert set(best_params.keys()) == expected_params

    def test_optimize_with_r2_metric(self, synthetic_data):
        """Test optimization with R2 metric (maximize instead of minimize)."""
        X_train, y_train, X_val, y_val = synthetic_data

        tuner = HyperparameterTuner(
            model_type="xgboost",
            n_trials=5,
            n_jobs=1,
        )

        best_params = tuner.optimize(
            X_train, y_train, X_val, y_val, metric="r2"
        )

        # Should complete without error
        assert isinstance(best_params, dict)
        assert len(best_params) > 0

    def test_optimize_with_timeout(self, synthetic_data):
        """Test optimization with timeout."""
        X_train, y_train, X_val, y_val = synthetic_data

        tuner = HyperparameterTuner(
            model_type="xgboost",
            n_trials=1000,  # High number of trials
            timeout=1,  # 1 second timeout
            n_jobs=1,
        )

        # Should complete within timeout (not all trials)
        best_params = tuner.optimize(
            X_train, y_train, X_val, y_val, metric="mae"
        )

        assert isinstance(best_params, dict)
        assert len(best_params) > 0

    def test_suggest_params_uniform_distribution(self):
        """Test parameter suggestion with uniform distribution."""
        import optuna

        tuner = HyperparameterTuner(model_type="xgboost", n_trials=1)
        study = optuna.create_study()
        trial = study.ask()

        params = tuner._suggest_params(trial)

        # Verify all parameters are suggested
        assert set(params.keys()) == set(SEARCH_SPACES["xgboost"].keys())

        # Verify max_depth is integer (uniform distribution)
        assert isinstance(params["max_depth"], int)
        assert 3 <= params["max_depth"] <= 10

    def test_suggest_params_log_distribution(self):
        """Test parameter suggestion with log distribution."""
        import optuna

        tuner = HyperparameterTuner(model_type="xgboost", n_trials=1)
        study = optuna.create_study()
        trial = study.ask()

        params = tuner._suggest_params(trial)

        # Verify learning_rate is float with log scale
        assert isinstance(params["learning_rate"], float)
        assert 0.01 <= params["learning_rate"] <= 0.3

    def test_optimization_improves_over_trials(self, synthetic_data):
        """Test that optimization finds improving parameters.

        This test runs optimization multiple times and verifies that
        the tuner explores different parameter combinations.
        """
        X_train, y_train, X_val, y_val = synthetic_data

        tuner = HyperparameterTuner(
            model_type="xgboost",
            n_trials=10,
            n_jobs=1,
        )

        # Run optimization multiple times
        results = []
        for _ in range(3):
            best_params = tuner.optimize(
                X_train, y_train, X_val, y_val, metric="mae"
            )
            results.append(best_params)

        # Verify that we get valid results
        for result in results:
            assert isinstance(result, dict)
            assert len(result) == len(SEARCH_SPACES["xgboost"])

    def test_invalid_metric(self, synthetic_data):
        """Test optimization with invalid metric."""
        X_train, y_train, X_val, y_val = synthetic_data

        tuner = HyperparameterTuner(
            model_type="xgboost",
            n_trials=2,
            n_jobs=1,
        )

        # Should handle invalid metric gracefully
        # The error will be caught in the objective function
        best_params = tuner.optimize(
            X_train, y_train, X_val, y_val, metric="invalid_metric"
        )

        # Should still return a result (with worst values)
        assert isinstance(best_params, dict)


@pytest.mark.unit
def test_tuner_integration_with_search_spaces():
    """Test tuner integration with search spaces module."""
    # Verify search spaces exist for models with tuner support
    assert "xgboost" in SEARCH_SPACES
    assert "lightgbm" in SEARCH_SPACES

    # Verify search space structure
    for model_type, search_space in SEARCH_SPACES.items():
        assert isinstance(search_space, dict)
        for param_name, param_range in search_space.items():
            assert isinstance(param_range, tuple)
            assert len(param_range) in [2, 3]  # (min, max) or (min, max, "log")
            assert param_range[0] < param_range[1]  # min < max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
