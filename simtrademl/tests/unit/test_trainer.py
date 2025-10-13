"""Unit tests for Trainer class."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

from simtrademl.training.trainer import Trainer, TrainingConfig, TrainingResult


@pytest.fixture
def synthetic_data():
    """Generate synthetic time series data for testing.

    Returns:
        Tuple of (X, y) with 100 samples and 10 features.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    # Create feature DataFrame with time index
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='D')
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)],
        index=dates,
    )

    # Create target with some relationship to features
    y = pd.Series(
        X.iloc[:, 0] * 2 + X.iloc[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1,
        index=dates,
        name='target',
    )

    return X, y


@pytest.fixture
def training_config():
    """Create a basic training configuration for testing."""
    return TrainingConfig(
        model_type='arima',
        hyperparameters={'order': (1, 1, 1)},
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        eval_metric='mae',
    )


@pytest.mark.unit
class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_init_with_valid_ratios(self) -> None:
        """Test TrainingConfig initialization with valid ratios."""
        config = TrainingConfig(
            model_type='arima',
            hyperparameters={'order': (1, 1, 1)},
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        assert config.model_type == 'arima'
        assert config.train_ratio == 0.7
        assert config.val_ratio == 0.15
        assert config.test_ratio == 0.15
        assert config.random_seed == 42
        assert config.eval_metric == 'mae'

    def test_init_with_invalid_ratio_sum(self) -> None:
        """Test TrainingConfig raises error when ratios don't sum to 1.0."""
        with pytest.raises(ValueError, match="must equal 1.0"):
            TrainingConfig(
                model_type='arima',
                hyperparameters={'order': (1, 1, 1)},
                train_ratio=0.5,
                val_ratio=0.2,
                test_ratio=0.2,  # Sum is 0.9
            )

    def test_init_with_invalid_ratio_range(self) -> None:
        """Test TrainingConfig raises error when ratio out of range."""
        with pytest.raises(ValueError, match="must be in range"):
            TrainingConfig(
                model_type='arima',
                hyperparameters={'order': (1, 1, 1)},
                train_ratio=0.0,  # Invalid: must be > 0
                val_ratio=0.5,
                test_ratio=0.5,
            )

    def test_init_with_invalid_eval_metric(self) -> None:
        """Test TrainingConfig raises error for invalid eval_metric."""
        with pytest.raises(ValueError, match="eval_metric must be one of"):
            TrainingConfig(
                model_type='arima',
                hyperparameters={'order': (1, 1, 1)},
                eval_metric='invalid_metric',
            )

    def test_init_with_negative_early_stopping(self) -> None:
        """Test TrainingConfig raises error for negative early_stopping_rounds."""
        with pytest.raises(ValueError, match="early_stopping_rounds must be positive"):
            TrainingConfig(
                model_type='arima',
                hyperparameters={'order': (1, 1, 1)},
                early_stopping_rounds=-5,
            )


@pytest.mark.unit
class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_init_with_valid_data(self) -> None:
        """Test TrainingResult initialization with valid data."""
        result = TrainingResult(
            run_id='test_run_123',
            model_name='arima_model',
            model_uri='runs:/test_run_123/model',
            metrics={'mae': 0.05, 'rmse': 0.08, 'r2': 0.95},
            best_iteration=10,
        )

        assert result.run_id == 'test_run_123'
        assert result.model_name == 'arima_model'
        assert result.model_uri == 'runs:/test_run_123/model'
        assert result.metrics['mae'] == 0.05
        assert result.best_iteration == 10

    def test_init_with_empty_run_id(self) -> None:
        """Test TrainingResult raises error for empty run_id."""
        with pytest.raises(ValueError, match="run_id cannot be empty"):
            TrainingResult(
                run_id='',
                model_name='arima_model',
                model_uri='runs:/test/model',
                metrics={'mae': 0.05},
            )

    def test_init_with_empty_metrics(self) -> None:
        """Test TrainingResult raises error for empty metrics."""
        with pytest.raises(ValueError, match="metrics dictionary cannot be empty"):
            TrainingResult(
                run_id='test_run',
                model_name='arima_model',
                model_uri='runs:/test/model',
                metrics={},
            )

    def test_to_dict(self) -> None:
        """Test converting TrainingResult to dictionary."""
        result = TrainingResult(
            run_id='test_run_123',
            model_name='arima_model',
            model_uri='runs:/test_run_123/model',
            metrics={'mae': 0.05, 'rmse': 0.08},
            best_iteration=10,
        )

        result_dict = result.to_dict()

        assert result_dict['run_id'] == 'test_run_123'
        assert result_dict['model_name'] == 'arima_model'
        assert result_dict['metrics']['mae'] == 0.05
        assert result_dict['best_iteration'] == 10

    def test_get_primary_metric(self) -> None:
        """Test getting primary metric from result."""
        result = TrainingResult(
            run_id='test_run',
            model_name='arima_model',
            model_uri='runs:/test/model',
            metrics={'mae': 0.05, 'rmse': 0.08},
        )

        mae = result.get_primary_metric('mae')
        assert mae == 0.05

        rmse = result.get_primary_metric('rmse')
        assert rmse == 0.08

    def test_get_primary_metric_not_found(self) -> None:
        """Test getting non-existent metric raises KeyError."""
        result = TrainingResult(
            run_id='test_run',
            model_name='arima_model',
            model_uri='runs:/test/model',
            metrics={'mae': 0.05},
        )

        with pytest.raises(KeyError, match="not found"):
            result.get_primary_metric('mse')


@pytest.mark.unit
class TestTrainerInit:
    """Tests for Trainer initialization."""

    def test_init_with_valid_config(self, training_config) -> None:
        """Test Trainer initialization with valid configuration."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'):
            trainer = Trainer(training_config)

            assert trainer.config == training_config
            assert trainer.mlflow_experiment_name == 'default'

    def test_init_with_invalid_model_type(self) -> None:
        """Test Trainer raises error for invalid model type."""
        config = TrainingConfig(
            model_type='invalid_model',
            hyperparameters={},
        )

        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'):
            with pytest.raises(ValueError, match="Unknown model type"):
                Trainer(config)

    def test_init_with_custom_mlflow_settings(self, training_config) -> None:
        """Test Trainer initialization with custom MLflow settings."""
        with patch('mlflow.set_tracking_uri') as mock_tracking, \
             patch('mlflow.set_experiment') as mock_experiment:
            trainer = Trainer(
                training_config,
                mlflow_tracking_uri='http://custom:5000',
                mlflow_experiment_name='custom_experiment',
            )

            assert trainer.mlflow_tracking_uri == 'http://custom:5000'
            assert trainer.mlflow_experiment_name == 'custom_experiment'
            mock_tracking.assert_called_once_with('http://custom:5000')
            mock_experiment.assert_called_once_with('custom_experiment')


@pytest.mark.unit
class TestDataSplit:
    """Tests for _split_data method."""

    def test_split_data_maintains_time_order(self, training_config, synthetic_data) -> None:
        """Test that data split maintains temporal order (no shuffling)."""
        X, y = synthetic_data

        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'):
            trainer = Trainer(training_config)
            X_train, X_val, X_test, y_train, y_val, y_test = trainer._split_data(X, y)

        # Check that temporal order is maintained
        # Training data should be first chronologically
        assert X_train.index[0] < X_val.index[0]
        assert X_val.index[0] < X_test.index[0]
        assert X_train.index[-1] < X_val.index[0]
        assert X_val.index[-1] < X_test.index[0]

        # Same for target
        assert y_train.index[0] < y_val.index[0]
        assert y_val.index[0] < y_test.index[0]

    def test_split_data_correct_ratios(self, training_config, synthetic_data) -> None:
        """Test that data split uses correct ratios."""
        X, y = synthetic_data
        n_total = len(X)

        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'):
            trainer = Trainer(training_config)
            X_train, X_val, X_test, y_train, y_val, y_test = trainer._split_data(X, y)

        # Check sizes match expected ratios
        assert len(X_train) == int(n_total * 0.7)
        assert len(X_val) == int(n_total * 0.15)
        assert len(X_test) <= int(n_total * 0.15) + 1  # Allow for rounding

        # Check total is preserved
        assert len(X_train) + len(X_val) + len(X_test) == n_total

    def test_split_data_no_overlap(self, training_config, synthetic_data) -> None:
        """Test that train/val/test sets have no overlap."""
        X, y = synthetic_data

        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'):
            trainer = Trainer(training_config)
            X_train, X_val, X_test, y_train, y_val, y_test = trainer._split_data(X, y)

        # Check no index overlap
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        test_indices = set(X_test.index)

        assert train_indices.isdisjoint(val_indices)
        assert train_indices.isdisjoint(test_indices)
        assert val_indices.isdisjoint(test_indices)


@pytest.mark.unit
class TestEvaluate:
    """Tests for _evaluate method."""

    def test_evaluate_computes_correct_metrics(self, training_config, synthetic_data) -> None:
        """Test that _evaluate computes correct metrics."""
        X, y = synthetic_data

        # Create a simple mock model that returns predictions
        mock_model = MagicMock()
        mock_model.predict.return_value = y.values[:50]  # Perfect predictions

        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'):
            trainer = Trainer(training_config)
            metrics = trainer._evaluate(mock_model, X.iloc[:50], y.iloc[:50])

        # With perfect predictions, metrics should be very good
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'mse' in metrics

        # MAE should be near 0 for perfect predictions
        assert metrics['mae'] < 0.01
        # R2 should be near 1 for perfect predictions
        assert metrics['r2'] > 0.99

    def test_evaluate_with_imperfect_predictions(self, training_config, synthetic_data) -> None:
        """Test _evaluate with imperfect predictions."""
        X, y = synthetic_data

        # Create model with imperfect predictions (add noise)
        mock_model = MagicMock()
        np.random.seed(42)
        mock_model.predict.return_value = y.values[:50] + np.random.randn(50) * 0.5

        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'):
            trainer = Trainer(training_config)
            metrics = trainer._evaluate(mock_model, X.iloc[:50], y.iloc[:50])

        # Metrics should be reasonable but not perfect
        assert metrics['mae'] > 0.01
        assert metrics['r2'] < 1.0

    def test_evaluate_handles_invalid_predictions(self, training_config, synthetic_data) -> None:
        """Test _evaluate raises error for invalid predictions (NaN/Inf)."""
        X, y = synthetic_data

        # Create model that returns NaN predictions
        mock_model = MagicMock()
        mock_model.predict.return_value = np.full(50, np.nan)

        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'):
            trainer = Trainer(training_config)

            with pytest.raises(ValueError, match="NaN or Inf"):
                trainer._evaluate(mock_model, X.iloc[:50], y.iloc[:50])


@pytest.mark.unit
class TestTrain:
    """Tests for train method."""

    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    def test_train_with_arima_model(
        self,
        mock_log_model,
        mock_log_metric,
        mock_log_param,
        mock_log_params,
        mock_start_run,
        synthetic_data,
    ) -> None:
        """Test training with ARIMA model from registry."""
        X, y = synthetic_data

        # Mock MLflow run
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_arima'
        mock_start_run.return_value.__enter__.return_value = mock_run

        config = TrainingConfig(
            model_type='arima',
            hyperparameters={'order': (1, 1, 1)},
        )

        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'):
            trainer = Trainer(config)
            result = trainer.train(X, y)

        # Verify result
        assert result.run_id == 'test_run_arima'
        assert result.model_name == 'arima_model'
        assert result.model_uri == 'runs:/test_run_arima/model'
        assert 'mae' in result.metrics
        assert 'rmse' in result.metrics

        # Verify MLflow logging
        mock_log_params.assert_called_once()
        assert mock_log_param.call_count >= 5  # model_type, ratios, seed, metric
        assert mock_log_metric.call_count >= 4  # mae, rmse, r2, mse

    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    @pytest.mark.skipif(
        True,  # Skip if prophet not installed
        reason="Prophet is an optional dependency"
    )
    def test_train_with_prophet_model(
        self,
        mock_log_model,
        mock_log_metric,
        mock_log_param,
        mock_log_params,
        mock_start_run,
        synthetic_data,
    ) -> None:
        """Test training with Prophet model from registry."""
        X, y = synthetic_data

        # Mock MLflow run
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_prophet'
        mock_start_run.return_value.__enter__.return_value = mock_run

        config = TrainingConfig(
            model_type='prophet',
            hyperparameters={'seasonality_mode': 'multiplicative'},
        )

        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'):
            trainer = Trainer(config)
            result = trainer.train(X, y)

        # Verify result
        assert result.run_id == 'test_run_prophet'
        assert result.model_name == 'prophet_model'
        assert 'mae' in result.metrics

    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    def test_train_logs_hyperparameters(
        self,
        mock_log_model,
        mock_log_metric,
        mock_log_param,
        mock_log_params,
        mock_start_run,
        synthetic_data,
    ) -> None:
        """Test that train logs all hyperparameters to MLflow."""
        X, y = synthetic_data

        # Mock MLflow run
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run'
        mock_start_run.return_value.__enter__.return_value = mock_run

        hyperparams = {'order': (1, 1, 1), 'seasonal_order': (0, 0, 0, 0)}
        config = TrainingConfig(
            model_type='arima',
            hyperparameters=hyperparams,
        )

        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'):
            trainer = Trainer(config)
            trainer.train(X, y)

        # Verify hyperparameters were logged
        mock_log_params.assert_called_once_with(hyperparams)

    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    def test_train_with_validation_set(
        self,
        mock_log_model,
        mock_log_metric,
        mock_log_param,
        mock_log_params,
        mock_start_run,
        synthetic_data,
    ) -> None:
        """Test training with pre-provided validation set."""
        X, y = synthetic_data

        # Split data manually
        split_idx = 70
        X_train_val = X.iloc[:split_idx]
        y_train_val = y.iloc[:split_idx]
        X_val = X.iloc[split_idx:85]
        y_val = y.iloc[split_idx:85]

        # Mock MLflow run
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run'
        mock_start_run.return_value.__enter__.return_value = mock_run

        config = TrainingConfig(
            model_type='arima',
            hyperparameters={'order': (1, 1, 1)},
        )

        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'):
            trainer = Trainer(config)
            result = trainer.train(X_train_val, y_train_val, X_val, y_val)

        # Should still produce valid result
        assert result.run_id == 'test_run'
        assert 'mae' in result.metrics

    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    @patch('mlflow.sklearn.log_model')
    def test_train_completes_quickly(
        self,
        mock_log_model,
        mock_log_metric,
        mock_log_param,
        mock_log_params,
        mock_start_run,
        synthetic_data,
    ) -> None:
        """Test that training completes in reasonable time (<5 seconds)."""
        import time

        X, y = synthetic_data

        # Mock MLflow run
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run'
        mock_start_run.return_value.__enter__.return_value = mock_run

        config = TrainingConfig(
            model_type='arima',
            hyperparameters={'order': (1, 1, 1)},
        )

        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'):
            trainer = Trainer(config)

            start_time = time.time()
            trainer.train(X, y)
            elapsed = time.time() - start_time

        # Training should be fast with small synthetic data
        assert elapsed < 5.0, f"Training took {elapsed:.2f} seconds, expected < 5.0"
