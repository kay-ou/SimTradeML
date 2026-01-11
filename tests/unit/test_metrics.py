# -*- coding: utf-8 -*-
"""
Unit tests for metrics functions
"""

import pytest
import numpy as np
from simtrademl.core.utils.metrics import (
    calculate_ic,
    calculate_rank_ic,
    calculate_icir,
    calculate_quantile_returns,
    calculate_direction_accuracy,
)


@pytest.mark.unit
class TestMetrics:
    """Test evaluation metrics"""

    def test_calculate_ic_perfect_correlation(self):
        """Test IC with perfect positive correlation"""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ic, p_value = calculate_ic(predictions, actuals)
        assert abs(ic - 1.0) < 0.01
        assert p_value < 0.05

    def test_calculate_ic_no_correlation(self):
        """Test IC with no correlation"""
        np.random.seed(42)
        predictions = np.random.randn(100)
        np.random.seed(43)
        actuals = np.random.randn(100)
        ic, p_value = calculate_ic(predictions, actuals)
        assert abs(ic) < 0.3  # Should be close to 0

    def test_calculate_ic_negative_correlation(self):
        """Test IC with negative correlation"""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actuals = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        ic, p_value = calculate_ic(predictions, actuals)
        assert ic < -0.9

    def test_calculate_ic_with_nan(self):
        """Test IC handles NaN values"""
        predictions = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        actuals = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        ic, p_value = calculate_ic(predictions, actuals)
        assert np.isfinite(ic)

    def test_calculate_ic_length_mismatch(self):
        """Test IC with mismatched lengths"""
        predictions = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            calculate_ic(predictions, actuals)

    def test_calculate_rank_ic(self):
        """Test Rank IC calculation"""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actuals = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
        rank_ic, p_value = calculate_rank_ic(predictions, actuals)
        assert rank_ic > 0.9
        assert p_value < 0.05

    def test_calculate_icir_short_data(self):
        """Test ICIR with short data"""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        icir, ic_std = calculate_icir(predictions, actuals)
        assert np.isfinite(icir)
        assert ic_std == 0.0

    def test_calculate_icir_long_data(self):
        """Test ICIR with sufficient data"""
        np.random.seed(42)
        predictions = np.random.randn(500)
        actuals = predictions + np.random.randn(500) * 0.1  # Add noise
        icir, ic_std = calculate_icir(predictions, actuals, window_size=100)
        assert np.isfinite(icir)
        assert ic_std > 0

    def test_calculate_quantile_returns_no_dates(self):
        """Test quantile returns without dates (global split)"""
        predictions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        actuals = np.array([0.01, 0.02, 0.01, 0.03, 0.02, 0.04, 0.03, 0.05, 0.04, 0.06])
        quantile_returns, long_short = calculate_quantile_returns(predictions, actuals)
        assert len(quantile_returns) == 5
        assert long_short >= 0  # Higher predictions should have higher returns

    def test_calculate_quantile_returns_with_dates(self, sample_predictions, sample_actuals, sample_dates):
        """Test quantile returns with daily rebalancing"""
        quantile_returns, long_short = calculate_quantile_returns(
            sample_predictions,
            sample_actuals,
            dates=sample_dates.values
        )
        assert len(quantile_returns) == 5
        assert np.isfinite(long_short)

    def test_calculate_direction_accuracy_perfect(self):
        """Test direction accuracy with perfect prediction"""
        predictions = np.array([1.0, -1.0, 2.0, -2.0, 1.0])
        actuals = np.array([0.5, -0.5, 1.0, -1.0, 0.3])
        accuracy = calculate_direction_accuracy(predictions, actuals)
        assert accuracy == 1.0

    def test_calculate_direction_accuracy_random(self, sample_predictions, sample_actuals):
        """Test direction accuracy with random predictions"""
        accuracy = calculate_direction_accuracy(sample_predictions, sample_actuals)
        assert 0.0 <= accuracy <= 1.0
        assert 0.3 <= accuracy <= 0.7  # Should be around 50% for random

    def test_calculate_direction_accuracy_with_nan(self):
        """Test direction accuracy handles NaN"""
        predictions = np.array([1.0, np.nan, -1.0])
        actuals = np.array([1.0, 1.0, np.nan])
        accuracy = calculate_direction_accuracy(predictions, actuals)
        assert np.isfinite(accuracy)

    def test_calculate_direction_accuracy_empty(self):
        """Test direction accuracy with empty arrays"""
        predictions = np.array([])
        actuals = np.array([])
        accuracy = calculate_direction_accuracy(predictions, actuals)
        assert accuracy == 0.0
