# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary"""
    return {
        'data': {
            'lookback_days': 60,
            'predict_days': 5,
            'sampling_window_days': 15,
        },
        'model': {
            'type': 'xgboost',
            'params': {
                'max_depth': 4,
                'learning_rate': 0.04,
            }
        }
    }


@pytest.fixture
def sample_predictions():
    """Sample prediction array"""
    np.random.seed(42)
    return np.random.randn(100)


@pytest.fixture
def sample_actuals():
    """Sample actual values array"""
    np.random.seed(43)
    return np.random.randn(100)


@pytest.fixture
def sample_dates():
    """Sample date array"""
    return pd.date_range('2020-01-01', periods=100, freq='D')
