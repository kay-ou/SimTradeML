# -*- coding: utf-8 -*-
"""
Unit tests for DataCollector class
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
from simtrademl.core.data.collector import DataCollector
from simtrademl.core.data.base import DataSource
from simtrademl.core.utils.config import Config


class MockDataSourceForCollector(DataSource):
    """Mock data source for testing collector"""

    def __init__(self, price_data=None, stocks=None):
        self.price_data = price_data or {}
        self.stocks = stocks or ['TEST1', 'TEST2']

    def get_stock_list(self):
        return self.stocks

    def get_trading_dates(self, start_date=None, end_date=None):
        dates = pd.date_range('2020-01-01', periods=200, freq='B')
        return dates.tolist()

    def get_price_data(self, stock, start_date=None, end_date=None, fields=None):
        if stock in self.price_data:
            return self.price_data[stock]

        # Generate mock price data
        dates = pd.date_range('2020-01-01', periods=150, freq='B')
        np.random.seed(42)
        base_price = 100.0
        prices = base_price + np.cumsum(np.random.randn(150) * 2)
        prices = np.maximum(prices, 50)  # Ensure positive prices

        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 150)
        }, index=dates)

    def get_fundamentals(self, stock, date, fields=None):
        return {'pe_ratio': 15.5}

    def get_market_data(self, benchmark='000300.SS', start_date=None, end_date=None):
        return self.get_price_data(benchmark, start_date, end_date)


@pytest.mark.unit
class TestDataCollector:
    """Test DataCollector class"""

    def test_collector_initialization(self):
        """Test collector initialization"""
        data_source = MockDataSourceForCollector()
        config = Config.from_dict({
            'data': {'lookback_days': 60, 'predict_days': 5, 'sampling_window_days': 15},
            'training': {'parallel_jobs': 1}
        })
        collector = DataCollector(data_source, config)

        assert collector.lookback_days == 60
        assert collector.predict_days == 5
        assert collector.sampling_window_days == 15
        assert collector.n_jobs == 1

    def test_collector_with_auto_parallel_jobs(self):
        """Test collector with -1 parallel jobs (auto-detect)"""
        data_source = MockDataSourceForCollector()
        config = Config.from_dict({
            'data': {'lookback_days': 60, 'predict_days': 5},
            'training': {'parallel_jobs': -1}
        })
        collector = DataCollector(data_source, config)
        assert collector.n_jobs >= 1

    def test_generate_windows(self):
        """Test window generation"""
        data_source = MockDataSourceForCollector()
        config = Config.from_dict({
            'data': {'lookback_days': 60, 'predict_days': 5, 'sampling_window_days': 15},
            'training': {'parallel_jobs': 1}
        })
        collector = DataCollector(data_source, config)

        trading_dates = data_source.get_trading_dates()
        windows = collector._generate_windows(trading_dates)

        assert len(windows) > 0
        assert all(isinstance(w, pd.Timestamp) for w in windows)

    def test_calculate_default_features(self):
        """Test default feature calculation"""
        data_source = MockDataSourceForCollector()
        config = Config.from_dict({
            'data': {'lookback_days': 60, 'predict_days': 5},
            'training': {'parallel_jobs': 1}
        })
        collector = DataCollector(data_source, config)

        # Create sample price data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        price_df = pd.DataFrame({
            'close': np.linspace(100, 110, 100),
            'high': np.linspace(101, 111, 100),
            'low': np.linspace(99, 109, 100),
            'volume': [1000000] * 100
        }, index=dates)

        features = collector._calculate_default_features(price_df, window_idx=80)

        assert features is not None
        assert 'ma5' in features
        assert 'ma10' in features
        assert 'ma20' in features
        assert 'return_5d' in features
        assert 'return_10d' in features
        assert 'volatility_20d' in features
        assert all(np.isfinite(v) for v in features.values())

    def test_calculate_default_features_insufficient_data(self):
        """Test feature calculation with insufficient data"""
        data_source = MockDataSourceForCollector()
        config = Config.from_dict({
            'data': {'lookback_days': 60, 'predict_days': 5},
            'training': {'parallel_jobs': 1}
        })
        collector = DataCollector(data_source, config)

        # Create minimal price data
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        price_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        }, index=dates)

        features = collector._calculate_default_features(price_df, window_idx=5)
        assert features is None

    def test_calculate_default_features_with_zero_prices(self):
        """Test feature calculation rejects zero prices"""
        data_source = MockDataSourceForCollector()
        config = Config.from_dict({
            'data': {'lookback_days': 60, 'predict_days': 5},
            'training': {'parallel_jobs': 1}
        })
        collector = DataCollector(data_source, config)

        # Create price data with zero
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        price_df = pd.DataFrame({
            'close': [100.0] * 50 + [0.0] + [100.0] * 49
        }, index=dates)

        features = collector._calculate_default_features(price_df, window_idx=80)
        assert features is None

    def test_calculate_default_features_with_inf(self):
        """Test feature calculation rejects inf prices"""
        data_source = MockDataSourceForCollector()
        config = Config.from_dict({
            'data': {'lookback_days': 60, 'predict_days': 5},
            'training': {'parallel_jobs': 1}
        })
        collector = DataCollector(data_source, config)

        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        price_df = pd.DataFrame({
            'close': [100.0] * 50 + [np.inf] + [100.0] * 49
        }, index=dates)

        features = collector._calculate_default_features(price_df, window_idx=80)
        assert features is None

    def test_calculate_default_features_with_negative_prices(self):
        """Test feature calculation rejects negative prices"""
        data_source = MockDataSourceForCollector()
        config = Config.from_dict({
            'data': {'lookback_days': 60, 'predict_days': 5},
            'training': {'parallel_jobs': 1}
        })
        collector = DataCollector(data_source, config)

        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        price_df = pd.DataFrame({
            'close': [100.0] * 50 + [-50.0] + [100.0] * 49
        }, index=dates)

        features = collector._calculate_default_features(price_df, window_idx=80)
        assert features is None

    def test_collect_single_process(self):
        """Test data collection with single process"""
        data_source = MockDataSourceForCollector(stocks=['TEST1'])
        config = Config.from_dict({
            'data': {'lookback_days': 30, 'predict_days': 5, 'sampling_window_days': 20},
            'training': {'parallel_jobs': 1}
        })
        collector = DataCollector(data_source, config)

        X, y, dates = collector.collect()

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, np.ndarray)
        assert isinstance(dates, pd.Series)
        assert len(X) == len(y) == len(dates)
        assert len(X) > 0

    def test_collect_with_stock_filter(self):
        """Test data collection with stock filter"""
        data_source = MockDataSourceForCollector(stocks=['TEST1', 'TEST2', 'TEST3'])
        config = Config.from_dict({
            'data': {'lookback_days': 30, 'predict_days': 5, 'sampling_window_days': 20},
            'training': {'parallel_jobs': 1}
        })
        collector = DataCollector(data_source, config)

        def stock_filter(stock):
            return stock == 'TEST1'

        X, y, dates = collector.collect(stock_filter=stock_filter)

        assert len(X) > 0

    def test_collect_with_custom_feature_calculator(self):
        """Test data collection with custom feature calculator"""
        data_source = MockDataSourceForCollector(stocks=['TEST1'])
        config = Config.from_dict({
            'data': {'lookback_days': 30, 'predict_days': 5, 'sampling_window_days': 20},
            'training': {'parallel_jobs': 1}
        })

        def custom_feature_calculator(stock, price_df, window_idx, window_date, data_source):
            return {'custom_feature': 1.0}

        collector = DataCollector(data_source, config, feature_calculator=custom_feature_calculator)
        X, y, dates = collector.collect()

        assert len(X) > 0
        assert 'custom_feature' in X.columns

    def test_process_window_with_invalid_data(self):
        """Test window processing with invalid data"""
        # Create data source with empty price data
        empty_df = pd.DataFrame()
        data_source = MockDataSourceForCollector(
            price_data={'INVALID': empty_df},
            stocks=['INVALID']
        )
        config = Config.from_dict({
            'data': {'lookback_days': 30, 'predict_days': 5},
            'training': {'parallel_jobs': 1}
        })
        collector = DataCollector(data_source, config)

        window_date = pd.Timestamp('2020-03-01')
        samples, targets, dates = collector._process_window(window_date, ['INVALID'])

        assert len(samples) == 0
        assert len(targets) == 0
        assert len(dates) == 0
