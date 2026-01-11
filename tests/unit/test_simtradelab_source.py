# -*- coding: utf-8 -*-
"""
Integration tests for SimTradeLabDataSource
Tests interactions without requiring actual SimTradeLab installation
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
import sys


# Mock simtradelab module before importing
mock_simtradelab = MagicMock()
sys.modules['simtradelab'] = mock_simtradelab
sys.modules['simtradelab.research'] = MagicMock()
sys.modules['simtradelab.research.api'] = MagicMock()

from simtrademl.data_sources.simtradelab_source import SimTradeLabDataSource


@pytest.mark.unit
class TestSimTradeLabDataSource:
    """Test SimTradeLabDataSource class"""

    def setup_method(self):
        """Setup for each test"""
        # Reset mock
        mock_simtradelab.reset_mock()

    def test_get_stock_list_basic(self):
        """Test getting stock list"""
        # Create mock API
        mock_api = Mock()
        mock_api.get_Ashares.return_value = ['600519.SS', '000858.SZ']

        # Create data source and inject mock API
        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)
        ds.api = mock_api
        ds._stock_cache = None

        stocks = ds.get_stock_list()

        assert stocks == ['600519.SS', '000858.SZ']
        mock_api.get_Ashares.assert_called_once()

    def test_get_stock_list_caching(self):
        """Test stock list is cached"""
        mock_api = Mock()
        mock_api.get_Ashares.return_value = ['600519.SS']

        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)
        ds.api = mock_api
        ds._stock_cache = None

        stocks1 = ds.get_stock_list()
        stocks2 = ds.get_stock_list()

        assert stocks1 == stocks2
        mock_api.get_Ashares.assert_called_once()

    def test_get_stock_list_returns_empty_when_none(self):
        """Test stock list returns empty list when API returns None"""
        mock_api = Mock()
        mock_api.get_Ashares.return_value = None

        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)
        ds.api = mock_api
        ds._stock_cache = None

        stocks = ds.get_stock_list()

        assert stocks == []

    def test_get_trading_dates(self):
        """Test getting trading dates"""
        mock_api = Mock()
        dates = pd.date_range('2020-01-01', periods=10, freq='B')
        mock_df = pd.DataFrame({'close': [100] * 10}, index=dates)
        mock_api.get_price.return_value = mock_df

        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)
        ds.api = mock_api

        trading_dates = ds.get_trading_dates()

        assert len(trading_dates) == 10
        assert all(isinstance(d, pd.Timestamp) for d in trading_dates)

    def test_get_trading_dates_empty(self):
        """Test getting trading dates when API returns None"""
        mock_api = Mock()
        mock_api.get_price.return_value = None

        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)
        ds.api = mock_api

        trading_dates = ds.get_trading_dates()

        assert trading_dates == []

    def test_get_price_data(self):
        """Test getting price data"""
        mock_api = Mock()
        dates = pd.date_range('2020-01-01', periods=10, freq='B')
        mock_df = pd.DataFrame({
            'open': [100] * 10,
            'high': [102] * 10,
            'low': [98] * 10,
            'close': [101] * 10,
            'volume': [1000000] * 10
        }, index=dates)
        mock_api.get_price.return_value = mock_df

        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)
        ds.api = mock_api

        start_date = pd.Timestamp('2020-01-01')
        end_date = pd.Timestamp('2020-01-10')
        price_df = ds.get_price_data('600519.SS', start_date=start_date, end_date=end_date)

        assert not price_df.empty
        assert 'close' in price_df.columns

    def test_get_price_data_empty(self):
        """Test getting price data returns empty DataFrame when None"""
        mock_api = Mock()
        mock_api.get_price.return_value = None

        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)
        ds.api = mock_api

        price_df = ds.get_price_data('INVALID')

        assert price_df.empty
        assert 'close' in price_df.columns

    def test_get_fundamentals(self):
        """Test getting fundamental data"""
        mock_api = Mock()
        mock_result = pd.DataFrame({
            'pe_ratio': [15.5],
            'pb_ratio': [2.3]
        })
        mock_api.get_fundamentals.return_value = mock_result

        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)
        ds.api = mock_api

        date = pd.Timestamp('2020-01-01')
        fundamentals = ds.get_fundamentals('600519.SS', date)

        assert fundamentals is not None
        assert 'pe_ratio' in fundamentals
        assert fundamentals['pe_ratio'] == 15.5

    def test_get_fundamentals_empty(self):
        """Test getting fundamentals returns None when empty"""
        mock_api = Mock()
        mock_api.get_fundamentals.return_value = pd.DataFrame()

        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)
        ds.api = mock_api

        date = pd.Timestamp('2020-01-01')
        fundamentals = ds.get_fundamentals('INVALID', date)

        assert fundamentals is None

    def test_get_fundamentals_exception(self):
        """Test getting fundamentals handles exceptions"""
        mock_api = Mock()
        mock_api.get_fundamentals.side_effect = Exception("API error")

        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)
        ds.api = mock_api

        date = pd.Timestamp('2020-01-01')
        fundamentals = ds.get_fundamentals('600519.SS', date)

        assert fundamentals is None

    def test_get_market_data(self):
        """Test getting market data"""
        mock_api = Mock()
        dates = pd.date_range('2020-01-01', periods=10, freq='B')
        mock_df = pd.DataFrame({'close': [3000] * 10}, index=dates)
        mock_api.get_price.return_value = mock_df

        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)
        ds.api = mock_api

        market_df = ds.get_market_data()

        assert not market_df.empty

    def test_get_history_batch_series(self):
        """Test getting history batch with Series"""
        mock_api = Mock()
        mock_series = pd.Series([100, 101, 102, 103, 104])
        mock_api.get_history.return_value = mock_series

        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)
        ds.api = mock_api

        hist = ds.get_history_batch('600519.SS', count=5, field='close')

        assert isinstance(hist, np.ndarray)
        assert len(hist) == 5

    def test_get_history_batch_array(self):
        """Test getting history batch with numpy array"""
        mock_api = Mock()
        mock_array = np.array([100, 101, 102])
        mock_api.get_history.return_value = mock_array

        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)
        ds.api = mock_api

        hist = ds.get_history_batch('600519.SS', count=3, field='close')

        assert isinstance(hist, np.ndarray)
        assert len(hist) == 3

    def test_get_history_batch_none(self):
        """Test history batch returns empty array when None"""
        mock_api = Mock()
        mock_api.get_history.return_value = None

        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)
        ds.api = mock_api

        hist = ds.get_history_batch('INVALID', count=5, field='close')

        assert isinstance(hist, np.ndarray)
        assert len(hist) == 0

    def test_get_history_batch_exception(self):
        """Test history batch handles exceptions"""
        mock_api = Mock()
        mock_api.get_history.side_effect = Exception("API error")

        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)
        ds.api = mock_api

        hist = ds.get_history_batch('600519.SS', count=5, field='close')

        assert isinstance(hist, np.ndarray)
        assert len(hist) == 0

    def test_supports_feature_type(self):
        """Test feature type support check"""
        ds = SimTradeLabDataSource.__new__(SimTradeLabDataSource)

        assert ds.supports_feature_type('price')
        assert ds.supports_feature_type('fundamental')
        assert ds.supports_feature_type('market')
        assert ds.supports_feature_type('technical')
        assert not ds.supports_feature_type('unsupported')

