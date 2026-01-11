# -*- coding: utf-8 -*-
"""
Unit tests for DataSource base class
"""

import pytest
import pandas as pd
from simtrademl.core.data.base import DataSource


class MockDataSource(DataSource):
    """Mock implementation for testing"""

    def __init__(self, stocks=None, should_fail=False):
        if stocks is not None:
            self.stocks = stocks
        else:
            self.stocks = ['600519.SS', '000858.SZ']
        self.should_fail = should_fail

    def get_stock_list(self):
        return self.stocks

    def get_trading_dates(self, start_date=None, end_date=None):
        return pd.date_range('2020-01-01', periods=100, freq='B').tolist()

    def get_price_data(self, stock, start_date=None, end_date=None, fields=None):
        dates = pd.date_range('2020-01-01', periods=50, freq='B')
        return pd.DataFrame({
            'open': [100.0] * 50,
            'high': [102.0] * 50,
            'low': [99.0] * 50,
            'close': [101.0] * 50,
            'volume': [1000000] * 50,
        }, index=dates)

    def get_fundamentals(self, stock, date, fields=None):
        return {'pe_ratio': 15.5, 'pb_ratio': 2.3}

    def get_market_data(self, benchmark='000300.SS', start_date=None, end_date=None):
        return self.get_price_data(benchmark, start_date, end_date)


@pytest.mark.unit
class TestDataSource:
    """Test DataSource base class"""

    def test_supports_feature_type_default(self):
        """Test default feature type support"""
        ds = MockDataSource()
        assert ds.supports_feature_type('price')
        assert ds.supports_feature_type('fundamental')
        assert ds.supports_feature_type('custom')

    def test_validate_success(self):
        """Test validation with valid data"""
        ds = MockDataSource(stocks=['600519.SS'])
        assert ds.validate()

    def test_validate_failure_no_stocks(self):
        """Test validation fails with no stocks"""
        ds = MockDataSource(stocks=[])
        with pytest.raises(ValueError):
            ds.validate()

    def test_get_stock_list(self):
        """Test getting stock list"""
        ds = MockDataSource()
        stocks = ds.get_stock_list()
        assert len(stocks) == 2
        assert '600519.SS' in stocks

    def test_get_trading_dates(self):
        """Test getting trading dates"""
        ds = MockDataSource()
        dates = ds.get_trading_dates()
        assert len(dates) > 0
        assert all(isinstance(d, pd.Timestamp) for d in dates)

    def test_get_price_data(self):
        """Test getting price data"""
        ds = MockDataSource()
        df = ds.get_price_data('600519.SS')
        assert not df.empty
        assert 'close' in df.columns
        assert 'volume' in df.columns

    def test_get_fundamentals(self):
        """Test getting fundamental data"""
        ds = MockDataSource()
        data = ds.get_fundamentals('600519.SS', pd.Timestamp('2020-01-01'))
        assert data is not None
        assert 'pe_ratio' in data

    def test_get_market_data(self):
        """Test getting market data"""
        ds = MockDataSource()
        df = ds.get_market_data()
        assert not df.empty
