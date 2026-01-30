# -*- coding: utf-8 -*-
"""
SimTradeLab data source implementation
Reads data from local parquet files using SimTradeLab Research API
"""

from simtrademl.core.data.base import DataSource
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('simtrademl')


class SimTradeLabDataSource(DataSource):
    """Data source using SimTradeLab Research API

    Reads data from local parquet files in data/ directory.
    Copy SimTradeData parquet data directory to data/.

    Args:
        data_path: Path to data directory (default: auto-detect project root/data)
    """

    def __init__(self, data_path: Optional[str] = None):
        try:
            from simtradelab.research.api import init_api
        except ImportError:
            raise ImportError(
                "SimTradeLab not installed. "
                "Install with: pip install simtradelab"
            )

        # Initialize SimTradeLab Research API
        # It will automatically load data from data_path
        self.api = init_api(data_path=data_path)
        self._stock_cache = None

    def get_stock_list(self) -> List[str]:
        """Get all A-share stocks

        Returns:
            List of stock codes
        """
        if self._stock_cache is None:
            self._stock_cache = self.api.get_Ashares()

        # Ensure we always return a list, never None
        if self._stock_cache is None:
            return []

        return self._stock_cache

    def get_trading_dates(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[pd.Timestamp]:
        """Get trading dates from price data

        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)

        Returns:
            List of trading dates
        """
        # Use a liquid stock (CSI300) to get trading dates
        df = self.api.get_price(
            '000300.SS',
            start_date=start_date,
            end_date=end_date,
            fq='pre'
        )

        if df is None or df.empty:
            return []

        return df.index.tolist()

    def get_price_data(
        self,
        stock: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get price data using SimTradeLab get_price API

        Args:
            stock: Stock code
            start_date: Start date
            end_date: End date
            fields: Fields to retrieve (default: ['open', 'high', 'low', 'close', 'volume'])

        Returns:
            DataFrame with DatetimeIndex and price columns
        """
        # Convert Timestamp to string
        start_str = start_date.strftime('%Y-%m-%d') if start_date else None
        end_str = end_date.strftime('%Y-%m-%d') if end_date else None

        # Get price data
        df = self.api.get_price(
            stock,
            start_date=start_str,
            end_date=end_str,
            frequency='1d',
            fields=fields,
            fq='pre'  # Use forward-adjusted prices
        )

        if df is None or df.empty:
            # Return empty DataFrame with expected columns
            default_fields = fields or ['open', 'high', 'low', 'close', 'volume']
            return pd.DataFrame(columns=default_fields)

        return df

    def get_fundamentals(
        self,
        stock: str,
        date: pd.Timestamp,
        fields: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get fundamental data using SimTradeLab get_fundamentals API

        Args:
            stock: Stock code
            date: Query date
            fields: Fields to retrieve (default: all valuation fields)

        Returns:
            Dictionary of fundamental data
        """
        date_str = date.strftime('%Y-%m-%d')

        # Default fields: valuation metrics
        if fields is None:
            fields = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio',
                     'total_value', 'circulating_value']

        try:
            # Get valuation data
            result = self.api.get_fundamentals(
                [stock],
                'valuation',
                fields,
                date_str
            )

            if result is None or result.empty:
                return None

            # Convert to dict
            return result.iloc[0].to_dict() if len(result) > 0 else None

        except Exception as e:
            # Log but don't crash - fundamental data may not be available for all stocks/dates
            logger.debug(f"Failed to get fundamentals for {stock} at {date_str}: {e}")
            return None

    def get_market_data(
        self,
        benchmark: str = '000300.SS',
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """Get benchmark/market data

        Args:
            benchmark: Benchmark index code (default: CSI300)
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with market data
        """
        return self.get_price_data(benchmark, start_date, end_date)

    def get_history_batch(
        self,
        stock: str,
        count: int,
        field: str,
        end_date: Optional[pd.Timestamp] = None
    ) -> np.ndarray:
        """Get historical data for a single field (optimized for feature calculation)

        Args:
            stock: Stock code
            count: Number of bars to retrieve
            field: Field name ('close', 'open', 'high', 'low', 'volume')
            end_date: End date (optional)

        Returns:
            Numpy array of historical values
        """
        end_str = end_date.strftime('%Y-%m-%d') if end_date else None

        try:
            hist = self.api.get_history(count, stock, field, end_date=end_str, fq='pre')

            if hist is None:
                return np.array([])

            # Convert to numpy array
            if isinstance(hist, pd.Series):
                return np.asarray(hist.values)
            elif isinstance(hist, np.ndarray):
                return hist
            else:
                return np.array(hist)

        except Exception as e:
            logger.debug(f"Failed to get history for {stock} {field}: {e}")
            return np.array([])

    def supports_feature_type(self, feature_type: str) -> bool:
        """Check feature type support

        Args:
            feature_type: Feature type

        Returns:
            True if supported
        """
        supported_types = {'price', 'fundamental', 'market', 'technical'}
        return feature_type in supported_types
