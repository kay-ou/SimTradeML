# -*- coding: utf-8 -*-
"""
Unit tests for Feature Registry
"""

import pytest
import pandas as pd
import numpy as np

from simtrademl.features.registry import FeatureRegistry


@pytest.fixture
def sample_price_df():
    """Create sample price data for testing"""
    dates = pd.date_range('2024-01-01', periods=100)
    data = {
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000000, 5000000, 100),
        'high': 102 + np.cumsum(np.random.randn(100) * 0.5),
        'low': 98 + np.cumsum(np.random.randn(100) * 0.5),
    }
    return pd.DataFrame(data, index=dates)


class TestFeatureRegistry:
    """Test Feature Registry"""

    def test_register_feature(self):
        """Test registering a feature"""
        FeatureRegistry.clear()

        @FeatureRegistry.register('test_feature', category='test')
        def test_func(price_df):
            return 42.0

        assert 'test_feature' in FeatureRegistry._features

    def test_duplicate_registration_error(self):
        """Test that duplicate registration raises error"""
        FeatureRegistry.clear()

        @FeatureRegistry.register('dup_feature')
        def func1(price_df):
            return 1.0

        with pytest.raises(ValueError, match="already registered"):
            @FeatureRegistry.register('dup_feature')
            def func2(price_df):
                return 2.0

    def test_get_feature(self):
        """Test getting feature function"""
        FeatureRegistry.clear()

        @FeatureRegistry.register('get_test')
        def test_func(price_df):
            return 123.0

        func = FeatureRegistry.get('get_test')
        assert func is not None
        assert callable(func)

    def test_get_nonexistent_feature(self):
        """Test getting non-existent feature raises error"""
        FeatureRegistry.clear()

        with pytest.raises(KeyError, match="not registered"):
            FeatureRegistry.get('nonexistent')

    def test_calculate_single_feature(self, sample_price_df):
        """Test calculating a single feature"""
        FeatureRegistry.clear()

        @FeatureRegistry.register('simple_ma')
        def simple_ma(price_df):
            return price_df['close'].mean()

        result = FeatureRegistry.calculate(['simple_ma'], sample_price_df)

        assert isinstance(result, dict)
        assert 'simple_ma' in result
        assert isinstance(result['simple_ma'], (int, float))

    def test_calculate_multiple_features(self, sample_price_df):
        """Test calculating multiple features"""
        FeatureRegistry.clear()

        @FeatureRegistry.register('feat1')
        def feat1(price_df):
            return price_df['close'].iloc[-1]

        @FeatureRegistry.register('feat2')
        def feat2(price_df):
            return price_df['volume'].mean()

        result = FeatureRegistry.calculate(['feat1', 'feat2'], sample_price_df)

        assert len(result) == 2
        assert 'feat1' in result
        assert 'feat2' in result

    def test_calculate_with_error(self, sample_price_df):
        """Test that calculation error is properly reported"""
        FeatureRegistry.clear()

        @FeatureRegistry.register('error_feat')
        def error_feat(price_df):
            raise ValueError("Test error")

        with pytest.raises(RuntimeError, match="Error calculating feature"):
            FeatureRegistry.calculate(['error_feat'], sample_price_df)

    def test_list_features(self):
        """Test listing features"""
        FeatureRegistry.clear()

        @FeatureRegistry.register('list_test1', category='technical')
        def feat1(price_df):
            return 1.0

        @FeatureRegistry.register('list_test2', category='fundamental')
        def feat2(price_df):
            return 2.0

        all_features = FeatureRegistry.list_features()
        assert len(all_features) == 2

        # Test category filter
        tech_features = FeatureRegistry.list_features(category='technical')
        assert len(tech_features) == 1
        assert tech_features[0]['name'] == 'list_test1'

    def test_get_info(self):
        """Test getting feature info"""
        FeatureRegistry.clear()

        @FeatureRegistry.register(
            'info_test',
            category='technical',
            version='1.0',
            description='Test description'
        )
        def test_func(price_df):
            return 1.0

        info = FeatureRegistry.get_info('info_test')

        assert info['name'] == 'info_test'
        assert info['category'] == 'technical'
        assert info['version'] == '1.0'
        assert info['description'] == 'Test description'
        assert 'function' not in info  # Function should not be exposed


class TestTechnicalFeatures:
    """Test auto-registered technical features (isolated)"""

    @pytest.fixture(autouse=True)
    def reload_features(self):
        """Reload features before each test in this class"""
        import importlib
        from simtrademl.features import FeatureRegistry
        import simtrademl.features.technical

        # Clear and reload to ensure features are registered
        FeatureRegistry.clear()
        importlib.reload(simtrademl.features.technical)

        yield

        # Reload again after test to restore for other tests
        FeatureRegistry.clear()
        importlib.reload(simtrademl.features.technical)

    def test_technical_features_registered(self):
        """Test that technical features are auto-registered"""
        from simtrademl.features import FeatureRegistry

        # Check that MVP features are registered
        expected_features = [
            'ma5', 'ma10', 'ma20', 'ma60',
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            'volatility_20d', 'volume_ratio', 'price_position'
        ]

        registered_features = list(FeatureRegistry._features.keys())

        for feature_name in expected_features:
            assert feature_name in registered_features, \
                f"Feature '{feature_name}' not found. Registered: {registered_features}"

    def test_technical_feature_calculation(self, sample_price_df):
        """Test that technical features can be calculated"""
        from simtrademl.features import FeatureRegistry

        # Calculate all MVP features
        feature_names = [
            'ma5', 'ma10', 'ma20', 'ma60',
            'return_1d', 'return_5d', 'return_10d', 'return_20d',
            'volatility_20d', 'volume_ratio', 'price_position'
        ]

        results = FeatureRegistry.calculate(feature_names, sample_price_df)

        # Check all features calculated
        assert len(results) == len(feature_names)
        for name in feature_names:
            assert name in results
            assert isinstance(results[name], (int, float))
            assert np.isfinite(results[name])

    def test_extended_technical_features_registered(self):
        """Test that all extended technical features are registered"""
        from simtrademl.features import FeatureRegistry

        extended_features = [
            # Momentum indicators
            'rsi14', 'cci20', 'roc10', 'williams_r14',
            # Trend indicators
            'macd', 'macd_signal', 'macd_histogram', 'bb_upper', 'bb_lower', 'atr14',
            # KDJ indicators
            'kdj_k', 'kdj_d', 'kdj_j',
            # MA ratios and differences
            'ma5_ma10_ratio', 'ma5_ma20_ratio', 'ma10_ma20_ratio', 'ma20_ma60_ratio',
            'price_ma5_diff', 'price_ma10_diff', 'price_ma20_diff', 'price_ma60_diff'
        ]

        registered_features = list(FeatureRegistry._features.keys())

        for feature_name in extended_features:
            assert feature_name in registered_features, \
                f"Extended feature '{feature_name}' not found"

    def test_momentum_indicators(self, sample_price_df):
        """Test momentum indicators calculation"""
        from simtrademl.features import FeatureRegistry

        momentum_features = ['rsi14', 'cci20', 'roc10', 'williams_r14']
        results = FeatureRegistry.calculate(momentum_features, sample_price_df)

        assert len(results) == len(momentum_features)
        for name in momentum_features:
            assert name in results
            assert isinstance(results[name], (int, float))
            assert np.isfinite(results[name])

    def test_trend_indicators(self, sample_price_df):
        """Test trend indicators calculation"""
        from simtrademl.features import FeatureRegistry

        trend_features = ['macd', 'macd_signal', 'macd_histogram',
                         'bb_upper', 'bb_lower', 'atr14']
        results = FeatureRegistry.calculate(trend_features, sample_price_df)

        assert len(results) == len(trend_features)
        for name in trend_features:
            assert name in results
            assert isinstance(results[name], (int, float))
            assert np.isfinite(results[name])

    def test_kdj_indicators(self, sample_price_df):
        """Test KDJ indicators calculation"""
        from simtrademl.features import FeatureRegistry

        kdj_features = ['kdj_k', 'kdj_d', 'kdj_j']
        results = FeatureRegistry.calculate(kdj_features, sample_price_df)

        assert len(results) == len(kdj_features)
        for name in kdj_features:
            assert name in results
            assert isinstance(results[name], (int, float))
            assert np.isfinite(results[name])

    def test_ma_ratios(self, sample_price_df):
        """Test MA ratio indicators calculation"""
        from simtrademl.features import FeatureRegistry

        ma_ratio_features = ['ma5_ma10_ratio', 'ma5_ma20_ratio',
                            'ma10_ma20_ratio', 'ma20_ma60_ratio']
        results = FeatureRegistry.calculate(ma_ratio_features, sample_price_df)

        assert len(results) == len(ma_ratio_features)
        for name in ma_ratio_features:
            assert name in results
            assert isinstance(results[name], (int, float))
            assert np.isfinite(results[name])

    def test_price_ma_diffs(self, sample_price_df):
        """Test price-MA difference indicators calculation"""
        from simtrademl.features import FeatureRegistry

        diff_features = ['price_ma5_diff', 'price_ma10_diff',
                        'price_ma20_diff', 'price_ma60_diff']
        results = FeatureRegistry.calculate(diff_features, sample_price_df)

        assert len(results) == len(diff_features)
        for name in diff_features:
            assert name in results
            assert isinstance(results[name], (int, float))
            assert np.isfinite(results[name])

    def test_insufficient_data_handling(self):
        """Test features handle insufficient data gracefully"""
        from simtrademl.features import FeatureRegistry

        # Create minimal price data
        dates = pd.date_range('2024-01-01', periods=5)
        data = {
            'close': [100, 101, 102, 101, 103],
            'volume': [1000000] * 5,
            'high': [102, 103, 104, 103, 105],
            'low': [98, 99, 100, 99, 101],
        }
        short_df = pd.DataFrame(data, index=dates)

        # These features should return numeric values (may be NaN for some)
        features_to_test = ['ma60', 'return_20d', 'volatility_20d', 'rsi14', 'macd']

        for feature_name in features_to_test:
            result = FeatureRegistry.calculate([feature_name], short_df)
            assert feature_name in result
            assert isinstance(result[feature_name], (int, float))

    def test_edge_cases(self, sample_price_df):
        """Test features with edge case data"""
        from simtrademl.features import FeatureRegistry

        # Create data with constant prices (zero volatility)
        dates = pd.date_range('2024-01-01', periods=100)
        data = {
            'close': [100.0] * 100,
            'volume': [1000000] * 100,
            'high': [100.0] * 100,
            'low': [100.0] * 100,
        }
        constant_df = pd.DataFrame(data, index=dates)

        # These features should handle zero volatility gracefully
        features_to_test = ['volatility_20d', 'price_position', 'rsi14',
                           'williams_r14', 'kdj_k']

        for feature_name in features_to_test:
            result = FeatureRegistry.calculate([feature_name], constant_df)
            assert feature_name in result
            assert isinstance(result[feature_name], (int, float))
            assert np.isfinite(result[feature_name])
