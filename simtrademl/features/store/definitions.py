"""Feature view definitions for the feature store.

This module defines feature views that group related features together. Feature views
provide logical grouping and enable efficient storage and retrieval of features.

Each feature view represents a logical group of features:
- momentum_features: Technical indicators related to price momentum
- volatility_features: Indicators measuring market volatility
- volume_features: Trading volume-related indicators

The FEATURE_VIEWS dictionary is used by the feature store to validate feature names
and organize features in the database.
"""

from typing import Dict, List

# Feature view definitions
FEATURE_VIEWS: Dict[str, List[str]] = {
    "momentum_features": [
        "rsi_14",  # Relative Strength Index (14-period)
        "macd",  # Moving Average Convergence Divergence
        "macd_signal",  # MACD Signal Line
        "macd_hist",  # MACD Histogram (MACD - Signal)
        "momentum_20",  # Rate of Change (20-period)
    ],
    "volatility_features": [
        "atr_14",  # Average True Range (14-period)
        "bb_upper",  # Bollinger Bands Upper Band
        "bb_middle",  # Bollinger Bands Middle Band (SMA)
        "bb_lower",  # Bollinger Bands Lower Band
        "historical_volatility_20",  # Historical Volatility (20-period)
    ],
    "volume_features": [
        "volume_sma_20",  # Volume Simple Moving Average (20-period)
        "volume_std_20",  # Volume Standard Deviation (20-period)
        "obv",  # On-Balance Volume
        "volume_ratio",  # Current Volume / Average Volume
    ],
}


def get_feature_view_names() -> List[str]:
    """Get list of all feature view names.

    Returns:
        List of feature view names
    """
    return list(FEATURE_VIEWS.keys())


def get_features_for_view(view_name: str) -> List[str]:
    """Get list of features for a specific view.

    Args:
        view_name: Name of the feature view

    Returns:
        List of feature names in the view

    Raises:
        KeyError: If view_name is not found
    """
    return FEATURE_VIEWS[view_name]


def validate_features(view_name: str, feature_names: List[str]) -> bool:
    """Validate that feature names match the feature view definition.

    Args:
        view_name: Name of the feature view
        feature_names: List of feature names to validate

    Returns:
        True if all feature names are valid for the view

    Raises:
        KeyError: If view_name is not found
    """
    expected_features = set(FEATURE_VIEWS[view_name])
    provided_features = set(feature_names)
    return provided_features.issubset(expected_features)
