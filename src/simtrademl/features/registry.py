# -*- coding: utf-8 -*-
"""
Feature Registry System

Centralized feature management for reusability and extensibility.
"""

from typing import Callable, Dict, Any, List, Optional
import pandas as pd
import numpy as np
from functools import wraps


class FeatureRegistry:
    """Central registry for feature definitions

    Allows features to be:
    - Registered once, used everywhere
    - Versioned and tracked
    - Easily extended

    Example:
        # Register a feature
        @FeatureRegistry.register('ma5', category='technical')
        def moving_average_5(price_df: pd.DataFrame) -> float:
            return price_df['close'].rolling(5).mean().iloc[-1]

        # Calculate features
        features = FeatureRegistry.calculate(['ma5', 'ma10'], price_df)
    """

    _features: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        category: str = 'technical',
        version: str = '1.0',
        description: str = ''
    ) -> Callable:
        """Decorator to register a feature calculation function

        Args:
            name: Feature name (must be unique)
            category: Feature category ('technical', 'fundamental', 'custom')
            version: Feature version
            description: Human-readable description

        Example:
            @FeatureRegistry.register('rsi14', category='technical')
            def rsi_14(price_df):
                return calculate_rsi(price_df['close'], 14).iloc[-1]
        """
        def decorator(func: Callable) -> Callable:
            if name in cls._features:
                raise ValueError(f"Feature '{name}' already registered")

            cls._features[name] = {
                'function': func,
                'category': category,
                'version': version,
                'description': description or func.__doc__ or 'No description'
            }

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        """Get feature calculation function by name

        Args:
            name: Feature name

        Returns:
            Feature calculation function

        Raises:
            KeyError: If feature not found
        """
        if name not in cls._features:
            raise KeyError(f"Feature '{name}' not registered")

        return cls._features[name]['function']

    @classmethod
    def calculate(
        cls,
        feature_names: List[str],
        price_df: pd.DataFrame,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate multiple features

        Args:
            feature_names: List of feature names to calculate
            price_df: Price DataFrame
            **kwargs: Additional arguments passed to feature functions

        Returns:
            Dictionary of feature values

        Raises:
            KeyError: If feature not found
        """
        results = {}

        for name in feature_names:
            func = cls.get(name)
            try:
                results[name] = func(price_df, **kwargs)
            except Exception as e:
                raise RuntimeError(f"Error calculating feature '{name}': {e}")

        return results

    @classmethod
    def list_features(
        cls,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all registered features

        Args:
            category: Optional category filter

        Returns:
            List of feature info dictionaries
        """
        features = []

        for name, info in cls._features.items():
            if category is None or info['category'] == category:
                features.append({
                    'name': name,
                    'category': info['category'],
                    'version': info['version'],
                    'description': info['description']
                })

        return sorted(features, key=lambda x: x['name'])

    @classmethod
    def clear(cls):
        """Clear all registered features (for testing)"""
        cls._features.clear()

    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        """Get detailed info about a feature

        Args:
            name: Feature name

        Returns:
            Feature info dictionary
        """
        if name not in cls._features:
            raise KeyError(f"Feature '{name}' not registered")

        info = cls._features[name].copy()
        info['name'] = name
        del info['function']  # Don't expose function object

        return info
