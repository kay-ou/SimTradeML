# -*- coding: utf-8 -*-
"""
Configuration management system
Supports YAML, dict, and programmatic configuration
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml
import copy


class Config:
    """Configuration manager with nested access support"""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize config

        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict or {}
        self._defaults = self._get_defaults()

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file

        Args:
            yaml_path: Path to YAML config file

        Returns:
            Config instance
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config instance
        """
        return cls(copy.deepcopy(config_dict))

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation

        Args:
            key: Configuration key (supports dot notation like 'data.source')
            default: Default value if key not found

        Returns:
            Configuration value

        Examples:
            >>> config.get('model.type')  # Access nested value
            'xgboost'
            >>> config.get('model.params.max_depth', 6)
            6
        """
        keys = key.split('.')
        value: Any = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    # Try defaults
                    return self._get_from_defaults(key, default)
            else:
                return default

        return value if value is not None else default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        target = self._config

        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        target[keys[-1]] = value

    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with new values

        Args:
            config_dict: Dictionary to merge into config
        """
        self._deep_update(self._config, config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary

        Returns:
            Configuration dictionary
        """
        return copy.deepcopy(self._config)

    def save(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file

        Args:
            yaml_path: Path to save YAML file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

    def _get_from_defaults(self, key: str, fallback: Any) -> Any:
        """Get value from defaults"""
        keys = key.split('.')
        value: Any = self._defaults

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return fallback
            else:
                return fallback

        return value if value is not None else fallback

    @staticmethod
    def _deep_update(target: dict, source: dict) -> None:
        """Deep merge dictionaries"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                Config._deep_update(target[key], value)
            else:
                target[key] = value

    @staticmethod
    def _get_defaults() -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            'data': {
                'source': 'simtradelab',
                'lookback_days': 60,
                'predict_days': 5,
                'sampling_window_days': 15,
            },
            'features': {
                'enable_feature_selection': True,
                'correlation_threshold': 0.85,
                'ic_threshold': 0.0005,
                'p_value_threshold': 0.05,
            },
            'model': {
                'type': 'xgboost',
                'params': {
                    'booster': 'gbtree',
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'max_depth': 4,
                    'learning_rate': 0.04,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'min_child_weight': 6,
                    'gamma': 0.3,
                    'lambda': 2.0,
                    'alpha': 0.2,
                    'seed': 42,
                },
                'num_boost_round': 2000,
                'early_stopping_rounds': 100,
            },
            'training': {
                'train_ratio': 0.70,
                'val_ratio': 0.15,
                'enable_time_series_cv': False,
                'n_splits': 5,
                'parallel_jobs': -1,  # -1 = use all CPUs
            },
            'output': {
                'model_path': 'models/model.pkl',
                'log_level': 'INFO',
            }
        }

    def __repr__(self) -> str:
        return f"Config({self._config})"
