# -*- coding: utf-8 -*-
"""
SimTradeML - Reusable ML Framework for Quantitative Trading
"""

__version__ = '0.2.0'

# Core modules
from .core.utils.config import Config
from .core.utils.logger import setup_logger
from .core.utils.metrics import (
    calculate_ic,
    calculate_rank_ic,
    calculate_icir,
    calculate_quantile_returns,
    calculate_direction_accuracy,
)

# Data sources
from .data_sources.simtradelab_source import SimTradeLabDataSource

__all__ = [
    # Version
    '__version__',

    # Core utilities
    'Config',
    'setup_logger',

    # Metrics
    'calculate_ic',
    'calculate_rank_ic',
    'calculate_icir',
    'calculate_quantile_returns',
    'calculate_direction_accuracy',

    # Data sources
    'SimTradeLabDataSource',
]
