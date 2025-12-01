# -*- coding: utf-8 -*-
"""
SimTradeML Core Framework
Reusable ML framework for quantitative trading
"""

from .utils.config import Config
from .utils.logger import setup_logger
from .utils.metrics import calculate_ic, calculate_rank_ic, calculate_icir

__all__ = [
    'Config',
    'setup_logger',
    'calculate_ic',
    'calculate_rank_ic',
    'calculate_icir',
]
