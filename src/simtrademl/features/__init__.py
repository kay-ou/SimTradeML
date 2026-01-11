# -*- coding: utf-8 -*-
"""
Feature Management System

Provides centralized feature registration and calculation.
"""

from .registry import FeatureRegistry

# Import technical features to auto-register them
from . import technical  # noqa: F401

__all__ = ['FeatureRegistry']
