# -*- coding: utf-8 -*-
"""
Unit tests for logger utilities
"""

import pytest
import logging
import tempfile
from pathlib import Path
from simtrademl.core.utils.logger import setup_logger


@pytest.mark.unit
class TestLogger:
    """Test logging utilities"""

    def test_setup_logger_basic(self):
        """Test basic logger setup"""
        logger = setup_logger('test_basic', level='INFO')
        assert logger.name == 'test_basic'
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_setup_logger_debug_level(self):
        """Test logger with DEBUG level"""
        logger = setup_logger('test_debug', level='DEBUG')
        assert logger.level == logging.DEBUG

    def test_setup_logger_with_file(self):
        """Test logger with file handler"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'test.log'
            logger = setup_logger('test_file', level='INFO', log_file=str(log_file))

            # Log a message
            logger.info('Test message')

            # Check file was created and contains message
            assert log_file.exists()
            content = log_file.read_text()
            assert 'Test message' in content

    def test_setup_logger_no_console(self):
        """Test logger without console output"""
        logger = setup_logger('test_no_console', level='INFO', console=False)
        # Should have no console handlers, only file handler if specified
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) == 0

    def test_setup_logger_creates_directory(self):
        """Test logger creates log directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'subdir' / 'test.log'
            logger = setup_logger('test_dir', level='INFO', log_file=str(log_file))
            logger.info('Test')
            assert log_file.exists()

    def test_logger_clears_handlers(self):
        """Test logger clears existing handlers"""
        logger1 = setup_logger('test_clear', level='INFO')
        num_handlers_1 = len(logger1.handlers)

        # Setup again - should clear old handlers
        logger2 = setup_logger('test_clear', level='DEBUG')
        num_handlers_2 = len(logger2.handlers)

        # Should have same number of handlers, not doubled
        assert num_handlers_2 == num_handlers_1
