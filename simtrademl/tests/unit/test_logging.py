"""Unit tests for logging module."""

import json
import logging
from pathlib import Path

import pytest
import structlog

from simtrademl.utils.logging import (
    add_environment,
    add_service_name,
    add_trace_id,
    configure_logging,
    get_logger,
    get_trace_id,
    set_trace_id,
)


@pytest.fixture(autouse=True)
def reset_logging() -> None:
    """Reset logging configuration before each test."""
    # Clear all handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    # Reset structlog
    structlog.reset_defaults()


class TestTraceId:
    """Tests for trace_id functionality."""

    def test_get_trace_id_generates_unique_id(self) -> None:
        """Test that get_trace_id generates a unique UUID."""
        trace_id = get_trace_id()
        assert trace_id is not None
        assert isinstance(trace_id, str)
        assert len(trace_id) == 36  # UUID format

    def test_set_trace_id(self) -> None:
        """Test setting a custom trace_id."""
        custom_id = "custom-trace-123"
        set_trace_id(custom_id)
        assert get_trace_id() == custom_id

    def test_add_trace_id_processor(self) -> None:
        """Test the add_trace_id processor adds trace_id to event dict."""
        event_dict = {}
        result = add_trace_id(None, "info", event_dict)
        assert "trace_id" in result
        assert isinstance(result["trace_id"], str)


class TestProcessors:
    """Tests for log processors."""

    def test_add_service_name(self) -> None:
        """Test add_service_name processor."""
        event_dict = {}
        result = add_service_name(None, "info", event_dict)
        assert result["service_name"] == "simtrademl"

    def test_add_environment_processor(self) -> None:
        """Test add_environment processor factory."""
        processor = add_environment("production")
        event_dict = {}
        result = processor(None, "info", event_dict)
        assert result["environment"] == "production"


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_default(self) -> None:
        """Test configure_logging with default parameters."""
        configure_logging()
        logger = get_logger(__name__)
        assert logger is not None
        # Logger is a BoundLoggerLazyProxy which is also valid
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")

    def test_configure_logging_debug_level(self) -> None:
        """Test configure_logging with DEBUG level."""
        configure_logging(log_level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_configure_logging_console_format(self) -> None:
        """Test configure_logging with console format."""
        configure_logging(log_format="console", environment="dev")
        logger = get_logger(__name__)
        # Logger should be configured without errors
        logger.info("Test message")

    def test_configure_logging_json_format(self) -> None:
        """Test configure_logging with JSON format."""
        configure_logging(log_format="json", environment="production")
        logger = get_logger(__name__)
        logger.info("Test message")

    def test_configure_logging_auto_format_dev(self) -> None:
        """Test that dev environment auto-selects console format."""
        configure_logging(environment="dev")
        # Should not raise an error
        logger = get_logger(__name__)
        logger.info("Test message")

    def test_configure_logging_with_file(self, tmp_path: Path) -> None:
        """Test configure_logging with file output."""
        log_file = tmp_path / "test.log"
        configure_logging(log_file=str(log_file), environment="production")

        logger = get_logger(__name__)
        logger.info("Test message to file")

        # Check that file was created
        assert log_file.exists()

        # Check file content
        content = log_file.read_text()
        assert len(content) > 0

    def test_configure_logging_creates_log_directory(self, tmp_path: Path) -> None:
        """Test that configure_logging creates log directory if needed."""
        log_file = tmp_path / "logs" / "nested" / "test.log"
        configure_logging(log_file=str(log_file))

        logger = get_logger(__name__)
        logger.info("Test message")

        assert log_file.exists()
        assert log_file.parent.exists()


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_bound_logger(self) -> None:
        """Test that get_logger returns a BoundLogger."""
        configure_logging()
        logger = get_logger(__name__)
        # Check that logger has the expected methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert hasattr(logger, "exception")

    def test_get_logger_with_custom_name(self) -> None:
        """Test get_logger with custom name."""
        configure_logging()
        logger = get_logger("custom.logger.name")
        assert logger is not None


class TestLogOutput:
    """Tests for actual log output."""

    def test_log_levels(self, tmp_path: Path) -> None:
        """Test that different log levels work correctly."""
        log_file = tmp_path / "test_levels.log"
        configure_logging(log_level="DEBUG", log_format="json", log_file=str(log_file))
        logger = get_logger(__name__)

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Check log file for all messages
        content = log_file.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 4

        # Verify each log level
        log_entries = [json.loads(line) for line in lines]
        levels = [entry["level"] for entry in log_entries]
        assert "debug" in levels
        assert "info" in levels
        assert "warning" in levels
        assert "error" in levels

    def test_log_with_context(self) -> None:
        """Test logging with additional context fields."""
        configure_logging(log_format="json", environment="production")
        logger = get_logger(__name__)

        # Should not raise an error
        logger.info("Test with context", user_id=123, action="test", metric=0.95)

    def test_log_exception(self) -> None:
        """Test exception logging."""
        configure_logging()
        logger = get_logger(__name__)

        try:
            raise ValueError("Test exception")
        except Exception:
            # Should not raise an error
            logger.exception("Exception occurred")

    def test_trace_id_in_logs(self, tmp_path: Path) -> None:
        """Test that trace_id appears in log output."""
        log_file = tmp_path / "test.log"
        configure_logging(
            log_format="json", log_file=str(log_file), environment="production"
        )

        custom_trace = "test-trace-456"
        set_trace_id(custom_trace)

        logger = get_logger(__name__)
        logger.info("Test message with trace_id")

        # Read log file and check for trace_id
        content = log_file.read_text()
        log_entry = json.loads(content.strip().split("\n")[-1])
        assert log_entry["trace_id"] == custom_trace

    def test_environment_in_logs(self, tmp_path: Path) -> None:
        """Test that environment appears in log output."""
        log_file = tmp_path / "test.log"
        configure_logging(
            log_format="json", log_file=str(log_file), environment="staging"
        )

        logger = get_logger(__name__)
        logger.info("Test message with environment")

        # Read log file and check for environment
        content = log_file.read_text()
        log_entry = json.loads(content.strip().split("\n")[-1])
        assert log_entry["environment"] == "staging"

    def test_service_name_in_logs(self, tmp_path: Path) -> None:
        """Test that service_name appears in log output."""
        log_file = tmp_path / "test.log"
        configure_logging(log_format="json", log_file=str(log_file), environment="production")

        logger = get_logger(__name__)
        logger.info("Test message with service_name")

        # Read log file and check for service_name
        content = log_file.read_text()
        log_entry = json.loads(content.strip().split("\n")[-1])
        assert log_entry["service_name"] == "simtrademl"


@pytest.mark.unit
class TestLoggingIntegration:
    """Integration tests for logging system."""

    def test_complete_logging_workflow(self, tmp_path: Path) -> None:
        """Test complete logging workflow from configuration to output."""
        log_file = tmp_path / "integration.log"

        # Configure logging
        configure_logging(
            log_level="INFO",
            log_format="json",
            log_file=str(log_file),
            environment="production",
        )

        # Set trace_id
        trace_id = "integration-test-123"
        set_trace_id(trace_id)

        # Get logger and log messages
        logger = get_logger("integration.test")
        logger.info("Integration test started", test_id=1)
        logger.warning("Test warning", code="W001")
        logger.error("Test error", code="E001", details="Test details")

        # Verify log file
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 3

        # Verify each log entry
        for line in lines:
            entry = json.loads(line)
            assert entry["service_name"] == "simtrademl"
            assert entry["environment"] == "production"
            assert entry["trace_id"] == trace_id
            assert "timestamp" in entry
            assert "level" in entry
            assert "event" in entry
