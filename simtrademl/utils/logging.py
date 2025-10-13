"""Structured logging configuration using structlog.

This module provides centralized logging configuration with JSON formatting
and trace_id support for distributed tracing.
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, Processor

# Context variable for trace_id (for distributed tracing)
trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)


def get_trace_id() -> str:
    """Get or create trace_id for current context."""
    trace_id = trace_id_var.get()
    if trace_id is None:
        trace_id = str(uuid.uuid4())
        trace_id_var.set(trace_id)
    return trace_id


def set_trace_id(trace_id: str) -> None:
    """Set trace_id for current context."""
    trace_id_var.set(trace_id)


def add_trace_id(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add trace_id to log event."""
    event_dict["trace_id"] = get_trace_id()
    return event_dict


def add_service_name(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add service_name to log event."""
    event_dict["service_name"] = "simtrademl"
    return event_dict


def add_environment(
    environment: str,
) -> Processor:
    """Create a processor that adds environment to log events.

    Args:
        environment: Environment name (dev, staging, production)

    Returns:
        Processor function
    """

    def _add_environment(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
        event_dict["environment"] = environment
        return event_dict

    return _add_environment


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    environment: str = "production",
) -> None:
    """Configure structured logging with structlog.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Output format ('json' or 'console')
        log_file: Optional log file path for file output
        environment: Environment name (dev, staging, production)
    """
    # Convert log_level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Auto-detect format based on environment if not explicitly set
    if environment == "dev" and log_format == "json":
        log_format = "console"
    elif environment in ("staging", "production") and log_format == "console":
        log_format = "json"

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        stream=sys.stdout,
        force=True,  # Force reconfiguration
    )

    # Build processor chain
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_trace_id,
        add_service_name,
        add_environment(environment),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
    ]

    # Add exception info processor
    if log_format == "json":
        processors.append(structlog.processors.format_exc_info)
    else:
        processors.append(structlog.dev.set_exc_info)

    # Add renderer based on format
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)

        # For file logging, always use JSON format
        file_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(file_formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# Example usage and testing
if __name__ == "__main__":
    print("=== Console Format (Development) ===")
    # Configure logging for development
    configure_logging(log_level="DEBUG", log_format="console", environment="dev")

    # Get a logger
    logger = get_logger(__name__)

    # Test different log levels
    logger.debug("Debug message", extra_data="some_value")
    logger.info("Info message", user_id=123, action="login")
    logger.warning("Warning message", threshold=0.8, current=0.9)
    logger.error("Error message", error_code="E001", details="Something went wrong")

    # Test trace_id
    set_trace_id("test-trace-123")
    logger.info("Message with custom trace_id", request_id="req-456")

    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.exception("Exception occurred", error_type=type(e).__name__)

    # Test JSON format
    print("\n=== JSON Format (Production) ===")
    configure_logging(log_level="INFO", log_format="json", environment="production")
    logger = get_logger(__name__)
    logger.info("JSON formatted message", metric="accuracy", value=0.95)

    # Test with file output
    print("\n=== File Logging ===")
    configure_logging(
        log_level="INFO",
        log_format="json",
        log_file="/tmp/simtrademl.log",
        environment="production",
    )
    logger = get_logger(__name__)
    logger.info("Message logged to file", component="test", status="success")
    print("Check /tmp/simtrademl.log for file output")
