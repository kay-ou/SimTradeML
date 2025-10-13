"""Configuration settings module using Pydantic Settings.

This module provides centralized configuration management with support for
environment variables and YAML files.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Supported environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LogFormat(str, Enum):
    """Supported log formats."""

    JSON = "json"
    CONSOLE = "console"


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment",
    )

    # SimTradeData API
    simtradedata_api_url: str = Field(
        default="https://api.simtradedata.com",
        description="SimTradeData API base URL",
    )
    simtradedata_api_key: Optional[str] = Field(
        default=None,
        description="SimTradeData API key",
    )

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI",
    )
    mlflow_artifact_store: Optional[str] = Field(
        default=None,
        description="MLflow artifact store (e.g., s3://bucket-name)",
    )
    mlflow_default_experiment: str = Field(
        default="default",
        description="Default MLflow experiment name",
    )

    # Database
    database_url: Optional[str] = Field(
        default=None,
        description="Database connection URL",
    )
    timescaledb_url: Optional[str] = Field(
        default=None,
        description="TimescaleDB connection URL for feature store",
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    # Feature Store
    feature_store_type: str = Field(
        default="timescale",
        description="Feature store backend type (timescale or feast)",
    )
    feature_cache_ttl: int = Field(
        default=300,
        description="Feature cache TTL in seconds for online inference",
        ge=1,
    )

    # AWS Configuration
    aws_region: str = Field(
        default="us-west-2",
        description="AWS region",
    )
    aws_account_id: Optional[str] = Field(
        default=None,
        description="AWS account ID",
    )
    ecr_repository: str = Field(
        default="simtrademl-inference",
        description="ECR repository name",
    )

    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API host",
    )
    api_port: int = Field(
        default=8000,
        description="API port",
        ge=1,
        le=65535,
    )
    api_workers: int = Field(
        default=4,
        description="Number of API workers",
        ge=1,
    )
    api_key_secret: Optional[str] = Field(
        default=None,
        description="Secret key for API authentication",
    )
    api_keys: list[str] = Field(
        default_factory=list,
        description='Valid API keys for authentication. Set API_KEYS env var as JSON array: \'["key1","key2"]\'',
    )

    # Rate Limiting
    rate_limit_requests_per_minute: int = Field(
        default=100,
        description="Maximum requests per minute per API key",
        ge=1,
    )
    rate_limit_burst_size: int = Field(
        default=20,
        description="Maximum burst size for rate limiting",
        ge=1,
    )

    # Logging
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level",
    )
    log_format: LogFormat = Field(
        default=LogFormat.JSON,
        description="Log output format",
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Log file path",
    )

    # Model Configuration
    model_cache_dir: Path = Field(
        default=Path("/tmp/simtrademl/models"),
        description="Directory for model cache",
    )
    model_cache_size: int = Field(
        default=10,
        description="Model LRU cache size (number of models to keep in memory)",
        ge=1,
    )
    model_preload_list: list[str] = Field(
        default_factory=list,
        description='Models to preload on startup. Set MODEL_PRELOAD_LIST env var as JSON array: \'["model1:v1","model2:v2"]\'',
    )
    default_model_version: str = Field(
        default="latest",
        description="Default model version to use",
    )

    # Security
    secret_key: Optional[str] = Field(
        default=None,
        description="Secret key for JWT tokens",
    )
    aws_secrets_manager_arn: Optional[str] = Field(
        default=None,
        description="AWS Secrets Manager ARN",
    )

    # Monitoring
    prometheus_port: int = Field(
        default=9090,
        description="Prometheus metrics port",
        ge=1,
        le=65535,
    )
    grafana_port: int = Field(
        default=3000,
        description="Grafana dashboard port",
        ge=1,
        le=65535,
    )

    # Data Validation
    data_validation_missing_threshold: float = Field(
        default=0.1,
        description="Maximum allowed missing value ratio (0-1)",
        ge=0.0,
        le=1.0,
    )
    data_validation_ks_pvalue: float = Field(
        default=0.05,
        description="KS test p-value threshold for distribution shift detection",
        ge=0.0,
        le=1.0,
    )
    great_expectations_context_root: Optional[Path] = Field(
        default=None,
        description="Great Expectations context root directory",
    )

    # Training Configuration
    training_default_train_ratio: float = Field(
        default=0.7,
        description="Default train/validation/test split ratio for training data",
        ge=0.0,
        le=1.0,
    )
    training_default_val_ratio: float = Field(
        default=0.15,
        description="Default validation split ratio",
        ge=0.0,
        le=1.0,
    )
    training_default_test_ratio: float = Field(
        default=0.15,
        description="Default test split ratio",
        ge=0.0,
        le=1.0,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("log_file", mode="before")
    @classmethod
    def create_log_dir(cls, v: Optional[str]) -> Optional[Path]:
        """Create log directory if it doesn't exist."""
        if v is not None:
            log_path = Path(v)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            return log_path
        return None

    @field_validator("model_cache_dir", mode="before")
    @classmethod
    def create_model_cache_dir(cls, v: str | Path) -> Path:
        """Create model cache directory if it doesn't exist."""
        cache_path = Path(v)
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    @field_validator("model_cache_size")
    @classmethod
    def validate_cache_size(cls, v: int) -> int:
        """Validate cache size is positive."""
        if v < 1:
            raise ValueError("model_cache_size must be >= 1")
        return v

    @field_validator("training_default_test_ratio")
    @classmethod
    def validate_training_ratios(cls, v: float, info) -> float:
        """Validate that train, val, and test ratios sum to 1.0."""
        if info.data.get("training_default_train_ratio") is not None and info.data.get(
            "training_default_val_ratio"
        ) is not None:
            train = info.data["training_default_train_ratio"]
            val = info.data["training_default_val_ratio"]
            total = train + val + v
            if not (0.99 <= total <= 1.01):  # Allow small floating point error
                raise ValueError(
                    f"Training ratios must sum to 1.0, got {total:.3f} "
                    f"(train={train}, val={val}, test={v})"
                )
        return v

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings singleton.

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings (useful for testing).

    Returns:
        New Settings instance
    """
    global _settings
    _settings = Settings()
    return _settings


# Example usage
if __name__ == "__main__":
    settings = get_settings()
    print(f"Environment: {settings.environment}")
    print(f"Log Level: {settings.log_level}")
    print(f"MLflow URI: {settings.mlflow_tracking_uri}")
    print(f"API Port: {settings.api_port}")
    print(f"Model Cache: {settings.model_cache_dir}")
