# -*- coding: utf-8 -*-
"""
Model Metadata System for PTrade Compatibility

This module provides a robust metadata system to ensure:
- Feature order consistency between training and inference
- Model traceability and versioning
- PTrade compatibility validation
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


@dataclass
class ModelMetadata:
    """Metadata for PTrade-compatible models

    This class ensures all necessary information is captured during training
    and available during inference to guarantee correct predictions.

    Attributes:
        model_id: Unique identifier for the model
        version: Model version string
        created_at: ISO format timestamp of model creation

        # Model information
        model_type: Type of model ('xgboost', 'lightgbm', etc.)
        model_library_version: Version of the ML library used

        # Feature information (CRITICAL for PTrade compatibility)
        features: Ordered list of feature names
        n_features: Number of features
        feature_version: Version identifier for feature definitions

        # Training information
        scaler_type: Type of feature scaler used (if any)
        train_start_date: Start date of training data
        train_end_date: End date of training data
        n_samples: Total number of training samples

        # Hyperparameters
        hyperparameters: Dictionary of model hyperparameters

        # Performance metrics
        metrics: Dictionary of evaluation metrics (IC, ICIR, etc.)

        # File references
        files: Dictionary mapping file types to filenames

        # Additional metadata
        description: Optional description of the model
        tags: Optional tags for categorization
    """

    # Basic information
    model_id: str
    version: str
    created_at: str

    # Model information
    model_type: str
    model_library_version: str

    # Feature information (CRITICAL!)
    features: List[str]
    n_features: int

    # Training information
    scaler_type: Optional[str] = None
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    n_samples: Optional[int] = None

    # Optional advanced information
    feature_version: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate metadata after initialization"""
        # Validate features
        if not self.features:
            raise ValueError("Feature list cannot be empty")

        if self.n_features != len(self.features):
            raise ValueError(
                f"n_features ({self.n_features}) does not match "
                f"length of features list ({len(self.features)})"
            )

        # Check for duplicate features
        if len(self.features) != len(set(self.features)):
            duplicates = [f for f in self.features if self.features.count(f) > 1]
            raise ValueError(f"Duplicate features found: {set(duplicates)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary

        Returns:
            Dictionary representation of metadata
        """
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert metadata to JSON string

        Args:
            indent: Number of spaces for JSON indentation

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str, indent: int = 2):
        """Save metadata to JSON file

        Args:
            filepath: Path to save the JSON file
            indent: Number of spaces for JSON indentation
        """
        with open(filepath, 'w') as f:
            f.write(self.to_json(indent=indent))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary

        Args:
            data: Dictionary containing metadata

        Returns:
            ModelMetadata instance
        """
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'ModelMetadata':
        """Create metadata from JSON string

        Args:
            json_str: JSON string representation

        Returns:
            ModelMetadata instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load(cls, filepath: str) -> 'ModelMetadata':
        """Load metadata from JSON file

        Args:
            filepath: Path to the JSON file

        Returns:
            ModelMetadata instance
        """
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())

    def validate_features(self, input_features: List[str]) -> bool:
        """Validate that input features match expected features

        Args:
            input_features: List of feature names from input data

        Returns:
            True if features match exactly

        Raises:
            ValueError: If features don't match
        """
        if input_features != self.features:
            missing = set(self.features) - set(input_features)
            extra = set(input_features) - set(self.features)

            error_msg = "Feature mismatch detected!\n"
            if missing:
                error_msg += f"  Missing features: {sorted(missing)}\n"
            if extra:
                error_msg += f"  Extra features: {sorted(extra)}\n"
            error_msg += f"  Expected order: {self.features}\n"
            error_msg += f"  Received order: {input_features}"

            raise ValueError(error_msg)

        return True

    def add_metric(self, name: str, value: float):
        """Add or update a performance metric

        Args:
            name: Metric name (e.g., 'ic', 'icir')
            value: Metric value
        """
        self.metrics[name] = value

    def add_file(self, file_type: str, filename: str):
        """Add a file reference

        Args:
            file_type: Type of file (e.g., 'model', 'scaler', 'features')
            filename: Filename or path
        """
        self.files[file_type] = filename

    def summary(self) -> str:
        """Get a human-readable summary of the metadata

        Returns:
            Formatted summary string
        """
        lines = [
            "="*60,
            "Model Metadata Summary",
            "="*60,
            f"Model ID: {self.model_id}",
            f"Version: {self.version}",
            f"Created: {self.created_at}",
            f"Type: {self.model_type} (v{self.model_library_version})",
            "",
            "Features:",
            f"  Count: {self.n_features}",
            f"  Order: {', '.join(self.features[:5])}{'...' if self.n_features > 5 else ''}",
        ]

        if self.scaler_type:
            lines.append(f"  Scaler: {self.scaler_type}")

        if self.train_start_date and self.train_end_date:
            lines.extend([
                "",
                "Training Data:",
                f"  Period: {self.train_start_date} to {self.train_end_date}",
            ])

        if self.n_samples:
            lines.append(f"  Samples: {self.n_samples:,}")

        if self.metrics:
            lines.extend(["", "Performance Metrics:"])
            for name, value in self.metrics.items():
                lines.append(f"  {name}: {value:.4f}")

        if self.files:
            lines.extend(["", "Files:"])
            for file_type, filename in self.files.items():
                lines.append(f"  {file_type}: {filename}")

        lines.append("="*60)

        return "\n".join(lines)


def create_model_id(prefix: str = "model") -> str:
    """Generate a unique model ID

    Args:
        prefix: Prefix for the model ID

    Returns:
        Unique model ID in format: prefix_YYYYMMDD_HHMMSS
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"
