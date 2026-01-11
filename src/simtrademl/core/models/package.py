# -*- coding: utf-8 -*-
"""
PTrade Single-File Model Package

Provides a unified single-file format for PTrade-compatible models.
All components (model, scaler, metadata) are bundled into one file.
"""

import pickle
import json
from typing import Any, Optional, Dict
from pathlib import Path
import xgboost as xgb
import numpy as np

from .metadata import ModelMetadata


class PTradeModelPackage:
    """Single-file model package for PTrade

    Bundles model + scaler + metadata into one file.

    Example:
        # Save
        package = PTradeModelPackage(model, scaler, metadata)
        package.save('my_model.ptp')

        # Load and predict
        package = PTradeModelPackage.load('my_model.ptp')
        prediction = package.predict(features_dict)
    """

    def __init__(
        self,
        model: Optional[xgb.Booster] = None,
        scaler: Optional[Any] = None,
        metadata: Optional[ModelMetadata] = None
    ):
        self.model = model
        self.scaler = scaler
        self.metadata = metadata

    def save(self, filepath: str):
        """Save complete package to single pickle file"""
        # Save XGBoost model to bytes
        model_bytes = None
        if self.model is not None:
            temp_path = Path(filepath).with_suffix('.tmp.json')
            self.model.save_model(str(temp_path))
            model_bytes = temp_path.read_bytes()
            temp_path.unlink()

        package_data = {
            'model_bytes': model_bytes,
            'scaler': self.scaler,
            'metadata': self.metadata.to_dict() if self.metadata else None,
            'format_version': '1.0'
        }

        with open(filepath, 'wb') as f:
            pickle.dump(package_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str) -> 'PTradeModelPackage':
        """Load package from single .ptp file

        Args:
            filepath: Path to .ptp package file

        Returns:
            Loaded PTradeModelPackage

        Raises:
            FileNotFoundError: If file not found
            ValueError: If file format invalid
        """
        with open(filepath, 'rb') as f:
            package_data = pickle.load(f)

        # Restore XGBoost model
        model = None
        if package_data.get('model_bytes'):
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
                temp_path = f.name
                f.write(package_data['model_bytes'])

            model = xgb.Booster(model_file=temp_path)
            Path(temp_path).unlink()

        # Restore metadata
        metadata = None
        if package_data.get('metadata'):
            metadata = ModelMetadata.from_dict(package_data['metadata'])

        return cls(
            model=model,
            scaler=package_data.get('scaler'),
            metadata=metadata
        )

    @classmethod
    def load_from_files(
        cls,
        model_path: str,
        metadata_path: Optional[str] = None,
        scaler_path: Optional[str] = None
    ) -> 'PTradeModelPackage':
        """Load package from separate files (for backward compatibility)

        Supports multiple model formats: .json, .model, .pkl

        Args:
            model_path: Path to model file (.json, .model, or .pkl)
            metadata_path: Optional path to metadata.json
            scaler_path: Optional path to scaler.pkl

        Returns:
            Loaded PTradeModelPackage

        Raises:
            FileNotFoundError: If model file not found
            ValueError: If model format not supported

        Example:
            # Load from JSON model
            package = PTradeModelPackage.load_from_files(
                'model.json',
                'metadata.json',
                'scaler.pkl'
            )

            # Load from pickle model
            package = PTradeModelPackage.load_from_files('model.pkl')
        """
        model_file = Path(model_path)

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Load model based on extension
        model = None
        suffix = model_file.suffix.lower()

        if suffix in ['.json', '.model']:
            # XGBoost native format
            model = xgb.Booster(model_file=str(model_file))
        elif suffix == '.pkl':
            # Pickle format
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(
                f"Unsupported model format: {suffix}. "
                f"Supported: .json, .model, .pkl"
            )

        # Load metadata if provided
        metadata = None
        if metadata_path:
            metadata_file = Path(metadata_path)
            if metadata_file.exists():
                json_str = metadata_file.read_text()
                metadata = ModelMetadata.from_json(json_str)

        # Load scaler if provided
        scaler = None
        if scaler_path:
            scaler_file = Path(scaler_path)
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    scaler = pickle.load(f)

        return cls(model=model, scaler=scaler, metadata=metadata)

    def predict(self, features_dict: Dict[str, float]) -> float:
        """Make prediction with feature validation

        Args:
            features_dict: Dictionary mapping feature names to values

        Returns:
            Predicted value

        Raises:
            ValueError: If model or metadata not loaded, or features invalid
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        if self.metadata is None:
            raise ValueError("Metadata not loaded")

        # Validate features
        self.metadata.validate_features(list(features_dict.keys()))

        # Construct feature vector in correct order
        X = np.array([features_dict[name] for name in self.metadata.features]).reshape(1, -1)

        # Apply scaler if available
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Predict
        dmatrix = xgb.DMatrix(X)
        return float(self.model.predict(dmatrix)[0])

    def predict_batch(self, features_list: list[Dict[str, float]]) -> np.ndarray:
        """Make batch predictions with feature validation

        Args:
            features_list: List of feature dictionaries

        Returns:
            Array of predictions

        Raises:
            ValueError: If model or metadata not loaded, or features invalid

        Example:
            features_list = [
                {'ma5': 100.0, 'ma10': 99.5, 'rsi14': 60.0},
                {'ma5': 101.0, 'ma10': 100.0, 'rsi14': 55.0}
            ]
            predictions = package.predict_batch(features_list)
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        if self.metadata is None:
            raise ValueError("Metadata not loaded")

        if not features_list:
            return np.array([])

        # Validate first sample (assume all have same features)
        self.metadata.validate_features(list(features_list[0].keys()))

        # Construct feature matrix in correct order
        X = np.array([
            [features[name] for name in self.metadata.features]
            for features in features_list
        ])

        # Apply scaler if available
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Predict
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)

    def summary(self) -> str:
        """Get package summary"""
        if self.metadata:
            return self.metadata.summary()
        return "No metadata available"
