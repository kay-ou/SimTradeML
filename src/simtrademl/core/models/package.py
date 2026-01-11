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
        """Load package from single file"""
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

    def predict(self, features_dict: Dict[str, float]) -> float:
        """Make prediction with feature validation"""
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

    def summary(self) -> str:
        """Get package summary"""
        if self.metadata:
            return self.metadata.summary()
        return "No metadata available"
