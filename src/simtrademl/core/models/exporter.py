# -*- coding: utf-8 -*-
"""
PTrade Model Exporter

Provides unified interface to export trained models as PTrade-compatible packages.
"""

import os
import pickle
import shutil
from typing import Any, Dict, Optional, List
from pathlib import Path

from .metadata import ModelMetadata


class PTradeModelExporter:
    """Export trained models as PTrade-compatible packages

    This class handles:
    - Model saving in multiple formats (JSON/Pickle/Model)
    - Scaler serialization
    - Metadata generation
    - Feature list export
    - Usage documentation generation

    Example:
        >>> exporter = PTradeModelExporter(output_dir='models/my_model')
        >>> exporter.export(
        ...     model=xgb_model,
        ...     scaler=robust_scaler,
        ...     metadata=model_metadata
        ... )
    """

    def __init__(self, output_dir: str):
        """Initialize exporter

        Args:
            output_dir: Directory to save model package
        """
        self.output_dir = Path(output_dir)

    def export(
        self,
        model: Any,
        metadata: ModelMetadata,
        scaler: Optional[Any] = None,
        model_format: str = 'json',
        overwrite: bool = False
    ) -> str:
        """Export complete model package

        Args:
            model: Trained model (XGBoost Booster)
            metadata: ModelMetadata instance
            scaler: Optional feature scaler
            model_format: Format to save model ('json', 'model', 'pickle')
            overwrite: Whether to overwrite existing directory

        Returns:
            Path to exported model package

        Raises:
            ValueError: If model_format is invalid
            FileExistsError: If output directory exists and overwrite=False
        """
        # Validate format
        valid_formats = ['json', 'model', 'pickle']
        if model_format not in valid_formats:
            raise ValueError(
                f"Invalid model_format '{model_format}'. "
                f"Must be one of: {valid_formats}"
            )

        # Create output directory
        if self.output_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Directory {self.output_dir} already exists. "
                    f"Use overwrite=True to replace."
                )
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_filename = self._save_model(model, model_format)
        metadata.add_file('model', model_filename)

        # Save scaler if provided
        if scaler is not None:
            scaler_filename = self._save_scaler(scaler)
            metadata.add_file('scaler', scaler_filename)

        # Save feature list
        features_filename = self._save_features(metadata.features)
        metadata.add_file('features', features_filename)

        # Save metadata
        metadata_filename = 'metadata.json'
        metadata.save(str(self.output_dir / metadata_filename))

        # Generate README
        self._generate_readme(metadata, model_format)

        # Generate usage example
        self._generate_usage_example(metadata, scaler is not None)

        return str(self.output_dir)

    def _save_model(self, model: Any, format: str) -> str:
        """Save model in specified format"""
        if format == 'json':
            filename = 'model.json'
            model.save_model(str(self.output_dir / filename))
        elif format == 'model':
            filename = 'model.model'
            model.save_model(str(self.output_dir / filename))
        elif format == 'pickle':
            filename = 'model.pkl'
            with open(self.output_dir / filename, 'wb') as f:
                pickle.dump(model, f)

        return filename

    def _save_scaler(self, scaler: Any) -> str:
        """Save feature scaler"""
        filename = 'scaler.pkl'
        with open(self.output_dir / filename, 'wb') as f:
            pickle.dump(scaler, f)

        return filename

    def _save_features(self, features: List[str]) -> str:
        """Save feature list"""
        import json

        filename = 'features.json'
        with open(self.output_dir / filename, 'w') as f:
            json.dump(features, f, indent=2)

        return filename

    def _generate_readme(self, metadata: ModelMetadata, model_format: str):
        """Generate README.md"""
        readme_content = f"""# Model Package: {metadata.model_id}

## Model Information

- **Model ID**: {metadata.model_id}
- **Version**: {metadata.version}
- **Created**: {metadata.created_at}
- **Model Type**: {metadata.model_type}

## Features ({metadata.n_features})

⚠️ **CRITICAL**: Feature order must match exactly!

```
{chr(10).join(f'{i+1:2d}. {name}' for i, name in enumerate(metadata.features))}
```

## Files

- `{metadata.files.get('model')}` - Model ({model_format})
- `{metadata.files.get('scaler', 'N/A')}` - Scaler
- `{metadata.files.get('features')}` - Features
- `metadata.json` - Metadata
- `usage_example.py` - Example

## Performance

{self._format_metrics(metadata.metrics) if metadata.metrics else 'No metrics recorded'}

---
Generated by SimTradeML
"""

        with open(self.output_dir / 'README.md', 'w') as f:
            f.write(readme_content)

    def _generate_usage_example(self, metadata: ModelMetadata, has_scaler: bool):
        """Generate usage example"""
        example = f"""#!/usr/bin/env python
\"\"\"Usage example for {metadata.model_id}\"\"\"

import numpy as np
import xgboost as xgb
import pickle
import json

# Load model
model = xgb.Booster(model_file='{metadata.files.get('model')}')

# Load scaler
{"with open('" + metadata.files.get('scaler', 'scaler.pkl') + "', 'rb') as f:" if has_scaler else "# No scaler"}
{"    scaler = pickle.load(f)" if has_scaler else ""}

# Load features
with open('{metadata.files.get('features')}', 'r') as f:
    feature_names = json.load(f)

# Predict
features_dict = {{{', '.join(f"'{name}': 0.0" for name in metadata.features[:3])}, ...}}
X = np.array([features_dict[n] for n in feature_names]).reshape(1, -1)
{"X_scaled = scaler.transform(X)" if has_scaler else "X_scaled = X"}
pred = model.predict(xgb.DMatrix(X_scaled))[0]
print(f'Prediction: {{pred:.6f}}')
"""

        with open(self.output_dir / 'usage_example.py', 'w') as f:
            f.write(example)

        os.chmod(self.output_dir / 'usage_example.py', 0o755)

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics"""
        return '\n'.join(f"- **{k.upper()}**: {v:.4f}" for k, v in metrics.items())
