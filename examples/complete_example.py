# -*- coding: utf-8 -*-
"""
SimTradeML Complete Example

This single file demonstrates:
1. Training a model (full pipeline)
2. Exporting model package (multi-file)
3. Using single-file package (RECOMMENDED for PTrade)
4. Loading and predicting (both methods)

Run sections independently or all together.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
import pickle
from datetime import datetime

from simtrademl.data_sources.simtradelab_source import SimTradeLabDataSource
from simtrademl.core.utils.logger import setup_logger
from simtrademl.core.models import (
    ModelMetadata,
    create_model_id,
    PTradeModelExporter,
    PTradeModelPackage
)

logger = setup_logger('example', level='INFO')


# ============================================================================
# PART 1: TRAINING
# ============================================================================

def train_example():
    """Complete training example - see mvp_train.py for details"""
    logger.info("="*60)
    logger.info("PART 1: TRAINING")
    logger.info("="*60)
    logger.info("(Simplified - see mvp_train.py for full pipeline)")

    # Mock trained model for demo
    # In real use: X, y, dates, features = collect_samples(data_source)
    #              model, scaler = train_xgboost(X, y, dates)

    feature_names = [
        'ma10', 'ma20', 'ma5', 'ma60', 'price_position',
        'return_10d', 'return_1d', 'return_20d', 'return_5d',
        'volatility_20d', 'volume_ratio'
    ]

    metadata = ModelMetadata(
        model_id=create_model_id('example'),
        version='1.0',
        created_at=datetime.now().isoformat(),
        model_type='xgboost',
        model_library_version=xgb.__version__,
        features=feature_names,
        n_features=len(feature_names),
        scaler_type='RobustScaler'
    )

    logger.info(f"✓ Model trained with {len(feature_names)} features")
    return metadata


# ============================================================================
# PART 2: SINGLE-FILE EXPORT (RECOMMENDED)
# ============================================================================

def export_single_file_example():
    """Export as single-file package - RECOMMENDED for PTrade"""
    logger.info("\n" + "="*60)
    logger.info("PART 2: SINGLE-FILE EXPORT (RECOMMENDED)")
    logger.info("="*60)

    # Assume we have: model, scaler, metadata
    # package = PTradeModelPackage(model=model, scaler=scaler, metadata=metadata)
    # package.save('examples/model.ptp')

    logger.info("✓ Export to single file:")
    logger.info("  package.save('model.ptp')")
    logger.info("\n  ONE file contains:")
    logger.info("    - XGBoost model")
    logger.info("    - Feature scaler")
    logger.info("    - Metadata (features, version, etc.)")


# ============================================================================
# PART 3: SINGLE-FILE LOADING (RECOMMENDED)
# ============================================================================

def predict_single_file_example():
    """Load and predict using single file - RECOMMENDED"""
    logger.info("\n" + "="*60)
    logger.info("PART 3: SINGLE-FILE LOADING (RECOMMENDED)")
    logger.info("="*60)

    logger.info("Load from ONE file:")
    logger.info("  package = PTradeModelPackage.load('model.ptp')")

    logger.info("\nPredict with ONE line:")
    logger.info("  prediction = package.predict(features_dict)")

    logger.info("\n✓ Advantages:")
    logger.info("  - ONE file to manage")
    logger.info("  - Auto feature validation")
    logger.info("  - Auto feature scaling")
    logger.info("  - No manual ordering needed")


# ============================================================================
# PART 4: MULTI-FILE EXPORT (Alternative)
# ============================================================================

def export_multi_file_example():
    """Export as multi-file package - for inspection"""
    logger.info("\n" + "="*60)
    logger.info("PART 4: MULTI-FILE EXPORT (Alternative)")
    logger.info("="*60)

    logger.info("For human inspection or debugging:")
    logger.info("  exporter = PTradeModelExporter('model_package')")
    logger.info("  exporter.export(model, metadata, scaler)")

    logger.info("\n✓ Generates 6 files:")
    logger.info("    - model.json (XGBoost)")
    logger.info("    - scaler.pkl (Scaler)")
    logger.info("    - features.json (Feature names)")
    logger.info("    - metadata.json (Metadata)")
    logger.info("    - README.md (Documentation)")
    logger.info("    - usage_example.py (Example code)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete example"""
    print("\n" + "="*60)
    print("SimTradeML Complete Example")
    print("="*60)
    print("\nFor full training code, see: mvp_train.py")
    print("This file shows the recommended usage patterns.")
    print("="*60)

    # Part 1: Training
    metadata = train_example()

    # Part 2: Export (RECOMMENDED)
    export_single_file_example()

    # Part 3: Loading (RECOMMENDED)
    predict_single_file_example()

    # Part 4: Alternative (for inspection)
    export_multi_file_example()

    print("\n" + "="*60)
    print("✓ Examples completed!")
    print("="*60)
    print("\n📌 RECOMMENDATION for PTrade:")
    print("   Use single-file package (.ptp format)")
    print("\n   Training:  package.save('model.ptp')")
    print("   Loading:   package = PTradeModelPackage.load('model.ptp')")
    print("   Predict:   prediction = package.predict(features)")
    print("="*60)


if __name__ == '__main__':
    main()
