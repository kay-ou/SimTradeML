# -*- coding: utf-8 -*-
"""
SimTradeML Complete Example

Demonstrates the complete workflow:
1. Training a model
2. Saving as single-file package
3. Loading and making predictions

See mvp_train.py for full training pipeline.
"""

import xgboost as xgb
from datetime import datetime

from simtrademl.core.utils.logger import setup_logger
from simtrademl.core.models import (
    ModelMetadata,
    create_model_id,
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
        'ma5', 'ma10', 'ma20', 'ma60',
        'return_1d', 'return_5d', 'return_10d', 'return_20d',
        'volatility_20d', 'volume_ratio', 'price_position'
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
# PART 2: SAVE MODEL
# ============================================================================

def save_example():
    """Save model as single-file package"""
    logger.info("\n" + "="*60)
    logger.info("PART 2: SAVE MODEL")
    logger.info("="*60)

    # Assume we have: model, scaler, metadata
    # package = PTradeModelPackage(model=model, scaler=scaler, metadata=metadata)
    # package.save('model.ptp')

    logger.info("Save to single file:")
    logger.info("  package = PTradeModelPackage(model, scaler, metadata)")
    logger.info("  package.save('model.ptp')")
    logger.info("\nONE file contains everything:")
    logger.info("  - XGBoost model")
    logger.info("  - Feature scaler")
    logger.info("  - Complete metadata (features, version, etc.)")


# ============================================================================
# PART 3: LOAD AND PREDICT
# ============================================================================

def predict_example():
    """Load and predict using single file"""
    logger.info("\n" + "="*60)
    logger.info("PART 3: LOAD AND PREDICT")
    logger.info("="*60)

    logger.info("Load from ONE file:")
    logger.info("  package = PTradeModelPackage.load('model.ptp')")

    logger.info("\nSingle prediction:")
    logger.info("  prediction = package.predict(features_dict)")

    logger.info("\nBatch prediction:")
    logger.info("  predictions = package.predict_batch(features_list)")

    logger.info("\n✓ Features are automatically:")
    logger.info("  - Validated (order and completeness)")
    logger.info("  - Scaled (using saved scaler)")
    logger.info("  - Fed to model in correct order")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete example"""
    print("\n" + "="*60)
    print("SimTradeML Usage Example")
    print("="*60)
    print("\nFor full training code, see: mvp_train.py")
    print("="*60)

    # Part 1: Training
    metadata = train_example()

    # Part 2: Save
    save_example()

    # Part 3: Load & Predict
    predict_example()

    print("\n" + "="*60)
    print("✓ Example completed!")
    print("="*60)
    print("\n📌 Quick Reference:")
    print("   Save:    package.save('model.ptp')")
    print("   Load:    package = PTradeModelPackage.load('model.ptp')")
    print("   Predict: prediction = package.predict(features_dict)")
    print("   Batch:   predictions = package.predict_batch(features_list)")
    print("="*60)


if __name__ == '__main__':
    main()
