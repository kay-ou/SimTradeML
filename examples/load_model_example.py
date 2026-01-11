# -*- coding: utf-8 -*-
"""
Example: Load and use trained model (PTrade compatible)

This demonstrates how to load a model trained by mvp_train.py
and use it for prediction with correct feature order and scaling.
"""

import sys
sys.path.insert(0, '/home/kay/dev/SimTradeML/src')

import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb


def load_model_package(model_dir='examples'):
    """Load complete model package

    Args:
        model_dir: Directory containing model files

    Returns:
        (model, scaler, feature_names, metadata)
    """
    # Load metadata
    with open(f'{model_dir}/mvp_metadata.json', 'r') as f:
        metadata = json.load(f)

    print("Model Metadata:")
    print(f"  Model ID: {metadata['model_id']}")
    print(f"  Created: {metadata['created_at']}")
    print(f"  XGBoost version: {metadata['xgboost_version']}")
    print(f"  Number of features: {metadata['n_features']}")
    print(f"  Scaler: {metadata['scaler_type']}")

    # Load model
    model = xgb.Booster(model_file=f"{model_dir}/{metadata['files']['model']}")
    print(f"\n✓ Model loaded from {metadata['files']['model']}")

    # Load scaler
    with open(f"{model_dir}/{metadata['files']['scaler']}", 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler loaded from {metadata['files']['scaler']}")

    # Load feature names
    with open(f"{model_dir}/{metadata['files']['features']}", 'r') as f:
        feature_names = json.load(f)
    print(f"✓ Feature names loaded from {metadata['files']['features']}")

    return model, scaler, feature_names, metadata


def predict_single(model, scaler, feature_names, features_dict):
    """Predict with single sample

    Args:
        model: Trained XGBoost model
        scaler: Fitted scaler
        feature_names: List of feature names in correct order
        features_dict: Dictionary of features

    Returns:
        prediction value
    """
    # Validate features
    missing_features = set(feature_names) - set(features_dict.keys())
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    extra_features = set(features_dict.keys()) - set(feature_names)
    if extra_features:
        print(f"Warning: Extra features will be ignored: {extra_features}")

    # Construct feature vector in correct order
    X = np.array([features_dict[name] for name in feature_names]).reshape(1, -1)

    # Apply scaler
    X_scaled = scaler.transform(X)

    # Predict
    dmatrix = xgb.DMatrix(X_scaled)
    prediction = model.predict(dmatrix)[0]

    return prediction


def predict_batch(model, scaler, feature_names, features_df):
    """Predict with multiple samples

    Args:
        model: Trained XGBoost model
        scaler: Fitted scaler
        feature_names: List of feature names in correct order
        features_df: DataFrame with features

    Returns:
        numpy array of predictions
    """
    # Ensure correct column order
    if list(features_df.columns) != feature_names:
        print("Warning: Reordering features to match training order")
        features_df = features_df[feature_names]

    # Apply scaler
    X_scaled = scaler.transform(features_df.values)

    # Predict
    dmatrix = xgb.DMatrix(X_scaled)
    predictions = model.predict(dmatrix)

    return predictions


def main():
    """Example usage"""
    print("="*60)
    print("PTrade Compatible Model Loading Example")
    print("="*60)

    # Load model package
    print("\n1. Loading model package...")
    model, scaler, feature_names, metadata = load_model_package()

    print(f"\nFeature order (CRITICAL for correct prediction):")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")

    # Example: Single prediction
    print("\n" + "="*60)
    print("2. Single Sample Prediction")
    print("="*60)

    # Simulated features (in practice, calculate from price data)
    sample_features = {
        'ma10': 50.2,
        'ma20': 49.8,
        'ma5': 50.5,
        'ma60': 48.9,
        'price_position': 0.65,
        'return_10d': 0.02,
        'return_1d': 0.005,
        'return_20d': 0.03,
        'return_5d': 0.015,
        'volatility_20d': 0.025,
        'volume_ratio': 1.2
    }

    prediction = predict_single(model, scaler, feature_names, sample_features)
    print(f"\nPrediction: {prediction:.6f}")
    print(f"Interpretation: {'UP' if prediction > 0 else 'DOWN'} ({abs(prediction)*100:.2f}%)")

    # Example: Batch prediction
    print("\n" + "="*60)
    print("3. Batch Prediction")
    print("="*60)

    # Simulated batch features
    batch_features = pd.DataFrame([
        sample_features,
        {**sample_features, 'ma5': 51.0, 'return_1d': -0.01},
        {**sample_features, 'ma10': 49.5, 'volatility_20d': 0.03}
    ])

    predictions = predict_batch(model, scaler, feature_names, batch_features)
    print(f"\nBatch predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"  Sample {i}: {pred:.6f}")

    print("\n" + "="*60)
    print("✓ All predictions completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
