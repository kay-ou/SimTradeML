# -*- coding: utf-8 -*-
"""
MVP Training Script - Minimum Viable Product
Simple end-to-end training pipeline to validate the framework

This script demonstrates the most basic usage:
1. Load data from SimTradeLab
2. Calculate simple features
3. Train XGBoost model
4. Evaluate performance
"""

import sys
sys.path.insert(0, '/home/kay/dev/SimTradeML/src')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from scipy.stats import pearsonr, spearmanr
import logging
import json
import pickle
from datetime import datetime

# Import our components
from simtrademl.data_sources.simtradelab_source import SimTradeLabDataSource
from simtrademl.core.utils.logger import setup_logger

# Setup logger
logger = setup_logger('mvp', level='INFO', log_file='examples/mvp_train.log')


def calculate_simple_features(price_df, lookback=60):
    """Calculate simple technical features

    Args:
        price_df: Price DataFrame with columns [open, high, low, close, volume]
        lookback: Days to look back

    Returns:
        dict of features
    """
    features = {}

    closes = price_df['close'].values
    volumes = price_df['volume'].values
    highs = price_df['high'].values
    lows = price_df['low'].values

    if len(closes) < lookback:
        return None

    # Moving averages
    features['ma5'] = np.mean(closes[-5:])
    features['ma10'] = np.mean(closes[-10:])
    features['ma20'] = np.mean(closes[-20:])
    features['ma60'] = np.mean(closes[-60:])

    # Returns
    features['return_1d'] = (closes[-1] - closes[-2]) / closes[-2]
    features['return_5d'] = (closes[-1] - closes[-6]) / closes[-6]
    features['return_10d'] = (closes[-1] - closes[-11]) / closes[-11]
    features['return_20d'] = (closes[-1] - closes[-21]) / closes[-21]

    # Volatility
    returns = np.diff(closes[-20:]) / closes[-21:-1]
    features['volatility_20d'] = np.std(returns)

    # Volume features
    features['volume_ratio'] = volumes[-1] / (np.mean(volumes[-20:]) + 1e-8)

    # Price position
    features['price_position'] = (closes[-1] - np.min(closes[-20:])) / (np.max(closes[-20:]) - np.min(closes[-20:]) + 1e-8)

    return features


def collect_samples(data_source, n_stocks=50, lookback=60, predict_days=5):
    """Collect training samples

    Args:
        data_source: SimTradeLabDataSource instance
        n_stocks: Number of stocks to use (for MVP testing)
        lookback: Days to look back for features
        predict_days: Days to predict forward

    Returns:
        (X, y, dates) - features, targets, sample dates
    """
    logger.info("Collecting training samples...")

    # Get stock list (limit for MVP)
    all_stocks = data_source.get_stock_list()[:n_stocks]
    logger.info(f"Using {len(all_stocks)} stocks")

    # Get trading dates
    trading_dates = data_source.get_trading_dates()
    logger.info(f"Date range: {trading_dates[0]} to {trading_dates[-1]}")

    samples = []
    targets = []
    dates = []

    # Sample every 10 days
    sample_dates = trading_dates[lookback+20::10]

    for sample_date in sample_dates:
        logger.info(f"Processing {sample_date.strftime('%Y-%m-%d')}...")

        for stock in all_stocks:
            try:
                # Get price data
                price_df = data_source.get_price_data(
                    stock,
                    start_date=sample_date - pd.Timedelta(days=lookback+30),
                    end_date=sample_date + pd.Timedelta(days=predict_days+10)
                )

                if price_df.empty or len(price_df) < lookback:
                    continue

                # Find sample_date in index
                if sample_date not in price_df.index:
                    valid_dates = price_df.index[price_df.index <= sample_date]
                    if len(valid_dates) == 0:
                        continue
                    sample_date_actual = valid_dates[-1]
                else:
                    sample_date_actual = sample_date

                sample_idx = price_df.index.get_loc(sample_date_actual)

                # Check data availability
                if sample_idx < lookback or sample_idx + predict_days >= len(price_df):
                    continue

                # Calculate features (use data up to sample_idx, not including it)
                hist_df = price_df.iloc[sample_idx-lookback:sample_idx]
                features = calculate_simple_features(hist_df, lookback)

                if features is None:
                    continue

                # Calculate target
                current_price = price_df.iloc[sample_idx]['close']
                future_price = price_df.iloc[sample_idx + predict_days]['close']

                if current_price <= 0 or future_price <= 0:
                    continue

                future_return = (future_price - current_price) / current_price

                if not np.isfinite(future_return):
                    continue

                samples.append(features)
                targets.append(future_return)
                dates.append(sample_date_actual)

            except Exception as e:
                logger.debug(f"Error processing {stock}: {e}")
                continue

    logger.info(f"Collected {len(samples)} samples")

    # Create DataFrame with fixed column order
    X = pd.DataFrame(samples)
    # Sort columns alphabetically to ensure consistent order
    X = X[sorted(X.columns)]
    feature_names = list(X.columns)

    y = np.array(targets)
    sample_dates = pd.Series(dates)

    logger.info(f"Feature names (in order): {feature_names}")

    return X, y, sample_dates, feature_names


def train_xgboost(X, y, sample_dates):
    """Train XGBoost model

    Args:
        X: Feature DataFrame
        y: Target array
        sample_dates: Sample dates

    Returns:
        (model, scaler) - trained model and fitted scaler
    """
    logger.info(f"\nTraining XGBoost model...")
    logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

    # Time-based split
    unique_dates = sorted(sample_dates.unique())
    train_end_idx = int(len(unique_dates) * 0.7)
    val_end_idx = int(len(unique_dates) * 0.85)

    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]

    train_mask = sample_dates.isin(train_dates).values
    val_mask = sample_dates.isin(val_dates).values
    test_mask = sample_dates.isin(test_dates).values

    X_train = X.values[train_mask]
    y_train = y[train_mask]
    X_val = X.values[val_mask]
    y_val = y[val_mask]
    X_test = X.values[test_mask]
    y_test = y[test_mask]

    logger.info(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Standardize features
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train XGBoost
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.04,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'seed': 42,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evals = [(dtrain, 'train'), (dval, 'val')]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    logger.info(f"Training completed at round {model.best_iteration}")

    # Evaluate
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SET")
    logger.info("="*60)
    evaluate(model, dval, y_val, sample_dates[val_mask].values)

    logger.info("\n" + "="*60)
    logger.info("TEST SET")
    logger.info("="*60)
    evaluate(model, dtest, y_test, sample_dates[test_mask].values)

    return model, scaler


def evaluate(model, dmatrix, y_true, dates=None):
    """Evaluate model performance

    Args:
        model: Trained model
        dmatrix: XGBoost DMatrix
        y_true: True labels
        dates: Sample dates (optional)
    """
    y_pred = model.predict(dmatrix)

    # IC metrics
    ic, ic_pval = pearsonr(y_pred, y_true)
    rank_ic, rank_ic_pval = spearmanr(y_pred, y_true)

    logger.info(f"IC (Pearson):  {ic:.4f} (p={ic_pval:.2e})")
    logger.info(f"Rank IC:       {rank_ic:.4f} (p={rank_ic_pval:.2e})")

    # Direction accuracy
    direction_acc = np.sum((y_pred * y_true) > 0) / len(y_true)
    logger.info(f"Direction Acc: {direction_acc:.2%}")

    # Quantile returns
    if dates is not None:
        quantile_returns = calculate_quantile_returns_daily(y_pred, y_true, dates)
        logger.info("\nQuantile Returns (daily rebalanced):")
        for i, ret in enumerate(quantile_returns, 1):
            logger.info(f"  Q{i}: {ret:.2%}")
        logger.info(f"  Long-Short: {quantile_returns[-1] - quantile_returns[0]:.2%}")


def calculate_quantile_returns_daily(predictions, actuals, dates, n_quantiles=5):
    """Calculate quantile returns with daily rebalancing"""
    unique_dates = np.unique(dates)
    daily_quantile_returns = [[] for _ in range(n_quantiles)]

    for date in unique_dates:
        date_mask = (dates == date)
        if np.sum(date_mask) < 10:
            continue

        day_pred = predictions[date_mask]
        day_actual = actuals[date_mask]

        # Quantile split within this day
        day_percentiles = np.percentile(day_pred, np.linspace(0, 100, n_quantiles + 1))
        day_quantiles = np.digitize(day_pred, day_percentiles[1:-1])

        for q in range(n_quantiles):
            q_mask = (day_quantiles == q)
            if np.sum(q_mask) > 0:
                q_return = np.mean(day_actual[q_mask])
                daily_quantile_returns[q].append(q_return)

    # Average across days
    quantile_returns = [
        np.mean(returns) if returns else 0.0
        for returns in daily_quantile_returns
    ]

    return quantile_returns


def main():
    """Main MVP training pipeline"""
    logger.info("="*60)
    logger.info("SimTradeML MVP Training")
    logger.info("="*60)

    # 1. Initialize data source
    logger.info("\n1. Initializing SimTradeLab data source...")
    data_source = SimTradeLabDataSource()

    # 2. Collect samples
    logger.info("\n2. Collecting training samples...")
    X, y, sample_dates, feature_names = collect_samples(data_source, n_stocks=100)

    # 3. Train model
    logger.info("\n3. Training model...")
    model, scaler = train_xgboost(X, y, sample_dates)

    # 4. Save model and metadata
    logger.info("\n4. Saving model and metadata...")

    # Save model (JSON format)
    model.save_model('examples/mvp_model.json')
    logger.info("✓ Model saved to examples/mvp_model.json")

    # Save scaler (Pickle format)
    with open('examples/mvp_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("✓ Scaler saved to examples/mvp_scaler.pkl")

    # Save feature names
    with open('examples/mvp_features.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    logger.info("✓ Feature names saved to examples/mvp_features.json")

    # Save metadata
    metadata = {
        'model_id': f'mvp_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'version': '1.0',
        'created_at': datetime.now().isoformat(),
        'xgboost_version': xgb.__version__,
        'features': feature_names,
        'n_features': len(feature_names),
        'scaler_type': 'RobustScaler',
        'n_samples': len(y),
        'files': {
            'model': 'mvp_model.json',
            'scaler': 'mvp_scaler.pkl',
            'features': 'mvp_features.json'
        }
    }

    with open('examples/mvp_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("✓ Metadata saved to examples/mvp_metadata.json")

    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
