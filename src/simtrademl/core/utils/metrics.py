# -*- coding: utf-8 -*-
"""
Evaluation metrics for quantitative models
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Tuple, List, Optional


def calculate_ic(predictions: np.ndarray, actuals: np.ndarray) -> Tuple[float, float]:
    """Calculate Information Coefficient (Pearson correlation)

    Args:
        predictions: Predicted values
        actuals: Actual values

    Returns:
        (IC, p-value)
    """
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have same length")

    # Remove NaN values
    mask = np.isfinite(predictions) & np.isfinite(actuals)
    predictions = predictions[mask]
    actuals = actuals[mask]

    if len(predictions) < 2:
        return 0.0, 1.0

    ic, p_value = pearsonr(predictions, actuals)
    return ic, p_value


def calculate_rank_ic(predictions: np.ndarray, actuals: np.ndarray) -> Tuple[float, float]:
    """Calculate Rank Information Coefficient (Spearman correlation)

    Args:
        predictions: Predicted values
        actuals: Actual values

    Returns:
        (Rank IC, p-value)
    """
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have same length")

    # Remove NaN values
    mask = np.isfinite(predictions) & np.isfinite(actuals)
    predictions = predictions[mask]
    actuals = actuals[mask]

    if len(predictions) < 2:
        return 0.0, 1.0

    rank_ic, p_value = spearmanr(predictions, actuals)
    return rank_ic, p_value


def calculate_icir(predictions: np.ndarray, actuals: np.ndarray, window_size: int = 200) -> Tuple[float, float]:
    """Calculate IC Information Ratio (IC mean / IC std)

    Args:
        predictions: Predicted values
        actuals: Actual values
        window_size: Rolling window size for IC calculation

    Returns:
        (ICIR, IC_std)
    """
    if len(predictions) < window_size:
        ic, _ = calculate_ic(predictions, actuals)
        return float(ic), 0.0

    ic_series = []
    for i in range(0, len(predictions), window_size):
        end_idx = min(i + window_size, len(predictions))
        if end_idx - i < 50:  # Minimum window
            continue

        window_pred = predictions[i:end_idx]
        window_actual = actuals[i:end_idx]

        ic, _ = calculate_ic(window_pred, window_actual)
        if np.isfinite(ic):
            ic_series.append(ic)

    if len(ic_series) < 2:
        return 0.0, 0.0

    ic_mean = np.mean(ic_series)
    ic_std = np.std(ic_series, ddof=1)
    icir = ic_mean / ic_std if ic_std > 0 else 0.0

    return float(icir), float(ic_std)


def calculate_quantile_returns(
    predictions: np.ndarray,
    actuals: np.ndarray,
    dates: Optional[np.ndarray] = None,
    n_quantiles: int = 5
) -> Tuple[List[float], float]:
    """Calculate returns for each quantile (daily rebalanced if dates provided)

    Args:
        predictions: Predicted values
        actuals: Actual returns
        dates: Optional dates for daily rebalancing
        n_quantiles: Number of quantiles

    Returns:
        (quantile_returns, long_short_return)
            quantile_returns: Average return for each quantile
            long_short_return: Q5 - Q1 return
    """
    if dates is not None and len(dates) > 0:
        # Daily rebalancing (more realistic)
        unique_dates = np.unique(dates)
        daily_quantile_returns: List[List[float]] = [[] for _ in range(n_quantiles)]

        for date in unique_dates:
            date_mask = (dates == date)
            if np.sum(date_mask) < 10:  # Too few samples
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
                    daily_quantile_returns[q].append(float(q_return))

        # Average across days
        quantile_returns: List[float] = [
            float(np.mean(returns)) if returns else 0.0
            for returns in daily_quantile_returns
        ]
    else:
        # Global quantile split (less realistic)
        percentiles = np.linspace(0, 100, n_quantiles + 1)
        quantile_labels = np.percentile(predictions, percentiles)
        pred_quantiles = np.digitize(predictions, quantile_labels[1:-1])

        quantile_returns = []
        for q in range(n_quantiles):
            mask = (pred_quantiles == q)
            if np.sum(mask) > 0:
                avg_return = np.mean(actuals[mask])
                quantile_returns.append(float(avg_return))
            else:
                quantile_returns.append(0.0)

    long_short_return = quantile_returns[-1] - quantile_returns[0] if len(quantile_returns) >= 5 else 0.0

    return quantile_returns, float(long_short_return)


def calculate_direction_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Calculate direction accuracy (same sign)

    Args:
        predictions: Predicted values
        actuals: Actual values

    Returns:
        Direction accuracy (0-1)
    """
    mask = np.isfinite(predictions) & np.isfinite(actuals)
    predictions = predictions[mask]
    actuals = actuals[mask]

    if len(predictions) == 0:
        return 0.0

    correct = np.sum((predictions * actuals) > 0)
    return correct / len(predictions)
