# -*- coding: utf-8 -*-
"""
Technical Indicator Features

Common technical analysis indicators for stock trading.
All features from mvp_train.py (11 basic features).
"""

import pandas as pd
import numpy as np
from .registry import FeatureRegistry


# ============================================================================
# Moving Averages (4 features)
# ============================================================================

@FeatureRegistry.register('ma5', category='technical', description='5-day moving average')
def ma5(price_df: pd.DataFrame) -> float:
    """5-day moving average of close price"""
    return float(price_df['close'].rolling(5).mean().iloc[-1])


@FeatureRegistry.register('ma10', category='technical')
def ma10(price_df: pd.DataFrame) -> float:
    """10-day moving average"""
    return float(price_df['close'].rolling(10).mean().iloc[-1])


@FeatureRegistry.register('ma20', category='technical')
def ma20(price_df: pd.DataFrame) -> float:
    """20-day moving average"""
    return float(price_df['close'].rolling(20).mean().iloc[-1])


@FeatureRegistry.register('ma60', category='technical')
def ma60(price_df: pd.DataFrame) -> float:
    """60-day moving average"""
    return float(price_df['close'].rolling(60).mean().iloc[-1])


# ============================================================================
# Returns (4 features)
# ============================================================================

@FeatureRegistry.register('return_1d', category='technical', description='1-day return')
def return_1d(price_df: pd.DataFrame) -> float:
    """1-day return"""
    closes = price_df['close'].values
    if len(closes) < 2:
        return 0.0
    return float((closes[-1] - closes[-2]) / closes[-2])


@FeatureRegistry.register('return_5d', category='technical')
def return_5d(price_df: pd.DataFrame) -> float:
    """5-day return"""
    closes = price_df['close'].values
    if len(closes) < 6:
        return 0.0
    return float((closes[-1] - closes[-6]) / closes[-6])


@FeatureRegistry.register('return_10d', category='technical')
def return_10d(price_df: pd.DataFrame) -> float:
    """10-day return"""
    closes = price_df['close'].values
    if len(closes) < 11:
        return 0.0
    return float((closes[-1] - closes[-11]) / closes[-11])


@FeatureRegistry.register('return_20d', category='technical')
def return_20d(price_df: pd.DataFrame) -> float:
    """20-day return"""
    closes = price_df['close'].values
    if len(closes) < 21:
        return 0.0
    return float((closes[-1] - closes[-21]) / closes[-21])


# ============================================================================
# Volatility & Volume (2 features)
# ============================================================================

@FeatureRegistry.register('volatility_20d', category='technical')
def volatility_20d(price_df: pd.DataFrame) -> float:
    """20-day volatility (std of returns)"""
    closes = price_df['close'].values
    if len(closes) < 21:
        return 0.0
    # Take last 21 values to calculate 20 returns
    recent_closes = closes[-21:]
    returns = np.diff(recent_closes) / recent_closes[:-1]
    return float(np.std(returns))


@FeatureRegistry.register('volume_ratio', category='technical')
def volume_ratio(price_df: pd.DataFrame) -> float:
    """Current volume / 20-day average volume"""
    volumes = price_df['volume'].values
    if len(volumes) < 20:
        return 1.0
    return float(volumes[-1] / (np.mean(volumes[-20:]) + 1e-8))


# ============================================================================
# Price Position (1 feature)
# ============================================================================

@FeatureRegistry.register('price_position', category='technical')
def price_position(price_df: pd.DataFrame) -> float:
    """Current price position in 20-day range (0-1)"""
    closes = price_df['close'].values[-20:]
    if len(closes) < 20:
        return 0.5

    price_min = np.min(closes)
    price_max = np.max(closes)
    current = closes[-1]

    if price_max - price_min < 1e-8:
        return 0.5

    return float((current - price_min) / (price_max - price_min))


# ============================================================================
# Total: 11 features (matching mvp_train.py)
# ============================================================================


# ============================================================================
# Extended Technical Indicators (20+ additional features)
# ============================================================================

# ============================================================================
# Momentum Indicators (4 features)
# ============================================================================

@FeatureRegistry.register('rsi14', category='technical', description='14-day RSI')
def rsi14(price_df: pd.DataFrame) -> float:
    """14-day Relative Strength Index"""
    closes = price_df['close'].values
    if len(closes) < 15:
        return 50.0  # Neutral RSI

    deltas = np.diff(closes[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)


@FeatureRegistry.register('cci20', category='technical', description='20-day CCI')
def cci20(price_df: pd.DataFrame) -> float:
    """20-day Commodity Channel Index"""
    if len(price_df) < 20:
        return 0.0

    typical_price = (price_df['high'] + price_df['low'] + price_df['close']) / 3
    sma = typical_price.rolling(20).mean().iloc[-1]
    mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean()).iloc[-1]

    if mad == 0:
        return 0.0

    cci = (typical_price.iloc[-1] - sma) / (0.015 * mad)
    return float(cci)


@FeatureRegistry.register('roc10', category='technical', description='10-day Rate of Change')
def roc10(price_df: pd.DataFrame) -> float:
    """10-day Rate of Change"""
    closes = price_df['close'].values
    if len(closes) < 11:
        return 0.0
    return float((closes[-1] - closes[-11]) / closes[-11] * 100)


@FeatureRegistry.register('williams_r14', category='technical', description='14-day Williams %R')
def williams_r14(price_df: pd.DataFrame) -> float:
    """14-day Williams %R"""
    if len(price_df) < 14:
        return -50.0  # Neutral value

    highs = price_df['high'].values[-14:]
    lows = price_df['low'].values[-14:]
    close = price_df['close'].values[-1]

    highest_high = np.max(highs)
    lowest_low = np.min(lows)

    if highest_high - lowest_low == 0:
        return -50.0

    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return float(wr)


# ============================================================================
# Trend Indicators (5 features)
# ============================================================================

@FeatureRegistry.register('macd', category='technical', description='MACD Line')
def macd(price_df: pd.DataFrame) -> float:
    """MACD Line (12-day EMA - 26-day EMA)"""
    closes = price_df['close']
    if len(closes) < 26:
        return 0.0

    ema12 = closes.ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = closes.ewm(span=26, adjust=False).mean().iloc[-1]

    return float(ema12 - ema26)


@FeatureRegistry.register('macd_signal', category='technical', description='MACD Signal Line')
def macd_signal(price_df: pd.DataFrame) -> float:
    """MACD Signal Line (9-day EMA of MACD)"""
    closes = price_df['close']
    if len(closes) < 35:  # Need 26 + 9 days
        return 0.0

    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean().iloc[-1]

    return float(signal)


@FeatureRegistry.register('macd_histogram', category='technical', description='MACD Histogram')
def macd_histogram(price_df: pd.DataFrame) -> float:
    """MACD Histogram (MACD - Signal)"""
    closes = price_df['close']
    if len(closes) < 35:
        return 0.0

    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line.iloc[-1] - signal.iloc[-1]

    return float(histogram)


@FeatureRegistry.register('bb_upper', category='technical', description='Bollinger Upper Band (20,2)')
def bb_upper(price_df: pd.DataFrame) -> float:
    """Bollinger Bands Upper Band"""
    closes = price_df['close']
    if len(closes) < 20:
        return float(closes.iloc[-1])

    ma20 = closes.rolling(20).mean().iloc[-1]
    std20 = closes.rolling(20).std().iloc[-1]

    return float(ma20 + 2 * std20)


@FeatureRegistry.register('bb_lower', category='technical', description='Bollinger Lower Band (20,2)')
def bb_lower(price_df: pd.DataFrame) -> float:
    """Bollinger Bands Lower Band"""
    closes = price_df['close']
    if len(closes) < 20:
        return float(closes.iloc[-1])

    ma20 = closes.rolling(20).mean().iloc[-1]
    std20 = closes.rolling(20).std().iloc[-1]

    return float(ma20 - 2 * std20)


@FeatureRegistry.register('atr14', category='technical', description='14-day Average True Range')
def atr14(price_df: pd.DataFrame) -> float:
    """14-day Average True Range"""
    if len(price_df) < 15:
        return 0.0

    high = price_df['high'].values[-15:]
    low = price_df['low'].values[-15:]
    close = price_df['close'].values[-15:]

    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])

    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.mean(tr)

    return float(atr)


# ============================================================================
# KDJ Indicator (3 features)
# ============================================================================

@FeatureRegistry.register('kdj_k', category='technical', description='KDJ K value')
def kdj_k(price_df: pd.DataFrame) -> float:
    """KDJ K value"""
    if len(price_df) < 9:
        return 50.0

    recent = price_df.iloc[-9:]
    low_min = recent['low'].min()
    high_max = recent['high'].max()
    close = price_df['close'].iloc[-1]

    if high_max - low_min == 0:
        return 50.0

    rsv = 100 * (close - low_min) / (high_max - low_min)
    return float(rsv)  # Simplified K = RSV for current calculation


@FeatureRegistry.register('kdj_d', category='technical', description='KDJ D value')
def kdj_d(price_df: pd.DataFrame) -> float:
    """KDJ D value (smoothed K)"""
    # Simplified implementation: use 3-period moving average of K
    if len(price_df) < 11:
        return 50.0

    k_values = []
    for i in range(3):
        window = price_df.iloc[-(9+i):len(price_df)-i] if i > 0 else price_df.iloc[-9:]
        low_min = window['low'].min()
        high_max = window['high'].max()
        close = window['close'].iloc[-1]

        if high_max - low_min == 0:
            k_values.append(50.0)
        else:
            rsv = 100 * (close - low_min) / (high_max - low_min)
            k_values.append(rsv)

    return float(np.mean(k_values))


@FeatureRegistry.register('kdj_j', category='technical', description='KDJ J value')
def kdj_j(price_df: pd.DataFrame) -> float:
    """KDJ J value (3K - 2D)"""
    k = kdj_k(price_df)
    d = kdj_d(price_df)
    j = 3 * k - 2 * d
    return float(j)


# ============================================================================
# Moving Average Ratios & Differences (8 features)
# ============================================================================

@FeatureRegistry.register('ma5_ma10_ratio', category='technical')
def ma5_ma10_ratio(price_df: pd.DataFrame) -> float:
    """MA5 / MA10 ratio"""
    if len(price_df['close']) < 10:
        return 1.0
    ma5_val = price_df['close'].rolling(5).mean().iloc[-1]
    ma10_val = price_df['close'].rolling(10).mean().iloc[-1]
    if ma10_val == 0:
        return 1.0
    return float(ma5_val / ma10_val)


@FeatureRegistry.register('ma5_ma20_ratio', category='technical')
def ma5_ma20_ratio(price_df: pd.DataFrame) -> float:
    """MA5 / MA20 ratio"""
    if len(price_df['close']) < 20:
        return 1.0
    ma5_val = price_df['close'].rolling(5).mean().iloc[-1]
    ma20_val = price_df['close'].rolling(20).mean().iloc[-1]
    if ma20_val == 0:
        return 1.0
    return float(ma5_val / ma20_val)


@FeatureRegistry.register('ma10_ma20_ratio', category='technical')
def ma10_ma20_ratio(price_df: pd.DataFrame) -> float:
    """MA10 / MA20 ratio"""
    if len(price_df['close']) < 20:
        return 1.0
    ma10_val = price_df['close'].rolling(10).mean().iloc[-1]
    ma20_val = price_df['close'].rolling(20).mean().iloc[-1]
    if ma20_val == 0:
        return 1.0
    return float(ma10_val / ma20_val)


@FeatureRegistry.register('ma20_ma60_ratio', category='technical')
def ma20_ma60_ratio(price_df: pd.DataFrame) -> float:
    """MA20 / MA60 ratio"""
    if len(price_df['close']) < 60:
        return 1.0
    ma20_val = price_df['close'].rolling(20).mean().iloc[-1]
    ma60_val = price_df['close'].rolling(60).mean().iloc[-1]
    if ma60_val == 0:
        return 1.0
    return float(ma20_val / ma60_val)


@FeatureRegistry.register('price_ma5_diff', category='technical')
def price_ma5_diff(price_df: pd.DataFrame) -> float:
    """(Price - MA5) / MA5"""
    if len(price_df['close']) < 5:
        return 0.0
    price = price_df['close'].iloc[-1]
    ma5_val = price_df['close'].rolling(5).mean().iloc[-1]
    if ma5_val == 0:
        return 0.0
    return float((price - ma5_val) / ma5_val)


@FeatureRegistry.register('price_ma10_diff', category='technical')
def price_ma10_diff(price_df: pd.DataFrame) -> float:
    """(Price - MA10) / MA10"""
    if len(price_df['close']) < 10:
        return 0.0
    price = price_df['close'].iloc[-1]
    ma10_val = price_df['close'].rolling(10).mean().iloc[-1]
    if ma10_val == 0:
        return 0.0
    return float((price - ma10_val) / ma10_val)


@FeatureRegistry.register('price_ma20_diff', category='technical')
def price_ma20_diff(price_df: pd.DataFrame) -> float:
    """(Price - MA20) / MA20"""
    if len(price_df['close']) < 20:
        return 0.0
    price = price_df['close'].iloc[-1]
    ma20_val = price_df['close'].rolling(20).mean().iloc[-1]
    if ma20_val == 0:
        return 0.0
    return float((price - ma20_val) / ma20_val)


@FeatureRegistry.register('price_ma60_diff', category='technical')
def price_ma60_diff(price_df: pd.DataFrame) -> float:
    """(Price - MA60) / MA60"""
    if len(price_df['close']) < 60:
        return 0.0
    price = price_df['close'].iloc[-1]
    ma60_val = price_df['close'].rolling(60).mean().iloc[-1]
    if ma60_val == 0:
        return 0.0
    return float((price - ma60_val) / ma60_val)


# ============================================================================
# Total: 11 (basic) + 21 (extended) = 32 technical indicators
# ============================================================================
