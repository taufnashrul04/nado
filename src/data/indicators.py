"""
Custom indicators (Stoch + EMA) — pure pandas/numpy implementation.
No ta-lib dependency.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=1).mean()


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 5, d_period: int = 3, smooth_k: int = 3) -> tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator (Slow).
    User request: 5, 3, 3 → k_period=5, d_period=3, smooth_k=3

    Returns (%K, %D)
    """
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    raw_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    k = raw_k.rolling(window=smooth_k, min_periods=1).mean()  # smooth %K
    d = k.rolling(window=d_period, min_periods=1).mean()       # %D
    return k, d


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) <= 0] = 0
    minus_dm[(minus_dm - plus_dm) <= 0] = 0

    tr = atr(high, low, close, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta).where(delta < 0, 0.0).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame,
                   ema_periods: list[int] = (13, 21, 34, 50, 90, 200),
                   stoch_params: tuple[int, int, int] = (5, 3, 3)) -> pd.DataFrame:
    """
    Adds all indicators per user spec:
    - EMA: 13, 21, 34, 50, 90, 200
    - Stochastic: 5, 3, 3
    - Plus ATR(14), ADX(14), RSI(14) for risk/filter
    """
    out = df.copy()
    for p in ema_periods:
        out[f"ema_{p}"] = ema(out["close"], p)

    k, d = stochastic(out["high"], out["low"], out["close"], *stoch_params)
    out["stoch_k"] = k
    out["stoch_d"] = d

    out["atr"] = atr(out["high"], out["low"], out["close"], 14)
    out["adx"] = adx(out["high"], out["low"], out["close"], 14)
    out["rsi"] = rsi(out["close"], 14)
    return out
