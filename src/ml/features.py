"""
Feature engineering for ML strategy.
Computes 40+ features per bar from OHLCV.
Target: TP-first (1) vs SL-first (0) within next N bars given fixed SL/TP distances.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 40+ features. Input must have OHLCV; output adds feature cols."""
    f = df.copy()
    o, h, l, c, v = f["open"], f["high"], f["low"], f["close"], f["volume"]

    # ===== Returns =====
    f["ret_1"] = c.pct_change(1)
    f["ret_4"] = c.pct_change(4)
    f["ret_16"] = c.pct_change(16)
    f["ret_96"] = c.pct_change(96)  # 1 day on 15m

    # ===== EMAs (user spec stack) =====
    for p in [13, 21, 34, 50, 90, 200]:
        f[f"ema_{p}"] = c.ewm(span=p, adjust=False).mean()

    f["ema_13_21_diff"] = (f["ema_13"] - f["ema_21"]) / c
    f["ema_21_50_diff"] = (f["ema_21"] - f["ema_50"]) / c
    f["ema_50_200_diff"] = (f["ema_50"] - f["ema_200"]) / c
    f["close_ema21_dist"] = (c - f["ema_21"]) / c
    f["close_ema50_dist"] = (c - f["ema_50"]) / c
    f["close_ema200_dist"] = (c - f["ema_200"]) / c

    # EMA stack alignment score (-1 perfect bear, +1 perfect bull)
    def stack_score(row):
        emas = [row["ema_13"], row["ema_21"], row["ema_34"], row["ema_50"], row["ema_90"], row["ema_200"]]
        if all(emas[i] > emas[i+1] for i in range(len(emas)-1)): return 1.0
        if all(emas[i] < emas[i+1] for i in range(len(emas)-1)): return -1.0
        if emas[0] > emas[1] > emas[2]: return 0.5
        if emas[0] < emas[1] < emas[2]: return -0.5
        return 0.0
    f["ema_stack"] = f.apply(stack_score, axis=1)

    # ===== Stochastic (5,3,3) per user spec =====
    period = 5
    k_period = 3
    d_period = 3
    low_min = l.rolling(period).min()
    high_max = h.rolling(period).max()
    raw_k = 100 * (c - low_min) / (high_max - low_min + 1e-10)
    f["stoch_k_raw"] = raw_k
    f["stoch_k"] = raw_k.rolling(k_period).mean()
    f["stoch_d"] = f["stoch_k"].rolling(d_period).mean()
    f["stoch_k_d_diff"] = f["stoch_k"] - f["stoch_d"]
    f["stoch_k_lag1"] = f["stoch_k"].shift(1)
    f["stoch_k_change"] = f["stoch_k"] - f["stoch_k_lag1"]

    # ===== ATR =====
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr"] = tr.ewm(span=14, adjust=False).mean()
    f["atr_pct"] = f["atr"] / c
    f["atr_zscore_50"] = (f["atr"] - f["atr"].rolling(50).mean()) / (f["atr"].rolling(50).std() + 1e-10)

    # ===== ADX =====
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move
    atr_smooth = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / (atr_smooth + 1e-10)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / (atr_smooth + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    f["adx"] = dx.ewm(span=14, adjust=False).mean()
    f["plus_di"] = plus_di
    f["minus_di"] = minus_di
    f["di_diff"] = plus_di - minus_di

    # ===== RSI =====
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    f["rsi"] = 100 - 100 / (1 + rs)

    # ===== Bollinger Band width =====
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    f["bb_width"] = 4 * bb_std / (bb_mid + 1e-10)
    f["bb_pct_b"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)

    # ===== Volume features =====
    f["vol_ema_20"] = v.ewm(span=20, adjust=False).mean()
    f["vol_ratio"] = v / (f["vol_ema_20"] + 1e-10)
    f["vol_zscore_50"] = (v - v.rolling(50).mean()) / (v.rolling(50).std() + 1e-10)

    # ===== Candle body / wick =====
    body = (c - o).abs()
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    candle_range = h - l + 1e-10
    f["body_pct"] = body / candle_range
    f["upper_wick_pct"] = upper_wick / candle_range
    f["lower_wick_pct"] = lower_wick / candle_range
    f["bull_candle"] = (c > o).astype(int)

    # ===== Recent extremes =====
    f["dist_from_high_20"] = (c - h.rolling(20).max()) / c
    f["dist_from_low_20"] = (c - l.rolling(20).min()) / c
    f["dist_from_high_96"] = (c - h.rolling(96).max()) / c
    f["dist_from_low_96"] = (c - l.rolling(96).min()) / c

    # ===== Time features =====
    f["hour"] = f.index.hour
    f["dow"] = f.index.dayofweek
    f["is_weekend"] = (f.index.dayofweek >= 5).astype(int)

    # ===== Lag features (price action context) =====
    for lag in [1, 2, 3, 5]:
        f[f"ret_lag_{lag}"] = f["ret_1"].shift(lag)
        f[f"stoch_k_lag_{lag}"] = f["stoch_k"].shift(lag)

    return f


def compute_labels(df: pd.DataFrame, side: str = "long",
                   sl_atr_mult: float = 1.2, tp_atr_mult: float = 1.0,
                   max_bars_hold: int = 16) -> pd.Series:
    """
    Compute labels: 1 if TP hit before SL within max_bars_hold; 0 otherwise.
    Needs 'atr' column.
    """
    labels = pd.Series(np.nan, index=df.index, dtype=float)
    atr = df["atr"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    for i in range(len(df) - max_bars_hold):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        entry = c[i]
        if side == "long":
            sl = entry - sl_atr_mult * atr[i]
            tp = entry + tp_atr_mult * atr[i]
        else:
            sl = entry + sl_atr_mult * atr[i]
            tp = entry - tp_atr_mult * atr[i]

        for j in range(i + 1, min(i + 1 + max_bars_hold, len(df))):
            hi, lo = h[j], l[j]
            if side == "long":
                # Conservative: assume SL hit first if both touched
                if lo <= sl and hi >= tp:
                    labels.iat[i] = 0  # assume SL first
                    break
                if lo <= sl:
                    labels.iat[i] = 0
                    break
                if hi >= tp:
                    labels.iat[i] = 1
                    break
            else:
                if hi >= sl and lo <= tp:
                    labels.iat[i] = 0
                    break
                if hi >= sl:
                    labels.iat[i] = 0
                    break
                if lo <= tp:
                    labels.iat[i] = 1
                    break
        else:
            # Time stop = neither hit; treat as 0 (no win)
            labels.iat[i] = 0

    return labels
