"""
V7 — 4H Intraday Swing Strategy (rule-based)
Hold: 6-12 bars (24-48 hours / 1-2 days max)
Target: WR ≥ 55%, PF ≥ 1.5

Confluence approach: trend MUST align + 3 of 5 signals
"""
from __future__ import annotations
import pandas as pd


def _ema_aligned_bull(row) -> bool:
    return (row["ema_13"] > row["ema_21"] > row["ema_34"]
            and row["ema_50"] > row["ema_200"])


def _ema_aligned_bear(row) -> bool:
    return (row["ema_13"] < row["ema_21"] < row["ema_34"]
            and row["ema_50"] < row["ema_200"])


def _touched_ema21(df, i, side, lookback=4) -> bool:
    start = max(0, i - lookback + 1)
    sub = df.iloc[start:i+1]
    if side == "long":
        return (sub["low"] <= sub["ema_21"] * 1.003).any()
    else:
        return (sub["high"] >= sub["ema_21"] * 0.997).any()


def _rsi_recovering_long(df, i, threshold=45) -> bool:
    if i < 2: return False
    return (df["rsi"].iat[i-2] < threshold
            and df["rsi"].iat[i] >= threshold
            and df["rsi"].iat[i] > df["rsi"].iat[i-1])


def _rsi_falling_short(df, i, threshold=55) -> bool:
    if i < 2: return False
    return (df["rsi"].iat[i-2] > threshold
            and df["rsi"].iat[i] <= threshold
            and df["rsi"].iat[i] < df["rsi"].iat[i-1])


def _stoch_cross_up(df, i, threshold=30) -> bool:
    if i < 1: return False
    k_prev = df["stoch_k"].iat[i-1]
    k_now = df["stoch_k"].iat[i]
    d_now = df["stoch_d"].iat[i]
    return k_prev < threshold and k_now > k_prev and k_now > d_now


def _stoch_cross_down(df, i, threshold=70) -> bool:
    if i < 1: return False
    k_prev = df["stoch_k"].iat[i-1]
    k_now = df["stoch_k"].iat[i]
    d_now = df["stoch_d"].iat[i]
    return k_prev > threshold and k_now < k_prev and k_now < d_now


def _volume_ok(df, i, lookback=20, min_ratio=0.8) -> bool:
    if i < lookback: return False
    avg = df["volume"].iloc[i-lookback:i].mean()
    if avg <= 0: return False
    return df["volume"].iat[i] >= avg * min_ratio


def _adx_ok(df, i, min_adx=18, max_adx=40) -> bool:
    """Filter chop & runaway."""
    adx = df["adx"].iat[i]
    if pd.isna(adx): return False
    return min_adx <= adx <= max_adx


def _higher_low(df, i, lookback=5) -> bool:
    if i < lookback: return False
    sub = df.iloc[i-lookback:i]
    return df["low"].iat[i] > sub["low"].min()


def _lower_high(df, i, lookback=5) -> bool:
    if i < lookback: return False
    sub = df.iloc[i-lookback:i]
    return df["high"].iat[i] < sub["high"].max()


def signal_fn(df: pd.DataFrame, i: int) -> dict | None:
    if i < 200: return None
    row = df.iloc[i]
    if pd.isna(row.get("ema_200")) or pd.isna(row.get("atr")) or pd.isna(row.get("rsi")):
        return None

    atr_val = row["atr"]
    close = row["close"]
    # 4h timeframe: SL 1.3xATR, TP 2.0xATR — RR 1:1.54, target WR ≥45% to break even
    sl_pct = max(1.3 * atr_val / close, 0.008)
    tp_pct = max(2.0 * atr_val / close, 0.012)

    # ===== LONG SETUP — trend MANDATORY + 3 of 5 signals =====
    if _ema_aligned_bull(row) and _adx_ok(df, i):
        signals = [
            _touched_ema21(df, i, "long", lookback=4),
            _rsi_recovering_long(df, i, threshold=45),
            _stoch_cross_up(df, i, threshold=35),
            _volume_ok(df, i, lookback=20, min_ratio=0.8),
            _higher_low(df, i, lookback=5),
        ]
        if sum(signals) >= 3:
            return {
                "action": "open_long",
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "max_bars_hold": 12,  # 12 × 4h = 48h max
                "reason": f"V7_long conf={sum(signals)}/5 RSI={row['rsi']:.0f} ADX={row['adx']:.0f}",
            }

    # ===== SHORT SETUP =====
    if _ema_aligned_bear(row) and _adx_ok(df, i):
        signals = [
            _touched_ema21(df, i, "short", lookback=4),
            _rsi_falling_short(df, i, threshold=55),
            _stoch_cross_down(df, i, threshold=65),
            _volume_ok(df, i, lookback=20, min_ratio=0.8),
            _lower_high(df, i, lookback=5),
        ]
        if sum(signals) >= 3:
            return {
                "action": "open_short",
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "max_bars_hold": 12,
                "reason": f"V7_short conf={sum(signals)}/5 RSI={row['rsi']:.0f} ADX={row['adx']:.0f}",
            }

    return None
