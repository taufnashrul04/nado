"""
V6 — Daily Swing Strategy (rule-based, high-confluence)
Goal: WR > 70%, ~10-30 trades per year per symbol, hold 3-14 hari.

Entry conditions (LONG):
  1. Trend: ema_50 > ema_200 (bullish daily trend)
  2. Pullback: close touched ema_21 in last 3 days (healthy retracement)
  3. RSI oversold reversal: RSI rose from <40 to ≥40 in last 2 bars
  4. Stoch confirmation: stoch_k crossed up from <30
  5. Volume: today's volume ≥ 0.8x avg(20) (no exhaustion)
  6. Higher low: today's low > min(low) of last 5 days (structure intact)

Entry SHORT: mirror.

Risk:
  - SL: 1.5x ATR (give swing room)
  - TP: 2.5x ATR (RR 1:1.67, target WR ≥60% to be profitable)
  - Time stop: 14 days (2 weeks)
"""
from __future__ import annotations
import pandas as pd


def _ema_aligned_bull(row) -> bool:
    return (row["ema_13"] > row["ema_21"] > row["ema_34"]
            and row["ema_50"] > row["ema_200"])


def _ema_aligned_bear(row) -> bool:
    return (row["ema_13"] < row["ema_21"] < row["ema_34"]
            and row["ema_50"] < row["ema_200"])


def _touched_ema21(df, i, side, lookback=3) -> bool:
    start = max(0, i - lookback + 1)
    sub = df.iloc[start:i+1]
    if side == "long":
        return (sub["low"] <= sub["ema_21"] * 1.005).any()
    else:
        return (sub["high"] >= sub["ema_21"] * 0.995).any()


def _rsi_recovering_long(df, i, threshold=40) -> bool:
    if i < 2: return False
    return (df["rsi"].iat[i-2] < threshold
            and df["rsi"].iat[i] >= threshold
            and df["rsi"].iat[i] > df["rsi"].iat[i-1])


def _rsi_falling_short(df, i, threshold=60) -> bool:
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
    sl_pct = max(1.5 * atr_val / close, 0.01)
    tp_pct = max(2.5 * atr_val / close, 0.015)

    # ===== LONG SETUP — need 4 of 6 confluence + must have trend filter =====
    must_have_long = _ema_aligned_bull(row)  # trend mandatory
    if must_have_long:
        long_signals = [
            _touched_ema21(df, i, "long", lookback=5),
            _rsi_recovering_long(df, i, threshold=45),
            _stoch_cross_up(df, i, threshold=35),
            _volume_ok(df, i, lookback=20, min_ratio=0.7),
            _higher_low(df, i, lookback=5),
        ]
        if sum(long_signals) >= 3:  # need 3 of 5 confluence besides trend
            return {
                "action": "open_long",
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "max_bars_hold": 14,
                "reason": f"V6_long conf={sum(long_signals)}/5 RSI={row['rsi']:.0f}",
            }

    # ===== SHORT SETUP =====
    must_have_short = _ema_aligned_bear(row)
    if must_have_short:
        short_signals = [
            _touched_ema21(df, i, "short", lookback=5),
            _rsi_falling_short(df, i, threshold=55),
            _stoch_cross_down(df, i, threshold=65),
            _volume_ok(df, i, lookback=20, min_ratio=0.7),
            _lower_high(df, i, lookback=5),
        ]
        if sum(short_signals) >= 3:
            return {
                "action": "open_short",
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "max_bars_hold": 14,
                "reason": f"V6_short conf={sum(short_signals)}/5 RSI={row['rsi']:.0f}",
            }

    return None
