"""
V4 — 15m Scalping Strategy (rule-based, no LLM)

Approach: Mean-reversion scalp di key EMA dengan Stoch extreme.
Target: WR > 65%, ~2-5 trades per symbol per hari.

Entry conditions (LONG):
  1. Trend filter: ema_50 > ema_200 (HTF bias bullish)
  2. Pullback: low touched ema_21 in last 3 bars (tested support)
  3. Stoch K crossed up from < 20 (oversold reversal)
  4. ADX between 15-35 (avoid range chop & strong trend that breaks pullback)
  5. Volume on entry bar > 1.2x avg(20) (confirmation)

Entry conditions (SHORT): mirror.

Risk:
  - SL: 0.8x ATR (tight, scalp logic)
  - TP: 1.2x ATR (RR 1:1.5, target WR 65%+ to be profitable)
  - Time stop: exit after 16 bars (4 jam) if no TP/SL
"""
from __future__ import annotations
import pandas as pd


def _ema_aligned_bull(row) -> bool:
    return row["ema_13"] > row["ema_21"] > row["ema_34"] and row["ema_50"] > row["ema_200"]


def _ema_aligned_bear(row) -> bool:
    return row["ema_13"] < row["ema_21"] < row["ema_34"] and row["ema_50"] < row["ema_200"]


def _touched_ema21_recently(df: pd.DataFrame, i: int, side: str, lookback: int = 3) -> bool:
    """Check if price touched ema_21 in last `lookback` bars."""
    start = max(0, i - lookback + 1)
    sub = df.iloc[start:i+1]
    if side == "long":
        # low must have touched or gone below ema_21
        return (sub["low"] <= sub["ema_21"]).any()
    else:
        return (sub["high"] >= sub["ema_21"]).any()


def _stoch_cross_up_from_oversold(df: pd.DataFrame, i: int, threshold: float = 20) -> bool:
    if i < 1: return False
    k_prev = df["stoch_k"].iat[i-1]
    k_now = df["stoch_k"].iat[i]
    d_now = df["stoch_d"].iat[i]
    return k_prev < threshold and k_now > k_prev and k_now > d_now


def _stoch_cross_down_from_overbought(df: pd.DataFrame, i: int, threshold: float = 80) -> bool:
    if i < 1: return False
    k_prev = df["stoch_k"].iat[i-1]
    k_now = df["stoch_k"].iat[i]
    d_now = df["stoch_d"].iat[i]
    return k_prev > threshold and k_now < k_prev and k_now < d_now


def _volume_spike(df: pd.DataFrame, i: int, lookback: int = 20, mult: float = 1.2) -> bool:
    if i < lookback: return False
    avg_vol = df["volume"].iloc[i-lookback:i].mean()
    if avg_vol <= 0: return False
    return df["volume"].iat[i] >= avg_vol * mult


def signal_fn(df: pd.DataFrame, i: int) -> dict | None:
    if i < 200: return None
    row = df.iloc[i]
    # Sanity: indicators ready
    if pd.isna(row.get("ema_200")) or pd.isna(row.get("stoch_k")) or pd.isna(row.get("atr")) or pd.isna(row.get("adx")):
        return None

    adx_val = row["adx"]
    # ADX filter — sweet spot for scalp (avoid total chop & strong runaway)
    if adx_val < 18 or adx_val > 30:
        return None

    atr_val = row["atr"]
    close = row["close"]

    # ATR sanity (skip ultra-low vol = no profit room)
    atr_pct = atr_val / close
    if atr_pct < 0.001:  # < 0.1% — too tight
        return None

    # V4.1: Widen SL (1.2x ATR), tighten TP (1.0x ATR) — prioritize WR over RR
    sl_pct = max(1.2 * atr_val / close, 0.004)
    tp_pct = max(1.0 * atr_val / close, 0.004)

    # LONG setup
    if (
        _ema_aligned_bull(row)
        and _touched_ema21_recently(df, i, "long", lookback=3)
        and _stoch_cross_up_from_oversold(df, i, threshold=25)
        and _volume_spike(df, i, lookback=20, mult=1.2)
    ):
        return {
            "action": "open_long",
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "max_bars_hold": 16,
            "reason": f"V4_long ADX={adx_val:.0f} StochK={row['stoch_k']:.0f}",
        }

    # SHORT setup
    if (
        _ema_aligned_bear(row)
        and _touched_ema21_recently(df, i, "short", lookback=3)
        and _stoch_cross_down_from_overbought(df, i, threshold=75)
        and _volume_spike(df, i, lookback=20, mult=1.2)
    ):
        return {
            "action": "open_short",
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "max_bars_hold": 16,
            "reason": f"V4_short ADX={adx_val:.0f} StochK={row['stoch_k']:.0f}",
        }

    return None
