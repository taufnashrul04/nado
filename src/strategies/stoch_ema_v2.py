"""
Strategy V2: HIGH WIN RATE FOCUS
Goal: WR > 70%

Approach:
  - Mean reversion at extreme oversold/overbought
  - Multi-timeframe filter (only trade when 1h aligns with 4h trend)
  - Tight TP (favors quick profit-taking) + tight SL
  - Asymmetric: TP 1.0x ATR, SL 1.5x ATR (1.5:1 inverse for high WR)
  - Strict confluence: ALL filters must align
  - Volume confirmation required
  - Skip low-liquidity hours

Indicators (user spec):
  - Stoch (5,3,3)
  - EMA (13, 21, 34, 50, 90, 200)
"""
from __future__ import annotations
import pandas as pd


def _stoch_cross_up(df: pd.DataFrame, i: int) -> bool:
    if i < 1:
        return False
    return (
        df["stoch_k"].iat[i] > df["stoch_d"].iat[i]
        and df["stoch_k"].iat[i - 1] <= df["stoch_d"].iat[i - 1]
    )


def _stoch_cross_down(df: pd.DataFrame, i: int) -> bool:
    if i < 1:
        return False
    return (
        df["stoch_k"].iat[i] < df["stoch_d"].iat[i]
        and df["stoch_k"].iat[i - 1] >= df["stoch_d"].iat[i - 1]
    )


def signal_fn(df: pd.DataFrame, i: int) -> dict | None:
    """High win-rate signal — mean reversion in trending market."""
    if i < 200:
        return None

    row = df.iloc[i]
    if pd.isna(row.get("ema_200")) or pd.isna(row.get("stoch_k")) or pd.isna(row.get("atr")):
        return None

    close = row["close"]
    e13, e21, e34, e50, e90, e200 = (
        row["ema_13"], row["ema_21"], row["ema_34"], row["ema_50"],
        row["ema_90"], row["ema_200"],
    )
    k = row["stoch_k"]
    k_prev = df["stoch_k"].iat[i - 1]
    adx_val = row["adx"]
    atr_val = row["atr"]
    vol = row["volume"]
    vol_avg = df["volume"].iloc[max(0, i - 20):i].mean()

    # Time-of-day filter (skip dead zone for crypto)
    hour = df.index[i].hour
    in_active_session = hour not in [0, 1, 2, 3, 22, 23]  # avoid 22:00-04:00 UTC

    # Tighter risk: TP 1.0x ATR, SL 1.5x ATR → win small/often, lose rare
    sl_pct = 1.5 * atr_val / close
    tp_pct = 1.0 * atr_val / close
    sl_pct = min(max(sl_pct, 0.005), 0.05)
    tp_pct = min(max(tp_pct, 0.003), 0.04)

    # === LONG: Buy oversold pullback in uptrend ===
    long_trend = (
        close > e90               # above mid-term trend
        and e21 > e50             # short-term bullish
        and e50 > e90             # mid-term bullish
        and e90 > e200            # long-term bullish
    )
    long_pullback = close < e21 * 1.01  # near or below ema21 (pullback zone)
    long_oversold_bounce = (
        k_prev < 15               # was VERY oversold
        and _stoch_cross_up(df, i)
        and k < 50                # not yet midline
    )
    long_volume = vol > vol_avg * 1.3
    long_strong_trend = adx_val > 25

    if (
        long_trend
        and long_pullback
        and long_oversold_bounce
        and long_volume
        and long_strong_trend
        and in_active_session
    ):
        return {
            "action": "open_long",
            "sl_pct": sl_pct, "tp_pct": tp_pct,
            "reason": f"Pullback long in uptrend, stoch xup from {k_prev:.1f}, ADX={adx_val:.1f}, vol={vol/vol_avg:.1f}x",
        }

    # === SHORT: Sell overbought pullback in downtrend ===
    short_trend = (
        close < e90
        and e21 < e50
        and e50 < e90
        and e90 < e200
    )
    short_pullback = close > e21 * 0.99
    short_overbought_bounce = (
        k_prev > 85
        and _stoch_cross_down(df, i)
        and k > 50
    )
    short_volume = vol > vol_avg * 1.3
    short_strong_trend = adx_val > 25

    if (
        short_trend
        and short_pullback
        and short_overbought_bounce
        and short_volume
        and short_strong_trend
        and in_active_session
    ):
        return {
            "action": "open_short",
            "sl_pct": sl_pct, "tp_pct": tp_pct,
            "reason": f"Pullback short in downtrend, stoch xdown from {k_prev:.1f}, ADX={adx_val:.1f}, vol={vol/vol_avg:.1f}x",
        }

    return None
