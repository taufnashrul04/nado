"""
Strategy: Stoch (5,3,3) + EMA stack (13,21,34,50,90,200)

User specification:
  - Stochastic: 5, 3, 3
  - EMA: 13, 21, 34, 50, 90, 200

Signal logic (multi-confluence):
  LONG entry:
    - EMA stack bullish: ema_13 > ema_21 > ema_34 > ema_50
    - Price above ema_90 and ema_200 (long-term trend filter)
    - Stoch %K crosses up %D from below 30 (oversold momentum reversal)
    - Stoch %K < 80 (not yet overbought)
    - ADX > 20 (trending market)

  SHORT entry:
    - EMA stack bearish: ema_13 < ema_21 < ema_34 < ema_50
    - Price below ema_90 and ema_200
    - Stoch %K crosses down %D from above 70
    - Stoch %K > 20
    - ADX > 20

Risk:
  - SL: 1.5x ATR away
  - TP: 3.0x ATR away (2:1 R:R)
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
    """Returns signal dict or None."""
    if i < 200:  # need full EMA history
        return None

    row = df.iloc[i]
    if pd.isna(row.get("ema_200")) or pd.isna(row.get("stoch_k")) or pd.isna(row.get("atr")):
        return None

    close = row["close"]
    e13, e21, e34, e50, e90, e200 = (
        row["ema_13"], row["ema_21"], row["ema_34"], row["ema_50"],
        row["ema_90"], row["ema_200"],
    )
    k, d = row["stoch_k"], row["stoch_d"]
    adx_val = row["adx"]
    atr_val = row["atr"]

    # ATR-based SL/TP (in % of price)
    sl_pct = 1.5 * atr_val / close
    tp_pct = 3.0 * atr_val / close
    # Cap min/max to prevent absurd values
    sl_pct = min(max(sl_pct, 0.005), 0.05)   # 0.5% to 5%
    tp_pct = min(max(tp_pct, 0.01), 0.10)    # 1% to 10%

    # LONG conditions
    ema_bull = e13 > e21 > e34 > e50
    above_long = close > e90 and close > e200
    stoch_up = _stoch_cross_up(df, i) and k < 80 and df["stoch_k"].iat[i - 1] < 30
    if ema_bull and above_long and stoch_up and adx_val > 20:
        return {
            "action": "open_long",
            "sl_pct": sl_pct, "tp_pct": tp_pct,
            "reason": f"EMA stack bull, stoch xup from {df['stoch_k'].iat[i-1]:.1f}, ADX={adx_val:.1f}",
        }

    # SHORT conditions
    ema_bear = e13 < e21 < e34 < e50
    below_long = close < e90 and close < e200
    stoch_down = _stoch_cross_down(df, i) and k > 20 and df["stoch_k"].iat[i - 1] > 70
    if ema_bear and below_long and stoch_down and adx_val > 20:
        return {
            "action": "open_short",
            "sl_pct": sl_pct, "tp_pct": tp_pct,
            "reason": f"EMA stack bear, stoch xdown from {df['stoch_k'].iat[i-1]:.1f}, ADX={adx_val:.1f}",
        }

    return None
