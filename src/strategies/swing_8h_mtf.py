"""
V8 — 8H Intraday Swing with Daily Trend Filter
Hold: max 6 bars × 8h = 48h (2 days max)
Approach: 2-layer filter
  - Daily: regime check (EMA aligned + ADX > 20)
  - 8H: entry confluence (Stoch + RSI + EMA21 pullback + volume)

Goal: WR ≥ 55%, PF ≥ 1.5
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


def _stoch_cross_up(df, i, threshold=35) -> bool:
    if i < 1: return False
    k_prev = df["stoch_k"].iat[i-1]
    k_now = df["stoch_k"].iat[i]
    d_now = df["stoch_d"].iat[i]
    return k_prev < threshold and k_now > k_prev and k_now > d_now


def _stoch_cross_down(df, i, threshold=65) -> bool:
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


def _daily_trend_ok(df_daily: pd.DataFrame, ts_8h: pd.Timestamp, side: str) -> bool:
    """Get the daily bar BEFORE ts_8h, check trend regime."""
    # Find the most recent daily bar AT OR BEFORE this 8h ts
    daily_idx = df_daily.index[df_daily.index <= ts_8h]
    if len(daily_idx) < 200:
        return False
    last_daily = df_daily.loc[daily_idx[-1]]
    if pd.isna(last_daily.get("ema_200")) or pd.isna(last_daily.get("adx")):
        return False
    adx_d = last_daily["adx"]
    if adx_d < 18:  # daily must have direction
        return False
    if side == "long":
        return _ema_aligned_bull(last_daily)
    else:
        return _ema_aligned_bear(last_daily)


def make_signal_fn_with_daily(df_daily: pd.DataFrame):
    """Returns signal_fn closured with daily reference frame."""
    def signal_fn(df, i):
        if i < 200: return None
        row = df.iloc[i]
        if pd.isna(row.get("ema_200")) or pd.isna(row.get("atr")) or pd.isna(row.get("rsi")):
            return None

        atr_val = row["atr"]
        close = row["close"]
        sl_pct = max(1.3 * atr_val / close, 0.008)
        tp_pct = max(2.0 * atr_val / close, 0.012)
        ts = df.index[i]

        # ===== LONG =====
        if _ema_aligned_bull(row):
            if not _daily_trend_ok(df_daily, ts, "long"):
                return None
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
                    "sl_pct": sl_pct, "tp_pct": tp_pct, "max_bars_hold": 6,
                    "reason": f"V8_long conf={sum(signals)}/5 RSI={row['rsi']:.0f}",
                }

        # ===== SHORT =====
        if _ema_aligned_bear(row):
            if not _daily_trend_ok(df_daily, ts, "short"):
                return None
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
                    "sl_pct": sl_pct, "tp_pct": tp_pct, "max_bars_hold": 6,
                    "reason": f"V8_short conf={sum(signals)}/5 RSI={row['rsi']:.0f}",
                }

        return None

    return signal_fn
