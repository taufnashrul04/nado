"""
V9 — AI Analyst Strategy with USER's exact method
Method (per user spec):
  - Stochastic Oscillator (5, 3, 3)
  - EMA stack: 13, 21, 34, 50, 90, 200

AI evaluates each candidate setup using PURE this method.
8h timeframe → hold 12-48h (1.5-6 bars).
"""
from __future__ import annotations
import json, os, time, hashlib
from pathlib import Path
import pandas as pd
from openai import OpenAI

CACHE_DIR = Path("/root/nado-bot/data/llm_cache_v9")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_client = None
_stats = {"candidates": 0, "llm_calls": 0, "cache_hits": 0,
          "long": 0, "short": 0, "skip": 0}


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL", "http://150.109.94.222:20128/v1"),
            timeout=45,
        )
    return _client


def _stoch_state(k: float, d: float, k_prev: float) -> str:
    """Classify current Stoch state."""
    if pd.isna(k) or pd.isna(d): return "unknown"
    region = "oversold" if k < 20 else ("overbought" if k > 80 else
              "lower" if k < 40 else "upper" if k > 60 else "mid")
    cross = "bullish_cross" if k > d and k_prev <= d else \
            "bearish_cross" if k < d and k_prev >= d else \
            "above_d" if k > d else "below_d"
    return f"{region}_{cross}"


def _ema_alignment(row) -> tuple[str, float]:
    """Return (label, score) for EMA stack alignment."""
    emas = [row["ema_13"], row["ema_21"], row["ema_34"],
            row["ema_50"], row["ema_90"], row["ema_200"]]
    if all(emas[i] > emas[i+1] for i in range(len(emas)-1)):
        return ("PERFECT_BULLISH", 1.0)
    if all(emas[i] < emas[i+1] for i in range(len(emas)-1)):
        return ("PERFECT_BEARISH", -1.0)
    if emas[0] > emas[1] > emas[2] and row["ema_50"] > row["ema_200"]:
        return ("PARTIAL_BULLISH", 0.5)
    if emas[0] < emas[1] < emas[2] and row["ema_50"] < row["ema_200"]:
        return ("PARTIAL_BEARISH", -0.5)
    return ("MIXED", 0.0)


def _is_candidate(df, i) -> bool:
    """Cheap pre-filter — only call AI when there's a real Stoch event."""
    if i < 200: return False
    row = df.iloc[i]
    if pd.isna(row.get("stoch_k")) or pd.isna(row.get("ema_200")):
        return False
    k = row["stoch_k"]
    d = row["stoch_d"]
    k_prev = df["stoch_k"].iat[i-1]

    # Trigger: any meaningful Stoch event
    bullish_cross = k > d and k_prev <= d and k < 50
    bearish_cross = k < d and k_prev >= d and k > 50
    deep_oversold = k < 20
    deep_overbought = k > 80
    return any([bullish_cross, bearish_cross, deep_oversold, deep_overbought])


def _build_prompt(df, i, symbol) -> str:
    """Build context for AI analyst — STRICTLY using user's method."""
    row = df.iloc[i]
    k = row["stoch_k"]
    d = row["stoch_d"]
    k_prev = df["stoch_k"].iat[i-1]
    stoch_state = _stoch_state(k, d, k_prev)
    align_label, align_score = _ema_alignment(row)

    # Last 8 bars summary (last 64h on 8h tf)
    start = max(0, i - 7)
    window = df.iloc[start:i+1]
    bars = []
    for ts, b in window.iterrows():
        bars.append(f"  {ts.strftime('%m-%d %H:%M')} | "
                   f"O={b['open']:.4g} H={b['high']:.4g} L={b['low']:.4g} C={b['close']:.4g} "
                   f"StochK={b['stoch_k']:.0f} StochD={b['stoch_d']:.0f}")
    bars_text = "\n".join(bars)

    return f"""You are a pure technical analyst. Use ONLY this method:
  - Stochastic Oscillator (5, 3, 3)
  - EMA stack: 13, 21, 34, 50, 90, 200

NO other indicators. NO news. NO fundamentals. Just price + these two tools.

Symbol: {symbol} (8h chart)
Last 8 bars (64 hours):
{bars_text}

Current bar state:
  Close: {row['close']:.4g}
  EMA(13): {row['ema_13']:.4g}
  EMA(21): {row['ema_21']:.4g}
  EMA(34): {row['ema_34']:.4g}
  EMA(50): {row['ema_50']:.4g}
  EMA(90): {row['ema_90']:.4g}
  EMA(200): {row['ema_200']:.4g}
  Stoch %K: {k:.1f}  %D: {d:.1f}  (prev %K: {k_prev:.1f})
  EMA alignment: {align_label}
  Stoch state: {stoch_state}

Rules of engagement (your only framework):
  1. LONG only if EMA stack is PERFECT_BULLISH or PARTIAL_BULLISH
     AND Stoch %K crosses up from oversold (<20) OR rises from below %D in lower zone (<40).
     Pullback to EMA21 is bonus.
  2. SHORT only if EMA stack is PERFECT_BEARISH or PARTIAL_BEARISH
     AND Stoch %K crosses down from overbought (>80) OR falls from above %D in upper zone (>60).
  3. SKIP if EMA alignment is MIXED, or if Stoch is mid-zone with no momentum shift.
  4. The closer the EMA stack to PERFECT alignment, the higher the confidence.

Hold horizon: 1-2 days (12-48 hours, 1.5-6 bars on 8h chart).
Risk: SL ~1.3x ATR, TP ~2.0x ATR.

Respond with EXACTLY this JSON, nothing else:
{{
  "action": "long" | "short" | "skip",
  "confidence": 0.0-1.0,
  "reason": "concise reason citing Stoch + EMA only (max 25 words)"
}}"""


def _llm_decide(prompt: str) -> dict:
    h = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{h}.json"
    if cache_file.exists():
        _stats["cache_hits"] += 1
        return json.loads(cache_file.read_text())

    _stats["llm_calls"] += 1
    cli = _get_client()
    model = os.environ.get("LLM_MODEL", "kr/claude-opus-4.7")
    try:
        resp = cli.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        content = resp.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        s = content.find("{")
        e = content.rfind("}")
        if s != -1 and e != -1:
            content = content[s:e+1]
        decision = json.loads(content)
    except Exception as ex:
        decision = {"action": "skip", "confidence": 0.0, "reason": f"err: {str(ex)[:60]}"}

    cache_file.write_text(json.dumps(decision))
    return decision


def signal_fn(df: pd.DataFrame, i: int) -> dict | None:
    if not _is_candidate(df, i):
        return None
    _stats["candidates"] += 1

    symbol = getattr(df, "attrs", {}).get("symbol", "BTC/USDT")
    prompt = _build_prompt(df, i, symbol)
    decision = _llm_decide(prompt)

    action = decision.get("action", "skip").lower()
    confidence = float(decision.get("confidence", 0.0))
    reason = decision.get("reason", "")

    if action == "skip" or confidence < 0.6:
        _stats["skip"] += 1
        return None

    row = df.iloc[i]
    atr = row.get("atr")
    close = row["close"]
    if pd.isna(atr) or atr <= 0:
        _stats["skip"] += 1
        return None

    sl_pct = max(1.3 * atr / close, 0.008)
    tp_pct = max(2.0 * atr / close, 0.012)

    if action == "long":
        _stats["long"] += 1
        return {
            "action": "open_long",
            "sl_pct": sl_pct, "tp_pct": tp_pct, "max_bars_hold": 6,
            "reason": f"V9_AI_long {reason} (conf={confidence:.2f})",
        }
    if action == "short":
        _stats["short"] += 1
        return {
            "action": "open_short",
            "sl_pct": sl_pct, "tp_pct": tp_pct, "max_bars_hold": 6,
            "reason": f"V9_AI_short {reason} (conf={confidence:.2f})",
        }
    _stats["skip"] += 1
    return None


def get_stats(): return dict(_stats)
def reset_stats():
    for k in _stats: _stats[k] = 0
