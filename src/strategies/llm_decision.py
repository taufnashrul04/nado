"""
Strategy V3 — LLM-Driven Decision Maker

Approach:
  1. Cheap rule-based pre-filter checks for "candidate setups"
     (Stoch cross OR EMA alignment shift OR pullback to key EMA)
  2. When candidate triggers, build a context snapshot of the last 30 bars
     plus current indicator state (per user spec: Stoch 5,3,3 + EMA 13/21/34/50/90/200)
  3. Send to LLM → returns LONG / SHORT / SKIP + reasoning
  4. SL/TP fixed via ATR (1.5x SL, 2.0x TP — adjust if WR target needs RR shift)

This keeps LLM calls economical (~5-15% of bars trigger pre-filter).
"""
from __future__ import annotations
import json
import os
import time
import hashlib
import logging
from pathlib import Path
import pandas as pd
from openai import OpenAI

logger = logging.getLogger(__name__)

CACHE_DIR = Path("/root/nado-bot/data/llm_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_client = None
def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL", "http://150.109.94.222:20128/v1"),
            timeout=30,
        )
    return _client


def _stoch_cross_up(df: pd.DataFrame, i: int) -> bool:
    if i < 1: return False
    return df["stoch_k"].iat[i] > df["stoch_d"].iat[i] and df["stoch_k"].iat[i-1] <= df["stoch_d"].iat[i-1]


def _stoch_cross_down(df: pd.DataFrame, i: int) -> bool:
    if i < 1: return False
    return df["stoch_k"].iat[i] < df["stoch_d"].iat[i] and df["stoch_k"].iat[i-1] >= df["stoch_d"].iat[i-1]


def _ema_alignment_score(row) -> float:
    """+1 if all EMAs perfectly bullish stacked, -1 if perfectly bearish, 0 mixed."""
    emas = [row["ema_13"], row["ema_21"], row["ema_34"], row["ema_50"], row["ema_90"], row["ema_200"]]
    if all(emas[i] > emas[i+1] for i in range(len(emas)-1)):
        return 1.0
    if all(emas[i] < emas[i+1] for i in range(len(emas)-1)):
        return -1.0
    # Partial alignment (top 3)
    top3_bull = emas[0] > emas[1] > emas[2]
    top3_bear = emas[0] < emas[1] < emas[2]
    if top3_bull: return 0.5
    if top3_bear: return -0.5
    return 0.0


def _is_candidate_setup(df: pd.DataFrame, i: int) -> str | None:
    """Returns setup type if candidate, else None. Cheap rule check."""
    if i < 200:
        return None
    row = df.iloc[i]
    if pd.isna(row.get("ema_200")) or pd.isna(row.get("stoch_k")) or pd.isna(row.get("atr")):
        return None

    stoch_k = row["stoch_k"]
    stoch_k_prev = df["stoch_k"].iat[i-1]
    adx_val = row["adx"]
    align = _ema_alignment_score(row)
    close = row["close"]
    ema21 = row["ema_21"]
    ema_dist = abs(close - ema21) / close

    # Setup 1: Stoch oversold cross with EMA support nearby (relaxed)
    if _stoch_cross_up(df, i) and stoch_k_prev < 30 and align >= 0.5 and ema_dist < 0.02:
        return "long_pullback_oversold"
    # Setup 2: Stoch overbought cross with EMA resistance nearby (relaxed)
    if _stoch_cross_down(df, i) and stoch_k_prev > 70 and align <= -0.5 and ema_dist < 0.02:
        return "short_rally_overbought"
    # Setup 3: Stoch deep oversold + strong trend down (potential bounce in continuation)
    if stoch_k < 20 and stoch_k > stoch_k_prev and align <= -0.5 and adx_val > 25:
        return "short_continuation_setup"
    # Setup 4: Stoch deep overbought + strong trend up
    if stoch_k > 80 and stoch_k < stoch_k_prev and align >= 0.5 and adx_val > 25:
        return "long_continuation_setup"

    return None


def _build_context(df: pd.DataFrame, i: int, setup: str, symbol: str) -> str:
    """Build the LLM prompt with last 30 bars + indicators."""
    start = max(0, i - 29)
    window = df.iloc[start:i+1]
    bars_lines = []
    for ts, b in window.iterrows():
        bars_lines.append(
            f"  {ts.strftime('%Y-%m-%d %H:%M')} | "
            f"O={b['open']:.4g} H={b['high']:.4g} L={b['low']:.4g} C={b['close']:.4g} V={b['volume']:.2g}"
        )
    bars_text = "\n".join(bars_lines[-15:])  # last 15 bars to save tokens

    row = df.iloc[i]
    indicators = (
        f"Current bar indicators:\n"
        f"  Close: {row['close']:.4g}\n"
        f"  EMA(13/21/34/50/90/200): "
        f"{row['ema_13']:.4g} / {row['ema_21']:.4g} / {row['ema_34']:.4g} / "
        f"{row['ema_50']:.4g} / {row['ema_90']:.4g} / {row['ema_200']:.4g}\n"
        f"  Stoch %K / %D: {row['stoch_k']:.1f} / {row['stoch_d']:.1f}\n"
        f"  ATR(14): {row['atr']:.4g}  ADX(14): {row['adx']:.1f}  RSI(14): {row['rsi']:.1f}\n"
    )

    return f"""You are a probabilistic scalp trader. The pre-filter already validated this is a real setup — your job is to filter out the WORST 30%, not require perfection.

Take roughly 50-60% of setups. SKIP only when something genuinely contradicts the setup (e.g. clear trend reversal, news spike, extreme exhaustion).

Symbol: {symbol}
Pre-filter setup: {setup}
  - "long_pullback_oversold" = bullish EMA stack, Stoch crossed up from oversold, near EMA21 pullback
  - "short_rally_overbought" = bearish EMA stack, Stoch crossed down from overbought, near EMA21 pullback
  - "long_continuation_setup" = bullish EMA, deep overbought rolling over (counter-trend warning, but trend strong)
  - "short_continuation_setup" = bearish EMA, deep oversold lifting (counter-trend warning)

Last 15 bars (1H):
{bars_text}

{indicators}

Decision logic:
  - If setup says LONG (long_pullback_oversold or long_continuation_setup): default to LONG unless price is breaking down hard
  - If setup says SHORT (short_rally_overbought or short_continuation_setup): default to SHORT unless price is breaking up hard
  - SKIP only if the last 3 bars contradict the setup direction OR ADX is collapsing fast

Confidence guide:
  - 0.7-0.9 = setup looks clean, all indicators aligned
  - 0.6-0.7 = setup is decent, minor noise
  - 0.5-0.6 = marginal, but if you don't see real contradiction, still take it
  - <0.5 = real concerns (only then skip)

Respond with EXACTLY this JSON (nothing else):
{{
  "action": "long" | "short" | "skip",
  "confidence": 0.0-1.0,
  "reason": "short reasoning (max 20 words)"
}}"""


def _llm_decide(prompt: str, model: str | None = None) -> dict:
    """Call LLM, return decision dict. Cached on prompt hash."""
    h = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{h}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    model = model or os.environ.get("LLM_MODEL", "kr/claude-opus-4.7")
    cli = _get_client()
    try:
        resp = cli.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        content = resp.choices[0].message.content.strip()
        # Extract JSON (model may wrap in ``` or add extra text)
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        # Find first { ... }
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            content = content[start:end+1]
        decision = json.loads(content)
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        decision = {"action": "skip", "confidence": 0.0, "reason": f"llm_error: {e}"}

    cache_file.write_text(json.dumps(decision))
    return decision


# Stats
_stats = {"candidates": 0, "llm_calls": 0, "long": 0, "short": 0, "skip": 0}


def signal_fn(df: pd.DataFrame, i: int) -> dict | None:
    """Hybrid signal: cheap pre-filter → LLM confirmation."""
    setup = _is_candidate_setup(df, i)
    if not setup:
        return None

    _stats["candidates"] += 1

    # Fetch symbol from df attrs (set by runner) or fallback
    symbol = getattr(df, "attrs", {}).get("symbol", "UNKNOWN")
    prompt = _build_context(df, i, setup, symbol)
    decision = _llm_decide(prompt)
    _stats["llm_calls"] += 1

    action = decision.get("action", "skip").lower()
    confidence = float(decision.get("confidence", 0.0))
    reason = decision.get("reason", "")

    if action == "skip" or confidence < 0.5:
        _stats["skip"] += 1
        return None

    row = df.iloc[i]
    atr_val = row["atr"]
    close = row["close"]
    sl_pct = max(min(1.5 * atr_val / close, 0.05), 0.005)
    tp_pct = max(min(2.0 * atr_val / close, 0.10), 0.005)

    if action == "long":
        _stats["long"] += 1
        return {
            "action": "open_long",
            "sl_pct": sl_pct, "tp_pct": tp_pct,
            "reason": f"[LLM/{setup}] {reason} (conf={confidence:.2f})",
        }
    if action == "short":
        _stats["short"] += 1
        return {
            "action": "open_short",
            "sl_pct": sl_pct, "tp_pct": tp_pct,
            "reason": f"[LLM/{setup}] {reason} (conf={confidence:.2f})",
        }
    _stats["skip"] += 1
    return None


def get_stats() -> dict:
    return dict(_stats)


def reset_stats():
    for k in _stats:
        _stats[k] = 0
