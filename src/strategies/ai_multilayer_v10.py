"""
V10 — Multi-Layer AI Analyst Strategy
=====================================
Layers:
  1. TECHNICAL (Stoch + EMA + RSI + MACD + ADX) — user's method extended
  2. MULTI-TIMEFRAME (1h, 4h, 1d trend regime) — alignment check
  3. TRADINGVIEW CONSENSUS — retail sentiment via tradingview-ta
  4. LLM ANALYST — synthesize all layers, give final verdict

Confluence: ALL 4 layers must agree before trade
Goal: Higher conviction setups, fewer but better trades for HIGH LEVERAGE
"""
from __future__ import annotations
import json, os, time, hashlib
from pathlib import Path
import pandas as pd
from openai import OpenAI

CACHE_DIR = Path("/root/nado-bot/data/llm_cache_v10")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_client = None
_stats = {"candidates": 0, "llm_calls": 0, "cache_hits": 0,
          "tv_calls": 0, "tv_errors": 0, "long": 0, "short": 0, "skip": 0,
          "veto_tv": 0, "veto_mtf": 0, "veto_ai": 0}

# Symbol mapping for TradingView
TV_SYMBOL_MAP = {
    "BTC-PERP": ("BTCUSDT", "crypto", "BINANCE"),
    "ETH-PERP": ("ETHUSDT", "crypto", "BINANCE"),
    "SOL-PERP": ("SOLUSDT", "crypto", "BINANCE"),
    "BNB-PERP": ("BNBUSDT", "crypto", "BINANCE"),
    "DOGE-PERP": ("DOGEUSDT", "crypto", "BINANCE"),
    "SUI-PERP": ("SUIUSDT", "crypto", "BINANCE"),
    "HYPE-PERP": ("HYPEUSDT", "crypto", "BINANCE"),
    "AVAX-PERP": ("AVAXUSDT", "crypto", "BINANCE"),
    "XRP-PERP": ("XRPUSDT", "crypto", "BINANCE"),
}


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL", "http://150.109.94.222:20128/v1"),
            timeout=45,
        )
    return _client


def _ema_alignment(row) -> tuple[str, float]:
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
    """Pre-filter — only call AI when there's a real Stoch event."""
    if i < 200:
        return False
    row = df.iloc[i]
    if pd.isna(row.get("stoch_k")) or pd.isna(row.get("ema_200")):
        return False
    k = row["stoch_k"]
    d = row["stoch_d"]
    k_prev = df["stoch_k"].iat[i-1]

    bullish_cross = k > d and k_prev <= d and k < 50
    bearish_cross = k < d and k_prev >= d and k > 50
    deep_oversold = k < 20
    deep_overbought = k > 80
    return any([bullish_cross, bearish_cross, deep_oversold, deep_overbought])


def _get_tv_consensus(symbol: str, interval: str = "4h") -> dict | None:
    """Fetch TradingView technical consensus. Cached."""
    cache_key = hashlib.sha256(f"tv_{symbol}_{interval}_{int(time.time()/3600)}".encode()).hexdigest()[:12]
    cache_file = CACHE_DIR / f"tv_{cache_key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    if symbol not in TV_SYMBOL_MAP:
        return None
    tv_sym, screener, exchange = TV_SYMBOL_MAP[symbol]

    try:
        from tradingview_ta import TA_Handler, Interval
        interval_map = {
            "1h": Interval.INTERVAL_1_HOUR,
            "4h": Interval.INTERVAL_4_HOURS,
            "1d": Interval.INTERVAL_1_DAY,
        }
        h = TA_Handler(symbol=tv_sym, screener=screener,
                      exchange=exchange, interval=interval_map.get(interval, Interval.INTERVAL_4_HOURS))
        a = h.get_analysis()
        result = {
            "rec": a.summary.get("RECOMMENDATION", "NEUTRAL"),
            "buy": a.summary.get("BUY", 0),
            "sell": a.summary.get("SELL", 0),
            "neutral": a.summary.get("NEUTRAL", 0),
            "osc": a.oscillators.get("RECOMMENDATION", "NEUTRAL"),
            "ma": a.moving_averages.get("RECOMMENDATION", "NEUTRAL"),
            "rsi": a.indicators.get("RSI", None),
            "macd_hist": a.indicators.get("MACD.macd", None),
            "ts": int(time.time()),
        }
        _stats["tv_calls"] += 1
        cache_file.write_text(json.dumps(result))
        return result
    except Exception as e:
        _stats["tv_errors"] += 1
        return None


def _check_mtf_alignment(df, i, side) -> dict:
    """Multi-timeframe alignment using current 4h df + simulated higher TF."""
    row = df.iloc[i]
    align_label, align_score = _ema_alignment(row)

    # Simulate 1d trend: average price last 30 bars (~5 days @ 4h) vs 60 bars (~10 days)
    sma_30 = df["close"].iloc[max(0,i-30):i+1].mean()
    sma_60 = df["close"].iloc[max(0,i-60):i+1].mean()
    daily_trend = "bull" if sma_30 > sma_60 * 1.01 else "bear" if sma_30 < sma_60 * 0.99 else "flat"

    # ADX strength
    adx = row.get("adx", 0)
    trend_strong = adx > 20 if not pd.isna(adx) else False

    aligned = False
    if side == "long":
        aligned = ("BULLISH" in align_label) and (daily_trend in ["bull", "flat"]) and trend_strong
    elif side == "short":
        aligned = ("BEARISH" in align_label) and (daily_trend in ["bear", "flat"]) and trend_strong

    return {
        "ema_4h": align_label,
        "daily_trend": daily_trend,
        "adx": float(adx) if not pd.isna(adx) else 0,
        "trend_strong": trend_strong,
        "aligned": aligned,
    }


def _check_tv_consensus_alignment(tv_data: dict, side: str) -> bool:
    """TV consensus must agree with our side direction."""
    if tv_data is None:
        return True  # don't veto if TV unavailable
    rec = tv_data.get("rec", "NEUTRAL")
    if side == "long":
        return rec in ["BUY", "STRONG_BUY"] or (rec == "NEUTRAL" and tv_data.get("buy", 0) > tv_data.get("sell", 0))
    elif side == "short":
        return rec in ["SELL", "STRONG_SELL"] or (rec == "NEUTRAL" and tv_data.get("sell", 0) > tv_data.get("buy", 0))
    return False


def _build_prompt(df, i, symbol, mtf_data, tv_data) -> str:
    """Build comprehensive prompt with ALL layers."""
    row = df.iloc[i]
    align_label, _ = _ema_alignment(row)

    # Last 5 bars
    start = max(0, i - 4)
    bars = []
    for ts, b in df.iloc[start:i+1].iterrows():
        bars.append(f"  {ts.strftime('%m-%d %H:%M')} | C={b['close']:.4g} StK={b['stoch_k']:.0f} StD={b['stoch_d']:.0f} RSI={b.get('rsi',50):.0f}")

    tv_summary = "N/A"
    if tv_data:
        tv_summary = (f"REC={tv_data['rec']} (BUY={tv_data['buy']}/SELL={tv_data['sell']}/NEUTRAL={tv_data['neutral']}), "
                     f"Oscillators={tv_data['osc']}, MA={tv_data['ma']}")
        if tv_data.get('rsi'):
            tv_summary += f", TV_RSI={tv_data['rsi']:.0f}"

    return f"""You are a multi-layer technical analyst combining 4 layers of evidence.
HIGH LEVERAGE TRADING (50x) — only A+ setups, mediocre = SKIP.

Symbol: {symbol} (4h chart)
Last 5 bars:
{chr(10).join(bars)}

LAYER 1 — Technical (Stoch 5,3,3 + EMA 13/21/34/50/90/200):
  Close: {row['close']:.4g}  EMA21: {row['ema_21']:.4g}  EMA50: {row['ema_50']:.4g}  EMA200: {row['ema_200']:.4g}
  Stoch %K: {row['stoch_k']:.1f}  %D: {row['stoch_d']:.1f}
  EMA alignment: {align_label}
  RSI: {row.get('rsi', 50):.0f}  ATR: {row.get('atr', 0):.4g}

LAYER 2 — Multi-Timeframe:
  4h EMA stack: {mtf_data['ema_4h']}
  Daily trend (30d vs 60d): {mtf_data['daily_trend']}
  ADX: {mtf_data['adx']:.1f}  (trend_strong={mtf_data['trend_strong']})

LAYER 3 — TradingView Consensus (1100+ retail traders):
  {tv_summary}

LAYER 4 — Decision:
RULES (strict, no compromise):
  1. LONG only if: EMA bullish + Daily trend bull + ADX>20 + TV consensus BUY/NEUTRAL_BUY + Stoch oversold cross
  2. SHORT only if: EMA bearish + Daily trend bear + ADX>20 + TV consensus SELL/NEUTRAL_SELL + Stoch overbought cross
  3. SKIP if ANY layer disagrees, or layers neutral/mixed
  4. High leverage = high stakes — bias to SKIP unless clearly all-aligned

Hold horizon: 12-48 hours (3-12 bars on 4h).

Respond with EXACTLY this JSON:
{{
  "action": "long" | "short" | "skip",
  "confidence": 0.0-1.0,
  "layers_agree": 0-4,
  "reason": "concise reason citing ALL 4 layers (max 30 words)"
}}"""


def _llm_decide(prompt: str) -> dict:
    h = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"v10_{h}.json"
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
            max_tokens=250,
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
        decision = {"action": "skip", "confidence": 0.0, "layers_agree": 0, "reason": f"err: {str(ex)[:60]}"}

    cache_file.write_text(json.dumps(decision))
    return decision


def signal_fn(df: pd.DataFrame, i: int) -> dict | None:
    if not _is_candidate(df, i):
        return None
    _stats["candidates"] += 1

    symbol = getattr(df, "attrs", {}).get("symbol", "BTC/USDT")
    row = df.iloc[i]

    # Determine candidate side from Stoch state
    k = row["stoch_k"]
    d = row["stoch_d"]
    k_prev = df["stoch_k"].iat[i-1]
    if k > d and k_prev <= d and k < 50:
        candidate_side = "long"
    elif k < d and k_prev >= d and k > 50:
        candidate_side = "short"
    elif k < 20:
        candidate_side = "long"
    elif k > 80:
        candidate_side = "short"
    else:
        return None

    # LAYER 2: MTF check
    mtf_data = _check_mtf_alignment(df, i, candidate_side)
    if not mtf_data["aligned"]:
        _stats["veto_mtf"] += 1
        return None

    # LAYER 3: TV consensus
    tv_data = _get_tv_consensus(symbol, "4h")
    if not _check_tv_consensus_alignment(tv_data, candidate_side):
        _stats["veto_tv"] += 1
        return None

    # LAYER 4: AI synthesis
    prompt = _build_prompt(df, i, symbol, mtf_data, tv_data)
    decision = _llm_decide(prompt)

    action = decision.get("action", "skip").lower()
    confidence = float(decision.get("confidence", 0.0))
    layers = int(decision.get("layers_agree", 0))
    reason = decision.get("reason", "")[:140]

    if action == "skip" or confidence < 0.65 or layers < 3:
        _stats["veto_ai"] += 1
        _stats["skip"] += 1
        return None

    if action != candidate_side:
        _stats["skip"] += 1
        return None

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
            "sl_pct": sl_pct, "tp_pct": tp_pct, "max_bars_hold": 12,
            "reason": f"V10_long L={layers}/4 conf={confidence:.2f} {reason}",
        }
    if action == "short":
        _stats["short"] += 1
        return {
            "action": "open_short",
            "sl_pct": sl_pct, "tp_pct": tp_pct, "max_bars_hold": 12,
            "reason": f"V10_short L={layers}/4 conf={confidence:.2f} {reason}",
        }
    return None


def get_stats(): return dict(_stats)
def reset_stats():
    for k in _stats: _stats[k] = 0
