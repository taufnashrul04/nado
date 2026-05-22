"""
Run backtest across all configured symbols.
Outputs: per-symbol metrics + LLM analyst commentary.
"""
from __future__ import annotations
import sys
import os
import logging
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.fetcher import NadoDataFetcher
from data.indicators import add_indicators
from backtest.engine import Backtester
from strategies.stoch_ema import signal_fn as v1_signal
from strategies.stoch_ema_v2 import signal_fn as v2_signal
from strategies.llm_decision import signal_fn as v3_signal, get_stats as v3_stats, reset_stats as v3_reset

import pandas as pd
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

console = Console()

# User-requested symbols (with boost weighting on TradFi)
SYMBOLS_FULL = {
    # Crypto majors (base weight)
    "BTC-PERP": {"weight": 1.0, "category": "crypto"},
    "ETH-PERP": {"weight": 1.0, "category": "crypto"},
    "SOL-PERP": {"weight": 1.0, "category": "crypto"},
    # TradFi / Equities (boost — user wants more allocation here)
    "WTI-PERP": {"weight": 2.0, "category": "boost"},
    "SPY-PERP": {"weight": 2.0, "category": "boost"},
    "QQQ-PERP": {"weight": 2.0, "category": "boost"},
    "NVDA-PERP": {"weight": 2.0, "category": "boost"},
    "TSLA-PERP": {"weight": 2.0, "category": "boost"},
    "AAPL-PERP": {"weight": 2.0, "category": "boost"},
    "GOOGL-PERP": {"weight": 2.0, "category": "boost"},
}

# For LLM strategy (v3) — keep small to control cost
SYMBOLS_LLM = {
    "BTC-PERP": SYMBOLS_FULL["BTC-PERP"],
    "ETH-PERP": SYMBOLS_FULL["ETH-PERP"],
    "SPY-PERP": SYMBOLS_FULL["SPY-PERP"],
    "NVDA-PERP": SYMBOLS_FULL["NVDA-PERP"],
    "AAPL-PERP": SYMBOLS_FULL["AAPL-PERP"],
}

SYMBOLS = SYMBOLS_LLM if os.environ.get("STRATEGY", "v2") == "v3" else SYMBOLS_FULL

TIMEFRAME = "1h"
HISTORY_CANDLES = int(os.environ.get("HISTORY_CANDLES", "1500"))
INITIAL_BALANCE = 130.0
SIZE_PCT = 0.05
LEVERAGE = 3.0
STRATEGY_NAME = os.environ.get("STRATEGY", "v2")

SIGNAL_FNS = {"v1": v1_signal, "v2": v2_signal, "v3": v3_signal}


def run_one(symbol: str, fetcher: NadoDataFetcher, signal_fn, strategy_label: str) -> dict:
    console.print(f"\n[cyan]┌─ {symbol} [{strategy_label}] ─[/cyan]")
    console.print(f"[dim]│ Fetching {HISTORY_CANDLES} {TIMEFRAME} candles...[/dim]")

    df = fetcher.fetch_history(symbol, TIMEFRAME, HISTORY_CANDLES)
    if df.empty or len(df) < 250:
        console.print(f"[red]│ ❌ Insufficient data ({len(df)} candles)[/red]")
        return {"symbol": symbol, "strategy": strategy_label, "error": "insufficient_data"}

    console.print(f"[dim]│ Got {len(df)} candles ({df.index[0]} → {df.index[-1]})[/dim]")
    console.print(f"[dim]│ Computing indicators...[/dim]")
    df = add_indicators(df)

    bt = Backtester(
        symbol=symbol, timeframe=TIMEFRAME,
        initial_balance=INITIAL_BALANCE,
        size_per_trade_pct=SIZE_PCT,
        leverage=LEVERAGE,
    )
    console.print(f"[dim]│ Running backtest with {strategy_label}...[/dim]")
    result = bt.run(df, signal_fn)
    summary = result.summary()
    summary["strategy"] = strategy_label
    color = "green" if result.total_return_pct > 0 else "red"
    console.print(
        f"[{color}]│ ✅ {result.num_trades} trades | "
        f"Return: {result.total_return_pct:+.2f}% | "
        f"WR: {result.win_rate:.1f}%[/{color}]"
    )
    return {"symbol": symbol, "strategy": strategy_label, "result": result, "summary": summary}


def print_results_table(rows: list[dict]):
    table = Table(title=f"\nBacktest Summary — Strategy {STRATEGY_NAME.upper()}", show_lines=True)
    cols = ["Symbol", "Strat", "Trades", "WR%", "Return%", "MaxDD%", "Sharpe", "PF", "EndBal$"]
    for c in cols:
        table.add_column(c, justify="right" if c != "Symbol" else "left")

    rows_sorted = sorted(
        [r for r in rows if "summary" in r],
        key=lambda r: r["summary"]["win_rate"], reverse=True,
    )

    for r in rows_sorted:
        s = r["summary"]
        ret = s["total_return_pct"]
        wr = s["win_rate"]
        ret_color = "green" if ret > 0 else "red" if ret < 0 else "white"
        wr_color = "green" if wr >= 70 else "yellow" if wr >= 50 else "red"
        table.add_row(
            s["symbol"],
            s["strategy"],
            str(s["num_trades"]),
            f"[{wr_color}]{wr:.1f}[/{wr_color}]",
            f"[{ret_color}]{ret:+.2f}[/{ret_color}]",
            f"{s['max_drawdown_pct']:.2f}",
            f"{s['sharpe']:.2f}",
            str(s["profit_factor"]),
            f"{s['end_balance']:.2f}",
        )
    console.print(table)


def llm_analysis(rows: list[dict]) -> str:
    """Use the configured LLM provider to comment on backtest results."""
    try:
        from openai import OpenAI
    except ImportError:
        return "LLM analysis skipped (openai package not installed)"

    summaries = [r["summary"] for r in rows if "summary" in r]
    if not summaries:
        return "No results to analyze"

    table_text = "\n".join([
        f"- {s['symbol']}: {s['num_trades']} trades, win_rate={s['win_rate']:.1f}%, "
        f"return={s['total_return_pct']:+.2f}%, max_dd={s['max_drawdown_pct']:.2f}%, "
        f"sharpe={s['sharpe']:.2f}, pf={s['profit_factor']}"
        for s in summaries
    ])

    prompt = f"""You are a quantitative trader analyzing backtest results from a custom strategy:

Strategy: Stochastic Oscillator (5,3,3) + EMA stack (13, 21, 34, 50, 90, 200)
Logic:
  - LONG when EMA stack is bullish (13>21>34>50), price above 90 & 200 EMA, stoch %K crosses %D from below 30, ADX > 20
  - SHORT when EMA stack is bearish, price below long EMAs, stoch %K crosses %D from above 70
  - SL: 1.5x ATR | TP: 3.0x ATR (2:1 R:R)
  - Initial balance: $130 | Size per trade: 5% | Leverage: 3x

Backtest period: ~62 days of 1H candles per symbol (sample: ~1500 bars)

Results per symbol:
{table_text}

Provide concise analysis in INDONESIAN (informal, langsung ke poin):
1. Strategi ini cocok untuk symbol mana? (best 3 performers)
2. Symbol mana yang harus dihindari? (worst performers)
3. Apakah strategi ini layak dipakai live? Jelaskan kenapa.
4. Saran improvement jika ada.

Max 250 words. Pakai bullet points. Jujur, kalau jelek bilang jelek."""

    api_key = os.environ.get("OPENAI_API_KEY", "sk-dummy")
    base_url = os.environ.get("OPENAI_BASE_URL", "http://150.109.94.222:20128/v1")
    model = os.environ.get("LLM_MODEL", "kr/claude-opus-4.7")

    try:
        cli = OpenAI(api_key=api_key, base_url=base_url, timeout=60)
        resp = cli.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"LLM call failed: {type(e).__name__}: {e}"


def main():
    console.print("[bold cyan]🤖 Nado Backtest Engine[/bold cyan]")
    console.print(f"   Symbols:  {len(SYMBOLS)}")
    console.print(f"   TF:       {TIMEFRAME}")
    console.print(f"   History:  {HISTORY_CANDLES} candles (~{HISTORY_CANDLES/24:.0f} days)")
    console.print(f"   Strategy: {STRATEGY_NAME.upper()}\n")

    fetcher = NadoDataFetcher()
    rows = []

    if STRATEGY_NAME == "both":
        strategies = [("v1", v1_signal), ("v2", v2_signal)]
    else:
        strategies = [(STRATEGY_NAME, SIGNAL_FNS[STRATEGY_NAME])]

    # Cache fetched data so we don't re-fetch when running both strategies
    df_cache = {}
    for sym in SYMBOLS:
        for label, sig in strategies:
            try:
                if sym not in df_cache:
                    df_cache[sym] = None  # marker
                r = run_one_cached(sym, fetcher, sig, label, df_cache)
                rows.append(r)
            except Exception as e:
                console.print(f"[red]│ ❌ {sym}/{label} failed: {e}[/red]")
                import traceback; traceback.print_exc()
                rows.append({"symbol": sym, "strategy": label, "error": str(e)})

    print_results_table(rows)

    console.print("\n[bold magenta]🧠 LLM Analyst[/bold magenta]")
    analysis = llm_analysis(rows)
    console.print(analysis)

    # Save full report
    suffix = STRATEGY_NAME
    report_path = Path(f"/root/nado-bot/reports/backtest_{suffix}.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"# Nado Backtest Report — Strategy {STRATEGY_NAME.upper()}\n\n")
        f.write(f"**Timeframe:** {TIMEFRAME}, {HISTORY_CANDLES} candles  \n")
        f.write(f"**Initial:** ${INITIAL_BALANCE} | Size: {SIZE_PCT*100}% | Leverage: {LEVERAGE}x\n\n")
        f.write("## Per-Symbol Summary\n\n")
        f.write("| Symbol | Strategy | Trades | WR% | Return% | MaxDD% | Sharpe | PF | EndBal$ |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            if "summary" in r:
                s = r["summary"]
                f.write(
                    f"| {s['symbol']} | {s['strategy']} | {s['num_trades']} | "
                    f"{s['win_rate']:.1f} | {s['total_return_pct']:+.2f} | "
                    f"{s['max_drawdown_pct']:.2f} | {s['sharpe']:.2f} | "
                    f"{s['profit_factor']} | {s['end_balance']:.2f} |\n"
                )
        f.write(f"\n## LLM Analysis\n\n{analysis}\n")
    console.print(f"\n[green]📄 Report saved: {report_path}[/green]")


def run_one_cached(symbol, fetcher, signal_fn, label, df_cache):
    """Reuse fetched data + indicators across strategies."""
    if df_cache.get(symbol) is None:
        console.print(f"\n[cyan]┌─ {symbol} [{label}] ─[/cyan]")
        console.print(f"[dim]│ Fetching {HISTORY_CANDLES} {TIMEFRAME} candles...[/dim]")
        df = fetcher.fetch_history(symbol, TIMEFRAME, HISTORY_CANDLES)
        if df.empty or len(df) < 250:
            console.print(f"[red]│ ❌ Insufficient data[/red]")
            return {"symbol": symbol, "strategy": label, "error": "insufficient_data"}
        console.print(f"[dim]│ Got {len(df)} candles[/dim]")
        df = add_indicators(df)
        df_cache[symbol] = df
    else:
        console.print(f"\n[cyan]┌─ {symbol} [{label}] ─[/cyan] [dim](cached)[/dim]")

    df = df_cache[symbol]
    df.attrs["symbol"] = symbol  # for LLM strategy
    bt = Backtester(
        symbol=symbol, timeframe=TIMEFRAME,
        initial_balance=INITIAL_BALANCE,
        size_per_trade_pct=SIZE_PCT,
        leverage=LEVERAGE,
    )
    if label == "v3":
        v3_reset()
    result = bt.run(df, signal_fn)
    summary = result.summary()
    summary["strategy"] = label
    if label == "v3":
        s = v3_stats()
        summary["llm_calls"] = s["llm_calls"]
        summary["candidates"] = s["candidates"]
    color = "green" if result.total_return_pct > 0 else "red"
    extra = f" | LLM calls: {v3_stats()['llm_calls']}" if label == "v3" else ""
    console.print(
        f"[{color}]│ {result.num_trades} trades | "
        f"Return: {result.total_return_pct:+.2f}% | "
        f"WR: {result.win_rate:.1f}%{extra}[/{color}]"
    )
    return {"symbol": symbol, "strategy": label, "result": result, "summary": summary}


if __name__ == "__main__":
    main()
