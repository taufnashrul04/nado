"""
V9 AI Backtest — User's exact symbols (Nado data, 4h)
Crypto: BTC, ETH, SOL
Equity perp: WTI, SPY, QQQ, NVDA, TSLA, AAPL, GOOGL
"""
import sys, time
sys.path.insert(0, 'src')
import pandas as pd

from data.fetcher import NadoDataFetcher
from data.indicators import add_indicators
from backtest.engine import Backtester
from strategies.ai_stoch_ema import signal_fn, get_stats, reset_stats

SYMBOLS = [
    "BTC-PERP", "ETH-PERP", "SOL-PERP",          # crypto core
    "WTI-PERP", "SPY-PERP", "QQQ-PERP",           # commodities + indices
    "NVDA-PERP", "TSLA-PERP", "AAPL-PERP", "GOOGL-PERP",  # mag stocks
]
TIMEFRAME = "4h"
TOTAL_CANDLES = 1100  # ~6 months @ 4h


def fmt_pct(x): return f"{x:+.2f}%"
def fmt_pf(trades):
    wins = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
    losses = abs(sum(t.pnl_usd for t in trades if t.pnl_usd < 0))
    if losses == 0: return float('inf') if wins > 0 else 0.0
    return wins / losses


def main():
    print("=" * 100)
    print(f"V9 AI BACKTEST — User method (Stoch 5,3,3 + EMA 13/21/34/50/90/200)")
    print(f"TF: {TIMEFRAME} | Hist: {TOTAL_CANDLES} candles (~6mo) | Nado data")
    print("=" * 100)
    print(f"\n{'Symbol':<14} {'Cand':>5} {'LLM':>5} {'Cache':>6} {'Trd':>4} {'WR%':>6} "
          f"{'Ret%':>8} {'PF':>5} {'Sharpe':>7} {'Hold':>7}")
    print("-" * 95)

    fetcher = NadoDataFetcher()
    all_trades = []
    rows = []
    total_return = 0
    total_llm = 0

    for sym in SYMBOLS:
        try:
            t0 = time.time()
            df = fetcher.fetch_history(sym, TIMEFRAME, total_candles=TOTAL_CANDLES)
            if df is None or len(df) < 200:
                print(f"{sym:<14} INSUFFICIENT DATA ({len(df) if df is not None else 0} candles)")
                continue
            df = add_indicators(df)
            df.attrs['symbol'] = sym

            reset_stats()
            bt = Backtester(sym, TIMEFRAME, initial_balance=130,
                          size_per_trade_pct=0.05, leverage=3.0, fee_pct=0.0005)
            r = bt.run(df, signal_fn)
            elapsed = time.time() - t0

            stats = get_stats()
            avg_hold_h = (sum((t.exit_ts - t.entry_ts).total_seconds()/3600
                            for t in r.trades if t.exit_ts) / max(r.num_trades, 1))
            pf = fmt_pf(r.trades)
            print(f"{sym:<14} {stats['candidates']:>5} {stats['llm_calls']:>5} "
                  f"{stats['cache_hits']:>6} {r.num_trades:>4} {r.win_rate:>5.1f}% "
                  f"{fmt_pct(r.total_return_pct):>8} {pf:>5.2f} {r.sharpe:>7.2f} "
                  f"{avg_hold_h:>5.0f}h  ({elapsed/60:.1f}min)", flush=True)

            all_trades.extend(r.trades)
            total_return += r.total_return_pct
            total_llm += stats['llm_calls']
            rows.append({"sym": sym, "trades": r.num_trades, "wr": r.win_rate,
                       "ret": r.total_return_pct, "pf": pf, "sharpe": r.sharpe,
                       "hold_h": avg_hold_h, "stats": stats})
        except Exception as e:
            import traceback
            print(f"{sym:<14} ERROR: {str(e)[:80]}")
            traceback.print_exc()

    print("-" * 95)
    if all_trades:
        wins = [t for t in all_trades if t.pnl_usd > 0]
        avg_wr = len(wins) / len(all_trades) * 100
        agg_pf = fmt_pf(all_trades)
        print(f"\n{'TOTAL':<14} {'':>5} {total_llm:>5} {'':>6} "
             f"{len(all_trades):>4} {avg_wr:>5.1f}% {fmt_pct(total_return):>8} {agg_pf:>5.2f}")

        reasons = {}
        for t in all_trades:
            r_short = ("tp" if "tp" in (t.reason or "") else
                      "sl" if "sl" in (t.reason or "") else
                      "time" if "time" in (t.reason or "") else "?")
            reasons[r_short] = reasons.get(r_short, 0) + 1
        print(f"\nExit reasons: {reasons}")

        long_t = [t for t in all_trades if t.side == "long"]
        short_t = [t for t in all_trades if t.side == "short"]
        if long_t:
            l_wr = sum(1 for t in long_t if t.pnl_usd > 0) / len(long_t) * 100
            print(f"Long only:  {len(long_t):>3} trades, WR={l_wr:.1f}%, PF={fmt_pf(long_t):.2f}")
        if short_t:
            s_wr = sum(1 for t in short_t if t.pnl_usd > 0) / len(short_t) * 100
            print(f"Short only: {len(short_t):>3} trades, WR={s_wr:.1f}%, PF={fmt_pf(short_t):.2f}")

        winners = sorted([r for r in rows if r["trades"] >= 2], key=lambda x: x["wr"], reverse=True)
        print(f"\nTop ranked (min 2 trades):")
        for r in winners:
            print(f"  {r['sym']:<14} Trades={r['trades']:>3}  WR={r['wr']:.1f}%  Ret={r['ret']:+.2f}%  "
                  f"PF={r['pf']:.2f}  Sharpe={r['sharpe']:.2f}")


if __name__ == "__main__":
    main()
