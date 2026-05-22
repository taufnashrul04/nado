"""
V10 AI Multi-Layer Backtest — BTC, ETH, SOL focus
4h, 6mo Nado data, lev assumed 50x analysis lens.
"""
import sys, time
sys.path.insert(0, 'src')
import pandas as pd

from data.fetcher import NadoDataFetcher
from data.indicators import add_indicators
from backtest.engine import Backtester
from strategies.ai_multilayer_v10 import signal_fn, get_stats, reset_stats

SYMBOLS = ["BTC-PERP", "ETH-PERP", "SOL-PERP", "BNB-PERP", "DOGE-PERP"]
TIMEFRAME = "4h"
TOTAL_CANDLES = 1100


def fmt_pct(x): return f"{x:+.2f}%"
def fmt_pf(trades):
    wins = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
    losses = abs(sum(t.pnl_usd for t in trades if t.pnl_usd < 0))
    if losses == 0: return float('inf') if wins > 0 else 0.0
    return wins / losses


def main():
    print("=" * 110, flush=True)
    print(f"V10 MULTI-LAYER AI BACKTEST — Stoch+EMA+MTF+TV consensus, 4h, 6mo", flush=True)
    print("=" * 110, flush=True)
    print(f"\n{'Symbol':<14} {'Cand':>5} {'vMTF':>5} {'vTV':>5} {'vAI':>5} {'LLM':>5} {'TV':>4} {'Trd':>4} "
          f"{'WR%':>6} {'Ret%':>8} {'PF':>5} {'Hold':>6}", flush=True)
    print("-" * 110, flush=True)

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
                print(f"{sym:<14} INSUFFICIENT DATA", flush=True)
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
            print(f"{sym:<14} {stats['candidates']:>5} {stats['veto_mtf']:>5} {stats['veto_tv']:>5} "
                  f"{stats['veto_ai']:>5} {stats['llm_calls']:>5} {stats['tv_calls']:>4} {r.num_trades:>4} "
                  f"{r.win_rate:>5.1f}% {fmt_pct(r.total_return_pct):>8} {pf:>5.2f} "
                  f"{avg_hold_h:>4.0f}h ({elapsed/60:.1f}m)", flush=True)

            all_trades.extend(r.trades)
            total_return += r.total_return_pct
            total_llm += stats['llm_calls']
            rows.append({"sym": sym, "trades": r.num_trades, "wr": r.win_rate,
                       "ret": r.total_return_pct, "pf": pf, "hold_h": avg_hold_h})
        except Exception as e:
            import traceback
            print(f"{sym:<14} ERROR: {str(e)[:80]}", flush=True)
            traceback.print_exc()

    print("-" * 110, flush=True)
    if all_trades:
        wins = [t for t in all_trades if t.pnl_usd > 0]
        avg_wr = len(wins) / len(all_trades) * 100
        agg_pf = fmt_pf(all_trades)
        print(f"\n{'TOTAL':<14} {'':>5} {'':>5} {'':>5} {'':>5} {total_llm:>5} {'':>4} "
             f"{len(all_trades):>4} {avg_wr:>5.1f}% {fmt_pct(total_return):>8} {agg_pf:>5.2f}", flush=True)

        long_t = [t for t in all_trades if t.side == "long"]
        short_t = [t for t in all_trades if t.side == "short"]
        if long_t:
            l_wr = sum(1 for t in long_t if t.pnl_usd > 0) / len(long_t) * 100
            print(f"\nLong only:  {len(long_t):>3} trades, WR={l_wr:.1f}%, PF={fmt_pf(long_t):.2f}", flush=True)
        if short_t:
            s_wr = sum(1 for t in short_t if t.pnl_usd > 0) / len(short_t) * 100
            print(f"Short only: {len(short_t):>3} trades, WR={s_wr:.1f}%, PF={fmt_pf(short_t):.2f}", flush=True)

        winners = sorted([r for r in rows if r["trades"] >= 2], key=lambda x: x["wr"], reverse=True)
        if winners:
            print(f"\nTop ranked (min 2 trades):", flush=True)
            for r in winners:
                print(f"  {r['sym']:<14} Trades={r['trades']:>3}  WR={r['wr']:.1f}%  Ret={r['ret']:+.2f}%  "
                      f"PF={r['pf']:.2f}", flush=True)


if __name__ == "__main__":
    main()
