"""
V4 Backtest Runner — 15m Scalping
Tests V4 strategy across all 10 symbols with 15m candles.
"""
import os, sys, time
sys.path.insert(0, 'src')
from data.fetcher import NadoDataFetcher
from data.indicators import add_indicators
from backtest.engine import Backtester
from strategies.scalp_15m import signal_fn

SYMBOLS = ["BTC-PERP", "ETH-PERP", "SOL-PERP", "WTI-PERP", "QQQ-PERP",
           "SPY-PERP", "AAPL-PERP", "GOOGL-PERP", "NVDA-PERP", "TSLA-PERP"]
TIMEFRAME = "15m"
TOTAL_CANDLES = 10000  # ~104 hari di 15m (max yang Nado API support)

def fmt_pct(x): return f"{x:+.2f}%"
def fmt_pf(t):
    wins = sum(1 for x in t if x.pnl_usd > 0)
    losses = sum(abs(x.pnl_usd) for x in t if x.pnl_usd < 0)
    gains = sum(x.pnl_usd for x in t if x.pnl_usd > 0)
    if losses == 0: return float('inf') if gains > 0 else 0.0
    return gains / losses

def main():
    fetcher = NadoDataFetcher()
    rows = []
    print(f"\n{'='*80}\nV4 BACKTEST — 15m Scalping (rule-based)\n{'='*80}\n")
    print(f"{'Symbol':<12} {'Trades':>7} {'WR%':>6} {'Ret%':>7} {'MaxDD':>7} {'PF':>5} {'Sharpe':>7}")
    print("-" * 60)

    total_trades = 0
    total_return_pct = 0
    aggregate_trades = []

    for sym in SYMBOLS:
        try:
            t0 = time.time()
            df = fetcher.fetch_history(sym, TIMEFRAME, total_candles=TOTAL_CANDLES)
            df = add_indicators(df)
            df.attrs['symbol'] = sym

            bt = Backtester(sym, TIMEFRAME, initial_balance=130, size_per_trade_pct=0.05, leverage=3.0)
            r = bt.run(df, signal_fn)
            elapsed = time.time() - t0

            pf = fmt_pf(r.trades)
            print(f"{sym:<12} {r.num_trades:>7} {r.win_rate:>5.1f}% {fmt_pct(r.total_return_pct):>7} "
                  f"{fmt_pct(r.max_drawdown_pct):>7} {pf:>5.2f} {r.sharpe:>7.2f}  ({elapsed:.0f}s)")

            rows.append({
                "symbol": sym, "trades": r.num_trades, "wr": r.win_rate,
                "ret": r.total_return_pct, "dd": r.max_drawdown_pct,
                "pf": pf, "sharpe": r.sharpe,
            })
            total_trades += r.num_trades
            total_return_pct += r.total_return_pct
            aggregate_trades.extend(r.trades)
        except Exception as e:
            print(f"{sym:<12} ERROR: {e}")

    print("-" * 60)
    avg_wr = sum(1 for x in aggregate_trades if x.pnl_usd > 0) / max(len(aggregate_trades), 1) * 100
    agg_pf = fmt_pf(aggregate_trades)
    print(f"{'TOTAL':<12} {total_trades:>7} {avg_wr:>5.1f}% {fmt_pct(total_return_pct):>7} "
          f"{'':>7} {agg_pf:>5.2f}")

    # Exit reason breakdown (aggregate)
    reasons = {}
    for t in aggregate_trades:
        r = t.reason or "?"
        reasons[r] = reasons.get(r, 0) + 1
    print(f"\nExit reasons: {reasons}")

    # Top 3 winners by Sharpe
    rows_sorted = sorted([r for r in rows if r["trades"] >= 5], key=lambda x: x["sharpe"], reverse=True)
    print(f"\nTop 3 by Sharpe (min 5 trades):")
    for r in rows_sorted[:3]:
        print(f"  {r['symbol']:<12} Trades={r['trades']} WR={r['wr']:.1f}% Ret={r['ret']:+.2f}% PF={r['pf']:.2f} Sharpe={r['sharpe']:.2f}")

if __name__ == "__main__":
    main()
