"""
V7 4H Intraday Swing Backtest — multi-symbol, 3 years history.
Hold: 24-48h max.
"""
import sys, time
sys.path.insert(0, 'src')
import pandas as pd

from data.ccxt_fetcher import fetch_kucoin_history
from data.indicators import add_indicators
from backtest.engine import Backtester
from strategies.swing_4h import signal_fn

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
           "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "POL/USDT"]
TIMEFRAME = "4h"
YEARS = 3.0


def fmt_pct(x): return f"{x:+.2f}%"
def fmt_pf(trades):
    wins = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
    losses = abs(sum(t.pnl_usd for t in trades if t.pnl_usd < 0))
    if losses == 0: return float('inf') if wins > 0 else 0.0
    return wins / losses


def main():
    print("=" * 90)
    print(f"V7 4H INTRADAY SWING BACKTEST — {YEARS}y, hold max 48h, 4h chart")
    print("=" * 90)
    print(f"\n{'Symbol':<14} {'Trades':>7} {'WR%':>6} {'Ret%':>8} {'MaxDD':>8} {'PF':>5} {'Sharpe':>7} {'Hold':>7}")
    print("-" * 80)

    all_trades = []
    rows = []
    total_return = 0

    for sym in SYMBOLS:
        try:
            t0 = time.time()
            df = fetch_kucoin_history(sym, TIMEFRAME, years_back=YEARS)
            df = add_indicators(df)
            df.attrs['symbol'] = sym

            bt = Backtester(sym, TIMEFRAME, initial_balance=130,
                           size_per_trade_pct=0.05, leverage=3.0, fee_pct=0.0005)
            r = bt.run(df, signal_fn)
            elapsed = time.time() - t0

            avg_hold_h = (sum((t.exit_ts - t.entry_ts).total_seconds()/3600
                             for t in r.trades if t.exit_ts) / max(r.num_trades, 1))
            pf = fmt_pf(r.trades)
            print(f"{sym:<14} {r.num_trades:>7} {r.win_rate:>5.1f}% {fmt_pct(r.total_return_pct):>8} "
                  f"{fmt_pct(r.max_drawdown_pct):>8} {pf:>5.2f} {r.sharpe:>7.2f} {avg_hold_h:>5.1f}h  ({elapsed:.0f}s)")

            all_trades.extend(r.trades)
            total_return += r.total_return_pct
            rows.append({
                "sym": sym, "trades": r.num_trades, "wr": r.win_rate,
                "ret": r.total_return_pct, "pf": pf, "sharpe": r.sharpe, "hold_h": avg_hold_h,
            })
        except Exception as e:
            print(f"{sym:<14} ERROR: {str(e)[:80]}")

    print("-" * 80)
    if all_trades:
        wins = [t for t in all_trades if t.pnl_usd > 0]
        avg_wr = len(wins) / len(all_trades) * 100
        agg_pf = fmt_pf(all_trades)
        print(f"{'TOTAL':<14} {len(all_trades):>7} {avg_wr:>5.1f}% {fmt_pct(total_return):>8} "
              f"{'':>8} {agg_pf:>5.2f}")

        reasons = {}
        for t in all_trades:
            reasons[t.reason or "?"] = reasons.get(t.reason or "?", 0) + 1
        print(f"\nExit reasons: {reasons}")

        long_t = [t for t in all_trades if t.side == "long"]
        short_t = [t for t in all_trades if t.side == "short"]
        if long_t:
            l_wr = sum(1 for t in long_t if t.pnl_usd > 0) / len(long_t) * 100
            print(f"\nLong only:  {len(long_t)} trades, WR={l_wr:.1f}%, PF={fmt_pf(long_t):.2f}")
        if short_t:
            s_wr = sum(1 for t in short_t if t.pnl_usd > 0) / len(short_t) * 100
            print(f"Short only: {len(short_t)} trades, WR={s_wr:.1f}%, PF={fmt_pf(short_t):.2f}")

        winners = sorted([r for r in rows if r["trades"] >= 10], key=lambda x: x["wr"], reverse=True)
        print(f"\nTop 5 by WR (min 10 trades):")
        for r in winners[:5]:
            print(f"  {r['sym']:<10} Trades={r['trades']:>4}  WR={r['wr']:.1f}%  Ret={r['ret']:+.2f}%  "
                  f"PF={r['pf']:.2f}  Sharpe={r['sharpe']:.2f}  Hold={r['hold_h']:.0f}h")


if __name__ == "__main__":
    main()
