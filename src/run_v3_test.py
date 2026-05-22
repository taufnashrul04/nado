"""V3 LLM strategy smoke test — outputs progress every 50 bars."""
import os, sys, time
sys.path.insert(0, '/root/nado-bot/src')
from data.fetcher import NadoDataFetcher
from data.indicators import add_indicators
from backtest.engine import Backtester
from strategies.llm_decision import signal_fn, get_stats, reset_stats

SYMBOLS = ["BTC-PERP", "ETH-PERP", "SPY-PERP", "NVDA-PERP", "AAPL-PERP"]
HISTORY = 1000  # ~42 days

fetcher = NadoDataFetcher()
all_results = []

for sym in SYMBOLS:
    print(f"\n{'='*60}", flush=True)
    print(f"  {sym} — fetching {HISTORY} candles", flush=True)
    print('='*60, flush=True)
    t0 = time.time()
    df = fetcher.fetch_history(sym, '1h', total_candles=HISTORY)
    if df.empty or len(df) < 250:
        print(f"  ❌ no data", flush=True)
        continue
    print(f"  ✓ got {len(df)} candles in {time.time()-t0:.1f}s", flush=True)
    df = add_indicators(df)
    df.attrs['symbol'] = sym

    reset_stats()
    bt = Backtester(sym, '1h', initial_balance=130, size_per_trade_pct=0.05, leverage=3.0)
    t0 = time.time()
    result = bt.run(df, signal_fn)
    elapsed = time.time() - t0
    s = get_stats()

    print(f"\n  📊 {sym} RESULTS:")
    print(f"     Trades: {result.num_trades}")
    print(f"     Win Rate: {result.win_rate:.1f}%")
    print(f"     Return: {result.total_return_pct:+.2f}%")
    print(f"     Max DD: {result.max_drawdown_pct:.2f}%")
    print(f"     Sharpe: {result.sharpe:.2f}")
    print(f"     LLM calls: {s['llm_calls']} ({s['long']}L / {s['short']}S / {s['skip']}skip)")
    print(f"     Elapsed: {elapsed:.0f}s", flush=True)
    if result.trades:
        wins = [t for t in result.trades if t.pnl_usd > 0]
        losses = [t for t in result.trades if t.pnl_usd < 0]
        print(f"     Wins: {len(wins)} avg ${sum(t.pnl_usd for t in wins)/max(len(wins),1):+.2f}")
        print(f"     Losses: {len(losses)} avg ${sum(t.pnl_usd for t in losses)/max(len(losses),1):+.2f}")

    all_results.append({
        'symbol': sym, 'trades': result.num_trades,
        'wr': result.win_rate, 'return': result.total_return_pct,
        'max_dd': result.max_drawdown_pct, 'sharpe': result.sharpe,
        'llm_calls': s['llm_calls'],
    })

print(f"\n\n{'='*70}")
print("SUMMARY")
print('='*70)
print(f"{'Symbol':<14} {'Trades':>7} {'WR%':>6} {'Return%':>9} {'MaxDD%':>8} {'Sharpe':>7} {'LLM':>5}")
for r in all_results:
    color_wr = "✅" if r['wr'] >= 70 else "⚠️" if r['wr'] >= 50 else "❌"
    print(f"{r['symbol']:<14} {r['trades']:>7} {r['wr']:>5.1f} {color_wr} "
          f"{r['return']:>+8.2f} {r['max_dd']:>+7.2f} {r['sharpe']:>+6.2f} {r['llm_calls']:>5}")
