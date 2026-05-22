"""
Full backtest using ML signals (LightGBM walk-forward).
Trades only when ML probability >= threshold for LONG or SHORT.
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/root/nado-bot/src')

import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import pickle
from pathlib import Path

from data.ccxt_fetcher import fetch_kucoin_history
from ml.features import compute_features, compute_labels
from backtest.engine import Backtester, BacktestResult


MODEL_DIR = Path("/root/nado-bot/data/ml_models")
DROP_COLS = ["open", "high", "low", "close", "volume", "label_long", "label_short"]


def get_feature_cols(df):
    return [c for c in df.columns if c not in DROP_COLS]


def generate_walkforward_signals(df: pd.DataFrame,
                                  train_days: int = 180, test_days: int = 30,
                                  step_days: int = 30, threshold: float = 0.65) -> pd.DataFrame:
    """
    Walk-forward: at each step, train on past data and predict on current test window.
    Returns DataFrame with new columns: 'proba_long', 'proba_short'.
    """
    feature_cols = get_feature_cols(df)
    df = df.copy()
    df["proba_long"] = np.nan
    df["proba_short"] = np.nan

    start_ts = df.index[0]
    end_ts = df.index[-1]
    train_td = pd.Timedelta(days=train_days)
    test_td = pd.Timedelta(days=test_days)
    step_td = pd.Timedelta(days=step_days)

    cur_train_start = start_ts
    fold = 0
    while True:
        train_end = cur_train_start + train_td
        test_end = train_end + test_td
        if test_end > end_ts:
            break

        train_df = df[(df.index >= cur_train_start) & (df.index < train_end)].dropna(subset=feature_cols + ["label_long", "label_short"])
        test_df = df[(df.index >= train_end) & (df.index < test_end)].dropna(subset=feature_cols)

        if len(train_df) < 1000 or len(test_df) < 100:
            cur_train_start += step_td
            continue

        X_tr = train_df[feature_cols].values
        X_te = test_df[feature_cols].values

        # Long model
        model_l = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=6, num_leaves=31,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        model_l.fit(X_tr, train_df["label_long"].values)
        proba_l = model_l.predict_proba(X_te)[:, 1]
        df.loc[test_df.index, "proba_long"] = proba_l

        # Short model
        model_s = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=6, num_leaves=31,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        model_s.fit(X_tr, train_df["label_short"].values)
        proba_s = model_s.predict_proba(X_te)[:, 1]
        df.loc[test_df.index, "proba_short"] = proba_s

        n_long = int((proba_l >= threshold).sum())
        n_short = int((proba_s >= threshold).sum())
        print(f"Fold {fold} | {train_end.date()}→{test_end.date()} | "
              f"Long@≥{threshold}: {n_long}, Short@≥{threshold}: {n_short}", flush=True)

        fold += 1
        cur_train_start += step_td

    return df


def make_signal_fn(threshold: float = 0.65, sl_atr_mult: float = 1.2,
                   tp_atr_mult: float = 1.0, max_bars_hold: int = 16):
    def signal_fn(df, i):
        if i < 200:
            return None
        row = df.iloc[i]
        proba_l = row.get("proba_long", np.nan)
        proba_s = row.get("proba_short", np.nan)
        atr = row.get("atr", np.nan)
        close = row["close"]
        if pd.isna(atr) or atr <= 0:
            return None

        sl_pct = max(sl_atr_mult * atr / close, 0.004)
        tp_pct = max(tp_atr_mult * atr / close, 0.004)

        if not pd.isna(proba_l) and proba_l >= threshold:
            return {
                "action": "open_long", "sl_pct": sl_pct, "tp_pct": tp_pct,
                "max_bars_hold": max_bars_hold,
                "reason": f"ML_long p={proba_l:.3f}",
            }
        if not pd.isna(proba_s) and proba_s >= threshold:
            return {
                "action": "open_short", "sl_pct": sl_pct, "tp_pct": tp_pct,
                "max_bars_hold": max_bars_hold,
                "reason": f"ML_short p={proba_s:.3f}",
            }
        return None
    return signal_fn


def main():
    print("=" * 70)
    print("V5 ML BACKTEST — BTC/USDT 15m KuCoin (2 years walk-forward)")
    print("=" * 70)

    # Cache check for signals (expensive to regenerate)
    sig_cache = MODEL_DIR / "btc_15m_signals_t065.pkl"

    print("\n[1/4] Loading 2y BTC/USDT 15m KuCoin data...")
    raw = fetch_kucoin_history("BTC/USDT", "15m", years_back=2.0)
    print(f"  Loaded {len(raw)} candles")

    print("\n[2/4] Computing features...")
    t0 = time.time()
    df = compute_features(raw)
    print(f"  Done in {time.time()-t0:.1f}s | shape={df.shape}")

    print("\n[3/4] Computing labels (long+short)...")
    t0 = time.time()
    df["label_long"] = compute_labels(df, side="long", sl_atr_mult=1.2, tp_atr_mult=1.0, max_bars_hold=16)
    df["label_short"] = compute_labels(df, side="short", sl_atr_mult=1.2, tp_atr_mult=1.0, max_bars_hold=16)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Drop bars where indicators not warm
    df = df.dropna(subset=["ema_200", "atr", "adx", "stoch_k"])

    if sig_cache.exists():
        print(f"\n[cache] Loading signals from {sig_cache.name}")
        df = pd.read_pickle(sig_cache)
    else:
        print("\n[4/4] Walk-forward signal generation...")
        t0 = time.time()
        df = generate_walkforward_signals(df, train_days=180, test_days=30, step_days=30, threshold=0.65)
        print(f"  Done in {time.time()-t0:.1f}s")
        df.to_pickle(sig_cache)
        print(f"  Saved to {sig_cache}")

    # Diagnostic
    n_long_signals = int((df["proba_long"] >= 0.65).sum())
    n_short_signals = int((df["proba_short"] >= 0.65).sum())
    print(f"\nSignals total: long={n_long_signals}, short={n_short_signals}")

    # Run backtest
    print(f"\n{'='*70}\nBACKTEST EXECUTION\n{'='*70}")
    bt = Backtester("BTC-PERP", "15m", initial_balance=130, size_per_trade_pct=0.05,
                    leverage=3.0, fee_pct=0.0005)
    sig_fn = make_signal_fn(threshold=0.65, sl_atr_mult=1.2, tp_atr_mult=1.0, max_bars_hold=16)
    result = bt.run(df, sig_fn)

    print(f"\nSymbol: BTC-PERP 15m")
    print(f"Test period covered: {df.index[180*96]} → {df.index[-1]}")
    print(f"Trades: {result.num_trades}")
    print(f"Win Rate: {result.win_rate:.1f}%")
    print(f"Return: {result.total_return_pct:+.2f}%")
    print(f"Max DD: {result.max_drawdown_pct:.2f}%")
    print(f"Sharpe: {result.sharpe:.2f}")

    wins = [t for t in result.trades if t.pnl_usd > 0]
    losses = [t for t in result.trades if t.pnl_usd < 0]
    print(f"\nWins: {len(wins)} (avg ${np.mean([t.pnl_usd for t in wins]):.3f})" if wins else "Wins: 0")
    print(f"Losses: {len(losses)} (avg ${np.mean([t.pnl_usd for t in losses]):.3f})" if losses else "Losses: 0")
    if wins and losses:
        gains = sum(t.pnl_usd for t in wins)
        loss_abs = abs(sum(t.pnl_usd for t in losses))
        print(f"Profit Factor: {gains / loss_abs:.2f}")

    # Exit reason breakdown
    reasons = {}
    for t in result.trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    print(f"\nExit reasons: {reasons}")

    print("\nFirst 10 trades:")
    for t in result.trades[:10]:
        print(f"  {t.entry_ts.strftime('%Y-%m-%d %H:%M')} {t.side:>5} "
              f"entry={t.entry_price:.2f} exit={(t.exit_price or 0):.2f} "
              f"pnl=${t.pnl_usd:+.3f} reason={t.reason}")


if __name__ == "__main__":
    main()
