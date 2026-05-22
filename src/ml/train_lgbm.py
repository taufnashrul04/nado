"""
ML training: LightGBM classifier untuk prediksi TP-first vs SL-first.
Walk-forward validation: train 6 bulan → test 1 bulan → slide forward.
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/root/nado-bot/src')

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from pathlib import Path
import pickle
import time

from data.ccxt_fetcher import fetch_kucoin_history
from ml.features import compute_features, compute_labels

MODEL_DIR = Path("/root/nado-bot/data/ml_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Drop these from feature matrix (target / metadata / OHLCV raw)
DROP_COLS = ["open", "high", "low", "close", "volume", "label_long", "label_short"]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLS]


def walk_forward_train(df: pd.DataFrame, side: str = "long",
                       train_days: int = 180, test_days: int = 30,
                       step_days: int = 30) -> dict:
    """
    Walk-forward CV.
    Returns dict with per-fold metrics + aggregate WR at high-confidence threshold.
    """
    label_col = f"label_{side}"

    # Compute time-based fold boundaries
    start_ts = df.index[0]
    end_ts = df.index[-1]
    train_td = pd.Timedelta(days=train_days)
    test_td = pd.Timedelta(days=test_days)
    step_td = pd.Timedelta(days=step_days)

    fold = 0
    fold_results = []
    cur_train_start = start_ts
    all_test_preds = []
    all_test_labels = []
    feature_cols = get_feature_cols(df)

    print(f"\n=== Walk-Forward {side.upper()} | features={len(feature_cols)} ===")
    print(f"Data range: {start_ts} → {end_ts}")
    print(f"Train: {train_days}d | Test: {test_days}d | Step: {step_days}d\n")

    while True:
        train_end = cur_train_start + train_td
        test_end = train_end + test_td
        if test_end > end_ts:
            break

        train_df = df[(df.index >= cur_train_start) & (df.index < train_end)].dropna(subset=feature_cols + [label_col])
        test_df = df[(df.index >= train_end) & (df.index < test_end)].dropna(subset=feature_cols + [label_col])

        if len(train_df) < 1000 or len(test_df) < 100:
            cur_train_start += step_td
            continue

        X_tr = train_df[feature_cols].values
        y_tr = train_df[label_col].values
        X_te = test_df[feature_cols].values
        y_te = test_df[label_col].values

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(30, verbose=False)])

        proba = model.predict_proba(X_te)[:, 1]
        all_test_preds.extend(proba.tolist())
        all_test_labels.extend(y_te.tolist())

        # Metrics at default threshold 0.5
        pred05 = (proba >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y_te, proba)
        except Exception:
            auc = float("nan")

        # WR at high threshold (only count predictions with proba >= 0.65)
        high_mask = proba >= 0.65
        if high_mask.sum() > 0:
            wr_high = y_te[high_mask].mean() * 100
            n_high = int(high_mask.sum())
        else:
            wr_high = float("nan")
            n_high = 0

        result = {
            "fold": fold,
            "train_range": f"{cur_train_start.date()} → {train_end.date()}",
            "test_range": f"{train_end.date()} → {test_end.date()}",
            "n_train": len(train_df),
            "n_test": len(test_df),
            "auc": auc,
            "acc_05": accuracy_score(y_te, pred05),
            "base_rate_pos": y_te.mean(),
            "wr_high_thresh": wr_high,
            "n_high_thresh": n_high,
        }
        fold_results.append(result)
        print(f"Fold {fold} | {result['test_range']} | n_test={len(test_df)} | "
              f"AUC={auc:.3f} base_rate={y_te.mean():.3f} | "
              f"WR@0.65={wr_high:.1f}% (n={n_high})")

        fold += 1
        cur_train_start += step_td

    # Aggregate
    all_proba = np.array(all_test_preds)
    all_y = np.array(all_test_labels)

    # WR at multiple thresholds
    print(f"\n=== AGGREGATE {side.upper()} ===")
    print(f"Total test samples: {len(all_y)}, base WR (always-trade): {all_y.mean()*100:.1f}%")
    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = all_proba >= thresh
        n = mask.sum()
        if n > 0:
            wr = all_y[mask].mean() * 100
            print(f"  Threshold ≥{thresh}: n_trades={n}, WR={wr:.1f}%")
        else:
            print(f"  Threshold ≥{thresh}: 0 trades")

    return {
        "side": side,
        "folds": fold_results,
        "all_proba": all_proba.tolist(),
        "all_labels": all_y.tolist(),
        "feature_cols": feature_cols,
    }


def main():
    print("Loading 2 years BTC/USDT 15m KuCoin data...")
    raw = fetch_kucoin_history("BTC/USDT", "15m", years_back=2.0)
    print(f"Loaded {len(raw)} candles")

    print("Computing features (40+)...")
    t0 = time.time()
    df = compute_features(raw)
    print(f"Features done in {time.time()-t0:.1f}s | df shape: {df.shape}")

    print("Computing labels (long, SL=1.2xATR, TP=1.0xATR, hold≤16 bars)...")
    t0 = time.time()
    df["label_long"] = compute_labels(df, side="long", sl_atr_mult=1.2, tp_atr_mult=1.0, max_bars_hold=16)
    df["label_short"] = compute_labels(df, side="short", sl_atr_mult=1.2, tp_atr_mult=1.0, max_bars_hold=16)
    print(f"Labels done in {time.time()-t0:.1f}s")
    print(f"Long label dist: {df['label_long'].value_counts(dropna=False).to_dict()}")
    print(f"Short label dist: {df['label_short'].value_counts(dropna=False).to_dict()}")

    # Drop bars where indicators not yet warm
    df = df.dropna(subset=["ema_200", "atr", "adx", "stoch_k"])

    long_results = walk_forward_train(df, side="long")
    short_results = walk_forward_train(df, side="short")

    # Save
    with open(MODEL_DIR / "walkforward_results.pkl", "wb") as fh:
        pickle.dump({"long": long_results, "short": short_results}, fh)
    print(f"\nSaved results to {MODEL_DIR / 'walkforward_results.pkl'}")


if __name__ == "__main__":
    main()
