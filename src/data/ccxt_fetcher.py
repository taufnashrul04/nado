"""
CCXT data fetcher — KuCoin BTC/USDT 15m for ML training.
(Binance geo-blocked, OKX limited to 1440 candles, KuCoin supports 2+ years history.)
Cached to disk via pickle.
"""
from __future__ import annotations
import time
import pandas as pd
from pathlib import Path
import ccxt

CACHE_DIR = Path("/root/nado-bot/data/ccxt_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_kucoin_history(symbol: str = "BTC/USDT", timeframe: str = "15m",
                         years_back: float = 2.0, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch historical OHLCV from KuCoin via CCXT.
    Walks forward from `years_back` ago to now.
    """
    cache_file = CACHE_DIR / f"kucoin_{symbol.replace('/','_')}_{timeframe}_{years_back}y.pkl"
    if use_cache and cache_file.exists():
        df = pd.read_pickle(cache_file)
        print(f"[cache] Loaded {len(df)} candles from {cache_file.name}")
        return df

    ex = ccxt.kucoin({"enableRateLimit": True})
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    chunk_size = 1500  # KuCoin max
    now_ms = ex.milliseconds()
    since_ms = now_ms - int(years_back * 365 * 24 * 60 * 60 * 1000)

    all_rows = []
    next_since = since_ms

    while next_since < now_ms:
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe, since=next_since, limit=chunk_size)
        except Exception as e:
            print(f"[fetch error] {e}; retrying after 5s")
            time.sleep(5)
            continue

        if not batch:
            print("Empty batch, stopping")
            break

        all_rows.extend(batch)
        last_ts = batch[-1][0]
        next_since = last_ts + tf_ms
        print(f"  fetched {len(all_rows)} total, last ts: {pd.Timestamp(last_ts, unit='ms')}", flush=True)

        if len(batch) < chunk_size:
            break

        time.sleep(0.4)

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("ts").drop_duplicates()
    df = df.sort_index()

    if use_cache:
        df.to_pickle(cache_file)
        print(f"[cache] Saved {len(df)} candles to {cache_file.name}")

    return df


if __name__ == "__main__":
    df = fetch_kucoin_history("BTC/USDT", "15m", years_back=2.0)
    print(f"\nFinal: {len(df)} candles")
    print(f"Range: {df.index[0]} → {df.index[-1]}")
    print(f"Days covered: {(df.index[-1] - df.index[0]).days}")
