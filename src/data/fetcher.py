"""
Nado data fetcher — pulls OHLCV candlesticks from Nado API.
"""
from __future__ import annotations
import time
import logging
import pandas as pd
from nado_protocol.client import create_nado_client, NadoClientMode
from nado_protocol.indexer_client.types.query import IndexerCandlesticksParams

logger = logging.getLogger(__name__)

# Symbol → product_id mapping (from get_all_product_symbols)
SYMBOL_TO_PID = {
    "BTC-PERP": 2, "ETH-PERP": 4, "SOL-PERP": 8, "XRP-PERP": 10,
    "BNB-PERP": 14, "HYPE-PERP": 16, "SUI-PERP": 24, "DOGE-PERP": 52,
    "ADA-PERP": 60, "AVAX-PERP": 64, "LINK-PERP": 74,
    # Boost (TradFi/Equities)
    "WTI-PERP": 90, "QQQ-PERP": 98, "SPY-PERP": 100,
    "AAPL-PERP": 102, "GOOGL-PERP": 106, "NVDA-PERP": 112,
    "TSLA-PERP": 114, "AMZN-PERP": 104, "META-PERP": 108, "MSFT-PERP": 110,
    # FX/Commodities
    "EURUSD-PERP": 92, "GBPUSD-PERP": 94, "USDJPY-PERP": 96,
    "XAUT-PERP": 28, "XAG-PERP": 88,
}

# Granularity in seconds → Nado candlestick period
TIMEFRAMES = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "8h": 14400, "1d": 86400,
}

# Resample mapping: Nado native TF → user TF (only when needed)
RESAMPLE_FROM = {
    "8h": ("4h", "8h"),  # fetch 4h, resample to 8h (pandas 2.x lowercase)
}


class NadoDataFetcher:
    def __init__(self, pk: str | None = None, mode: NadoClientMode = NadoClientMode.MAINNET):
        if pk is None:
            with open("/root/nado-bot/secrets/1ct_key.txt") as f:
                pk = f.read().strip()
        self.client = create_nado_client(mode, pk)

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h",
                    limit: int = 500, end_ts: int | None = None) -> pd.DataFrame:
        """
        Fetch OHLCV from Nado.

        Args:
            symbol: e.g. "BTC-PERP"
            timeframe: "1m", "5m", "15m", "1h", "4h", "1d"
            limit: number of candles to fetch (max 500 per call)
            end_ts: end timestamp (seconds, default now)

        Returns DataFrame with index=timestamp, columns=[open,high,low,close,volume]
        """
        pid = SYMBOL_TO_PID.get(symbol)
        if pid is None:
            raise ValueError(f"Unknown symbol: {symbol}. Add to SYMBOL_TO_PID.")

        # For non-native TF (e.g. 8h), redirect to fetch_history which handles resample
        if timeframe in RESAMPLE_FROM:
            return self.fetch_history(symbol, timeframe, total_candles=limit)

        period = TIMEFRAMES[timeframe]

        if end_ts is None:
            end_ts = int(time.time())

        candles = self.client.market.get_candlesticks(
            IndexerCandlesticksParams(
                product_id=pid,
                granularity=period,
                limit=limit,
                max_time=end_ts,
            )
        )

        if not candles or not candles.candlesticks:
            return pd.DataFrame()

        rows = []
        for c in candles.candlesticks:
            rows.append({
                "ts": int(c.timestamp),
                "open": float(c.open_x18) / 1e18,
                "high": float(c.high_x18) / 1e18,
                "low": float(c.low_x18) / 1e18,
                "close": float(c.close_x18) / 1e18,
                "volume": float(c.volume),
            })
        df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
        df.index = pd.to_datetime(df["ts"], unit="s")
        return df[["open", "high", "low", "close", "volume"]]

    def fetch_history(self, symbol: str, timeframe: str = "1h",
                      total_candles: int = 2000) -> pd.DataFrame:
        """
        Fetch long history by paginating backwards.
        For non-native TFs (e.g. 8h), fetch from base TF and resample.
        """
        # Handle non-native TFs via resample
        if timeframe in RESAMPLE_FROM:
            base_tf, pandas_rule = RESAMPLE_FROM[timeframe]
            # Need ~2x candles in base TF for 8h (2 × 4h = 1 × 8h)
            base_count = total_candles * 2 + 50
            base_df = self.fetch_history(symbol, base_tf, total_candles=base_count)
            if base_df.empty:
                return base_df
            resampled = base_df.resample(pandas_rule, label="left", closed="left").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"
            }).dropna()
            return resampled.tail(total_candles)

        period = TIMEFRAMES[timeframe]
        all_dfs = []
        end_ts = int(time.time())
        remaining = total_candles
        max_pages = 20

        for page in range(max_pages):
            chunk = min(500, remaining)
            df = self.fetch_ohlcv(symbol, timeframe, limit=chunk, end_ts=end_ts)
            if df.empty:
                break
            all_dfs.append(df)
            remaining -= len(df)
            if remaining <= 0:
                break
            end_ts = int(df.iloc[0]["ts"] if "ts" in df.columns else df.index[0].timestamp()) - period
            time.sleep(0.2)  # rate-limit friendly

        if not all_dfs:
            return pd.DataFrame()
        full = pd.concat(all_dfs).sort_index()
        full = full[~full.index.duplicated(keep="first")]
        return full.tail(total_candles)
