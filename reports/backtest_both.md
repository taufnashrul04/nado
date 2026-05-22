# Nado Backtest Report — Strategy BOTH

**Timeframe:** 1h, 4000 candles  
**Initial:** $130.0 | Size: 5.0% | Leverage: 3.0x

## Per-Symbol Summary

| Symbol | Strategy | Trades | WR% | Return% | MaxDD% | Sharpe | PF | EndBal$ |
|---|---|---|---|---|---|---|---|---|
| BTC-PERP | v1 | 88 | 37.5 | -1.74 | -4.00 | -1.10 | 0.83 | 127.74 |
| BTC-PERP | v2 | 1 | 100.0 | +0.14 | 0.00 | 1.48 | ∞ | 130.18 |
| ETH-PERP | v1 | 72 | 36.1 | -0.10 | -2.79 | -0.03 | 0.99 | 129.87 |
| ETH-PERP | v2 | 2 | 100.0 | +0.29 | -0.14 | 1.79 | ∞ | 130.38 |
| SOL-PERP | v1 | 67 | 31.3 | -3.70 | -5.57 | -2.06 | 0.7 | 125.18 |
| SOL-PERP | v2 | 1 | 100.0 | +0.11 | 0.00 | 1.48 | ∞ | 130.14 |
| WTI-PERP | v1 | 22 | 27.3 | -1.92 | -2.81 | -2.61 | 0.59 | 127.51 |
| WTI-PERP | v2 | 4 | 75.0 | +0.22 | -0.40 | 1.22 | 1.54 | 130.28 |
| SPY-PERP | v1 | 11 | 45.5 | +0.13 | -0.35 | 1.32 | 1.25 | 130.17 |
| SPY-PERP | v2 | 1 | 0.0 | -0.09 | -0.09 | -3.10 | 0.0 | 129.88 |
| QQQ-PERP | v1 | 8 | 25.0 | -0.26 | -0.41 | -2.46 | 0.51 | 129.66 |
| QQQ-PERP | v2 | 0 | 0.0 | +0.00 | 0.00 | 0.00 | ∞ | 130.00 |
| NVDA-PERP | v1 | 13 | 30.8 | +0.05 | -0.49 | 0.31 | 1.04 | 130.06 |
| NVDA-PERP | v2 | 2 | 50.0 | -0.14 | -0.23 | -2.75 | 0.37 | 129.82 |
| TSLA-PERP | v1 | 11 | 36.4 | -0.05 | -0.65 | -0.22 | 0.95 | 129.94 |
| TSLA-PERP | v2 | 0 | 0.0 | +0.00 | 0.00 | 0.00 | ∞ | 130.00 |
| AAPL-PERP | v1 | 9 | 44.4 | +0.19 | -0.32 | 1.53 | 1.4 | 130.25 |
| AAPL-PERP | v2 | 0 | 0.0 | +0.00 | 0.00 | 0.00 | ∞ | 130.00 |
| GOOGL-PERP | v1 | 6 | 16.7 | -0.41 | -0.45 | -3.89 | 0.31 | 129.47 |
| GOOGL-PERP | v2 | 1 | 0.0 | -0.10 | -0.10 | -3.46 | 0.0 | 129.87 |

## LLM Analysis

## Analisis Backtest

**Catatan:** Tiap symbol ada 2 baris (kemungkinan LONG vs SHORT terpisah). Gue gabungin buat analisa keseluruhan, tapi sample size kecil (1-4 trades) gue treat sebagai noise, bukan sinyal.

### 1. Best 3 Performers (yang lumayan)
- **AAPL-PERP**: WR 44.4%, return +0.19%, PF 1.4, Sharpe 1.53. Paling konsisten.
- **SPY-PERP**: WR 45.5%, return +0.13%, PF 1.25. Index lebih jinak, cocok sama strategi trend-following.
- **NVDA-PERP**: WR ~33%, return +0.05% (nyaris flat). Marginal, masuk top 3 cuma karena yang lain lebih parah.

### 2. Hindari (worst performers)
- **SOL-PERP**: -3.70%, Sharpe -2.06, PF 0.70, 67 trades. Statistically signifikan jeleknya.
- **WTI-PERP**: -1.92%, Sharpe -2.61, PF 0.59. Oil punya gap & news shock, ATR-based stop kurang nampol.
- **GOOGL-PERP**: WR 16.7%, PF 0.31. Sample kecil tapi rasionya ngeri.
- **BTC-PERP**: -1.74% di 88 trades, PF 0.83. Sample paling besar, jadi paling reliable signal nya, dan signalnya: rugi.

### 3. Layak live? **TIDAK.**
- Aggregate return negatif di mayoritas symbol dengan sample memadai (BTC, ETH, SOL, WTI).
- Win rate 30-37% dengan R:R 2:1 secara teori break-even di 33%, tapi setelah fee + slippage + funding, ini auto-rugi.
- Sharpe negatif di 6 dari 10 symbol. PF < 1 di mayoritas. Belum diuji out-of-sample.
- 62 hari = sample period kependekan, satu regime market doang.

### 4. Saran Improvement
- **Filter regime**: cek ADX threshold dinaikin ke 25-30, atau tambah filter volatility (ATR percentile).
- **Fix R:R**: TP 3x ATR di crypto sering kena retrace duluan. Coba trailing stop atau partial TP di 1.5x ATR.
- **Drop symbol jelek**: SOL, WTI, GOOGL keluarin dari universe.
- **Pisahin LONG vs SHORT stats** dan optimasi terpisah, struktur market beda.
- **Walk-forward test** minimal 6-12 bulan sebelum mikirin live.

Bottom line: strategi mentah ini **belum siap**. Balik ke meja riset dulu.
