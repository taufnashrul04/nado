# Nado Backtest Report

**Strategy:** Stoch(5,3,3) + EMA(13,21,34,50,90,200)  
**Timeframe:** 1h, 1500 candles  
**Initial:** $130.0 | Size: 5.0% | Leverage: 3.0x

## Per-Symbol Summary

| Symbol | Trades | WinRate% | Return% | MaxDD% | Sharpe | ProfitFactor | EndBal$ |
|---|---|---|---|---|---|---|---|
| BTC-PERP | 32 | 31.2 | -1.40 | -2.05 | -3.08 | 0.58 | 128.18 |
| ETH-PERP | 23 | 30.4 | -1.10 | -1.57 | -2.22 | 0.62 | 128.57 |
| SOL-PERP | 24 | 25.0 | -1.62 | -2.18 | -3.25 | 0.55 | 127.89 |
| WTI-PERP | 22 | 27.3 | -1.92 | -2.81 | -2.61 | 0.59 | 127.51 |
| SPY-PERP | 11 | 45.5 | +0.13 | -0.35 | 1.32 | 1.25 | 130.17 |
| QQQ-PERP | 8 | 25.0 | -0.26 | -0.41 | -2.46 | 0.51 | 129.66 |
| NVDA-PERP | 13 | 30.8 | +0.05 | -0.49 | 0.31 | 1.04 | 130.06 |
| TSLA-PERP | 11 | 36.4 | -0.05 | -0.65 | -0.22 | 0.95 | 129.94 |
| AAPL-PERP | 9 | 44.4 | +0.19 | -0.32 | 1.53 | 1.4 | 130.25 |
| GOOGL-PERP | 6 | 16.7 | -0.41 | -0.45 | -3.89 | 0.31 | 129.47 |

## LLM Analysis

# Analisis Backtest Strategi Stoch + EMA Stack

## 1. Top 3 Performers
- **AAPL-PERP**: WR 44.4%, return +0.19%, Sharpe 1.53, PF 1.4 ✅
- **SPY-PERP**: WR 45.5%, return +0.13%, Sharpe 1.32, PF 1.25 ✅
- **NVDA-PERP**: WR 30.8%, return +0.05%, Sharpe 0.31, PF 1.04 ⚠️ (marginal)

Pattern jelas: strategi ini cuma jalan di **equity/index yang trending halus**, bukan crypto/komoditas.

## 2. Hindari
- **GOOGL-PERP**: WR 16.7%, PF 0.31 — bencana, sample size kecil (6 trades) bikin makin gak reliable
- **SOL-PERP, WTI-PERP, BTC-PERP**: Sharpe < -3, PF di bawah 0.6 — strategi keganggu sama volatility tinggi
- Crypto secara umum (BTC/ETH/SOL): semua negatif, WR ~25-31%

## 3. Layak Live? **TIDAK.**
- 7 dari 10 symbol rugi, cuma 3 yang profit tipis
- WR rata-rata ~31% padahal R:R 2:1 butuh minimum WR ~35% buat breakeven (belum termasuk fees + funding)
- Sample size kekecilan: AAPL cuma 9 trades, SPY 11 trades — Sharpe positif di sini **belum signifikan secara statistik**
- 62 hari backtest terlalu pendek, gak cover regime change
- PF > 1 cuma 3 symbol, dan tipis banget (1.04-1.4)

## 4. Improvement
- **Filter regime**: tambahin ADX threshold lebih tinggi (>25) atau filter pakai market regime detector — entry cuma pas trending kuat
- **Asymmetric R:R**: coba TP 2x ATR (1.33:1) buat naikin WR, current setup kebanyakan kena SL sebelum TP
- **Drop crypto & komoditas** dari universe, fokus equity index
- **Re-test minimum 6-12 bulan** data buat dapet sample 100+ trades per symbol
- **Cek slippage & funding cost** di leverage 3x — backtest ini kayaknya belum include
- Eksplor exit dinamis (trailing stop berbasis EMA21) daripada fixed TP

**Verdict: paper trade dulu, jangan live.**
