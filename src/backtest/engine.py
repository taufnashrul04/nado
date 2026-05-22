"""
Backtest engine — simulates strategies on historical OHLCV.
Tracks trades, PnL, drawdown, Sharpe, etc.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

Side = Literal["long", "short", "flat"]


@dataclass
class Trade:
    entry_ts: pd.Timestamp
    side: Side
    entry_price: float
    size_usd: float       # notional in USDT
    sl: float
    tp: float
    leverage: float
    entry_idx: int = -1
    max_bars_hold: int = 0  # 0 = no time stop
    exit_ts: pd.Timestamp | None = None
    exit_price: float | None = None
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""

    @property
    def is_open(self) -> bool:
        return self.exit_ts is None

    def close(self, ts: pd.Timestamp, price: float, reason: str, fee_pct: float = 0.0005):
        self.exit_ts = ts
        self.exit_price = price
        if self.side == "long":
            raw_ret = (price - self.entry_price) / self.entry_price
        else:
            raw_ret = (self.entry_price - price) / self.entry_price
        # Apply leverage to return, deduct round-trip fees
        net_ret = raw_ret * self.leverage - 2 * fee_pct * self.leverage
        self.pnl_pct = net_ret
        self.pnl_usd = self.size_usd * net_ret
        self.reason = reason


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    start_balance: float
    end_balance: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series | None = None

    @property
    def total_return_pct(self) -> float:
        return (self.end_balance / self.start_balance - 1) * 100

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl_usd > 0)
        return wins / len(self.trades) * 100

    @property
    def avg_win(self) -> float:
        wins = [t.pnl_usd for t in self.trades if t.pnl_usd > 0]
        return np.mean(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl_usd for t in self.trades if t.pnl_usd < 0]
        return np.mean(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl_usd for t in self.trades if t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in self.trades if t.pnl_usd < 0))
        return gross_win / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown_pct(self) -> float:
        if self.equity_curve is None or self.equity_curve.empty:
            return 0.0
        running_max = self.equity_curve.cummax()
        dd = (self.equity_curve - running_max) / running_max
        return float(dd.min() * 100)

    @property
    def sharpe(self) -> float:
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return 0.0
        returns = self.equity_curve.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        # Annualize assuming hourly bars (24*365=8760)
        return float(returns.mean() / returns.std() * np.sqrt(8760))

    def summary(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_balance": self.start_balance,
            "end_balance": round(self.end_balance, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "num_trades": self.num_trades,
            "win_rate": round(self.win_rate, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 2) if self.profit_factor != float("inf") else "∞",
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe": round(self.sharpe, 2),
        }


class Backtester:
    def __init__(self, symbol: str, timeframe: str,
                 initial_balance: float = 1000.0,
                 size_per_trade_pct: float = 0.05,
                 leverage: float = 3.0,
                 fee_pct: float = 0.0005,
                 max_concurrent: int = 1):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.size_pct = size_per_trade_pct
        self.leverage = leverage
        self.fee_pct = fee_pct
        self.max_concurrent = max_concurrent

    def run(self, df: pd.DataFrame, signal_fn: Callable[[pd.DataFrame, int], dict | None]) -> BacktestResult:
        """
        signal_fn(df, i) → dict | None
        dict keys: action ('open_long'|'open_short'|'close'),
                   sl_pct, tp_pct, reason

        df must already have indicators applied.
        """
        balance = self.initial_balance
        trades: list[Trade] = []
        open_trades: list[Trade] = []
        equity = []

        for i in range(len(df)):
            row = df.iloc[i]
            price = row["close"]
            ts = df.index[i]

            # Check open trades for SL/TP
            still_open = []
            for t in open_trades:
                exit_price = None
                reason = None
                # Use next bar's high/low for realistic SL/TP fill
                hi = row["high"]
                lo = row["low"]
                if t.side == "long":
                    if lo <= t.sl:
                        exit_price = t.sl
                        reason = "sl"
                    elif hi >= t.tp:
                        exit_price = t.tp
                        reason = "tp"
                else:  # short
                    if hi >= t.sl:
                        exit_price = t.sl
                        reason = "sl"
                    elif lo <= t.tp:
                        exit_price = t.tp
                        reason = "tp"

                # Time stop check (only if no SL/TP hit and max_bars_hold set)
                if exit_price is None and t.max_bars_hold > 0 and t.entry_idx >= 0:
                    if (i - t.entry_idx) >= t.max_bars_hold:
                        exit_price = price
                        reason = "time_stop"

                if exit_price is not None:
                    t.close(ts, exit_price, reason, self.fee_pct)
                    balance += t.pnl_usd
                else:
                    still_open.append(t)
            open_trades = still_open

            # Check for new signal (only if not already at max concurrent)
            sig = signal_fn(df, i)
            if sig and len(open_trades) < self.max_concurrent:
                act = sig.get("action")
                if act == "open_long":
                    sl_pct = sig.get("sl_pct", 0.02)
                    tp_pct = sig.get("tp_pct", 0.04)
                    max_bars_hold = sig.get("max_bars_hold", 0)
                    size = balance * self.size_pct
                    t = Trade(
                        entry_ts=ts, side="long", entry_price=price,
                        size_usd=size, sl=price * (1 - sl_pct),
                        tp=price * (1 + tp_pct), leverage=self.leverage,
                        entry_idx=i, max_bars_hold=max_bars_hold,
                    )
                    open_trades.append(t)
                    trades.append(t)
                elif act == "open_short":
                    sl_pct = sig.get("sl_pct", 0.02)
                    tp_pct = sig.get("tp_pct", 0.04)
                    max_bars_hold = sig.get("max_bars_hold", 0)
                    size = balance * self.size_pct
                    t = Trade(
                        entry_ts=ts, side="short", entry_price=price,
                        size_usd=size, sl=price * (1 + sl_pct),
                        tp=price * (1 - tp_pct), leverage=self.leverage,
                        entry_idx=i, max_bars_hold=max_bars_hold,
                    )
                    open_trades.append(t)
                    trades.append(t)
                elif act == "close":
                    for t in open_trades:
                        t.close(ts, price, sig.get("reason", "signal_close"), self.fee_pct)
                        balance += t.pnl_usd
                    open_trades = []

            equity.append(balance + sum(
                t.size_usd * (
                    ((row["close"] - t.entry_price) / t.entry_price * t.leverage)
                    if t.side == "long"
                    else ((t.entry_price - row["close"]) / t.entry_price * t.leverage)
                )
                for t in open_trades
            ))

        # Force-close remaining at last price
        for t in open_trades:
            t.close(df.index[-1], df.iloc[-1]["close"], "end_of_data", self.fee_pct)
            balance += t.pnl_usd

        return BacktestResult(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_balance=self.initial_balance,
            end_balance=balance,
            trades=trades,
            equity_curve=pd.Series(equity, index=df.index),
        )
