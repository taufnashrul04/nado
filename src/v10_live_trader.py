"""
Live Trader V10 — Multi-Layer AI (Stoch+EMA+MTF+TV+LLM)
========================================================
Settings (per user spec):
  - Max concurrent: 3 positions
  - Margin per trade: $0.50 (test mode)
  - Leverage: MAX per symbol (BTC 50x, SOL 40x, BNB 20x, DOGE 10x)
  - TP @ +100% ROE → close 75%, trail rest with 0.4 ROE distance
  - SL @ -70% ROE (safety buffer before liq at -100%)
  - Time stop: max 48h hold (12 bars × 4h)

Modes:
  - paper_trade=True: simulate fills, log to file
  - paper_trade=False: place real orders via Nado SDK

Risk math (margin $0.50, lev 50x = $25 notional):
  - Win @ TP partial = +$0.375 (75% of +$0.50 ROE 100%)
  - Loss @ SL = -$0.35 (-70%)
  - Liq buffer @ -100% = -$0.50 max (1% adverse for lev 50x)

Strategy: V10 multi-layer
  Layer 1: Stoch(5,3,3) + EMA(13/21/34/50/90/200) confluence
  Layer 2: MTF (4h EMA stack + Daily SMA trend + ADX>20)
  Layer 3: TradingView consensus (BUY/SELL/NEUTRAL)
  Layer 4: LLM analyst (Claude Opus 4.7) synthesis
  Veto chain: any layer disagree → SKIP
"""
from __future__ import annotations
import sys, os, json, time, traceback
sys.path.insert(0, '/root/nado-bot/src')
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

from data.fetcher import NadoDataFetcher
from data.indicators import add_indicators
from strategies.ai_multilayer_v10 import signal_fn, get_stats, reset_stats


CONFIG_PATH = Path("/root/nado-bot/config/v10_live.json")
STATE_PATH = Path("/root/nado-bot/state/v10_state.json")
LOG_PATH = Path("/root/nado-bot/logs/v10_trader.log")
TRADE_LOG = Path("/root/nado-bot/logs/v10_trades.jsonl")

for p in [CONFIG_PATH.parent, STATE_PATH.parent, LOG_PATH.parent]:
    p.mkdir(parents=True, exist_ok=True)


def log(msg, level="INFO"):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_config():
    return json.loads(CONFIG_PATH.read_text())


def load_state():
    if not STATE_PATH.exists():
        return {
            "open_positions": [],
            "closed_today": [],
            "daily_pnl_usd": 0.0,
            "daily_reset_ts": datetime.now(timezone.utc).isoformat(),
            "last_check_ts": None,
            "total_pnl_usd": 0.0,
            "total_trades": 0,
            "total_wins": 0,
        }
    return json.loads(STATE_PATH.read_text())


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def telegram_send(token: str, chat_id: str, text: str):
    if not token or token == "":
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        log(f"Telegram send failed: {e}", "WARN")


def reset_daily_if_needed(state):
    last_reset = datetime.fromisoformat(state["daily_reset_ts"]).astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    if (now - last_reset) >= timedelta(hours=24):
        state["daily_pnl_usd"] = 0.0
        state["daily_reset_ts"] = now.isoformat()
        state["closed_today"] = []
        log("Daily counters reset")


def calc_pnl_usd(pos, exit_price):
    """ROE-based PnL. Returns USD pnl on margin."""
    entry = pos["entry_price"]
    margin = pos["remaining_margin_usd"]
    lev = pos["leverage"]
    direction = 1 if pos["side"] == "long" else -1
    price_pct = direction * (exit_price - entry) / entry
    roe = price_pct * lev  # ROE on margin
    pnl_usd = margin * roe
    # Approximate fee: 0.05% taker × 2 legs × notional
    fee = 2 * 0.0005 * margin * lev
    return pnl_usd - fee, roe


def update_open_positions(state, fetcher, cfg, telegram):
    """Per-bar update: check TP partial, SL, trail, time_stop."""
    still_open = []
    for pos in state["open_positions"]:
        try:
            df = fetcher.fetch_ohlcv(pos["symbol"], cfg["timeframe"], limit=2)
            if df is None or len(df) == 0:
                still_open.append(pos)
                continue
            cur_price = float(df["close"].iloc[-1])
            cur_high = float(df["high"].iloc[-1])
            cur_low = float(df["low"].iloc[-1])

            entry = pos["entry_price"]
            lev = pos["leverage"]
            direction = 1 if pos["side"] == "long" else -1

            # Calculate ROE at high/low/cur
            def price_to_roe(px):
                return direction * (px - entry) / entry * lev

            roe_cur = price_to_roe(cur_price)
            roe_high = price_to_roe(cur_high if pos["side"] == "long" else cur_low)
            roe_low = price_to_roe(cur_low if pos["side"] == "long" else cur_high)

            tp_roe = cfg["tp_roe_pct"]
            sl_roe = -cfg["sl_roe_pct"]
            trail_activate = cfg["trail_activate_roe_pct"]
            trail_distance = cfg["trail_distance_roe_pct"]

            # ===== TP PARTIAL CLOSE @ tp_roe (only first time) =====
            if not pos.get("tp_hit") and roe_high >= tp_roe:
                close_pct = cfg["tp_partial_close_pct"]
                closed_margin = pos["remaining_margin_usd"] * close_pct
                # Realize partial PnL at tp_roe ROE
                partial_pnl = closed_margin * tp_roe - 2 * 0.0005 * closed_margin * lev
                pos["realized_pnl_usd"] = pos.get("realized_pnl_usd", 0) + partial_pnl
                pos["remaining_margin_usd"] *= (1 - close_pct)
                pos["tp_hit"] = True
                pos["peak_roe"] = max(pos.get("peak_roe", 0), roe_high)
                state["daily_pnl_usd"] += partial_pnl

                msg = (f"🎯 *TP HIT* {pos['symbol']} {pos['side'].upper()}\n"
                       f"Closed 75% @ ROE +{tp_roe*100:.0f}% = +${partial_pnl:+.2f}\n"
                       f"Trailing rest 25% (margin ${pos['remaining_margin_usd']:.2f})\n"
                       f"Daily: ${state['daily_pnl_usd']:+.2f}")
                log(f"TP_PARTIAL {pos['symbol']} closed75% pnl=${partial_pnl:+.2f}")
                telegram(msg)
                still_open.append(pos)
                continue

            # ===== TRAILING STOP (after TP hit) =====
            if pos.get("tp_hit"):
                pos["peak_roe"] = max(pos.get("peak_roe", 0), roe_high)
                trail_stop_roe = pos["peak_roe"] - trail_distance
                if roe_low <= trail_stop_roe:
                    # Close remainder at trail price
                    trail_price = entry * (1 + (direction * trail_stop_roe / lev))
                    pnl, roe = calc_pnl_usd(pos, trail_price)
                    pos["realized_pnl_usd"] += pnl
                    state["daily_pnl_usd"] += pnl
                    closed = {**pos, "exit_ts": datetime.now(timezone.utc).isoformat(),
                             "exit_price": trail_price, "exit_reason": "trail",
                             "final_pnl_usd": pos["realized_pnl_usd"]}
                    state["closed_today"].append(closed)
                    state["total_pnl_usd"] += pos["realized_pnl_usd"]
                    state["total_trades"] += 1
                    if pos["realized_pnl_usd"] > 0:
                        state["total_wins"] += 1
                    with open(TRADE_LOG, "a") as f:
                        f.write(json.dumps(closed, default=str) + "\n")

                    msg = (f"🟢 *TRAIL EXIT* {pos['symbol']} {pos['side'].upper()}\n"
                           f"Total realized: ${pos['realized_pnl_usd']:+.2f}\n"
                           f"Peak ROE: {pos['peak_roe']*100:.0f}% → trail @ {trail_stop_roe*100:.0f}%\n"
                           f"Daily: ${state['daily_pnl_usd']:+.2f}")
                    log(f"TRAIL_EXIT {pos['symbol']} total=${pos['realized_pnl_usd']:+.2f}")
                    telegram(msg)
                    continue

            # ===== HARD SL =====
            if roe_low <= sl_roe:
                sl_price = entry * (1 + (direction * sl_roe / lev))
                pnl, roe = calc_pnl_usd(pos, sl_price)
                pos["realized_pnl_usd"] = pos.get("realized_pnl_usd", 0) + pnl
                state["daily_pnl_usd"] += pnl
                closed = {**pos, "exit_ts": datetime.now(timezone.utc).isoformat(),
                         "exit_price": sl_price, "exit_reason": "sl",
                         "final_pnl_usd": pos["realized_pnl_usd"]}
                state["closed_today"].append(closed)
                state["total_pnl_usd"] += pos["realized_pnl_usd"]
                state["total_trades"] += 1
                with open(TRADE_LOG, "a") as f:
                    f.write(json.dumps(closed, default=str) + "\n")

                msg = (f"🔴 *SL HIT* {pos['symbol']} {pos['side'].upper()}\n"
                       f"Total realized: ${pos['realized_pnl_usd']:+.2f} (ROE {sl_roe*100:.0f}%)\n"
                       f"Daily: ${state['daily_pnl_usd']:+.2f}")
                log(f"SL_EXIT {pos['symbol']} total=${pos['realized_pnl_usd']:+.2f}")
                telegram(msg)
                continue

            # ===== TIME STOP =====
            time_stop = datetime.fromisoformat(pos["time_stop_ts"]).astimezone(timezone.utc)
            if datetime.now(timezone.utc) >= time_stop:
                pnl, roe = calc_pnl_usd(pos, cur_price)
                pos["realized_pnl_usd"] = pos.get("realized_pnl_usd", 0) + pnl
                state["daily_pnl_usd"] += pnl
                closed = {**pos, "exit_ts": datetime.now(timezone.utc).isoformat(),
                         "exit_price": cur_price, "exit_reason": "time_stop",
                         "final_pnl_usd": pos["realized_pnl_usd"]}
                state["closed_today"].append(closed)
                state["total_pnl_usd"] += pos["realized_pnl_usd"]
                state["total_trades"] += 1
                if pos["realized_pnl_usd"] > 0:
                    state["total_wins"] += 1
                with open(TRADE_LOG, "a") as f:
                    f.write(json.dumps(closed, default=str) + "\n")

                emoji = "🟢" if pos["realized_pnl_usd"] > 0 else "🔴"
                msg = (f"{emoji} *TIME STOP* {pos['symbol']} {pos['side'].upper()}\n"
                       f"Held 48h, exit @ {cur_price:.4g}\n"
                       f"Total realized: ${pos['realized_pnl_usd']:+.2f}\n"
                       f"Daily: ${state['daily_pnl_usd']:+.2f}")
                log(f"TIMESTOP_EXIT {pos['symbol']} total=${pos['realized_pnl_usd']:+.2f}")
                telegram(msg)
                continue

            still_open.append(pos)
        except Exception as e:
            log(f"Error updating {pos.get('symbol')}: {e}", "ERROR")
            still_open.append(pos)

    state["open_positions"] = still_open


def scan_signals(state, fetcher, cfg, telegram):
    """Scan all symbols for V9 AI signal. Open if found."""
    if len(state["open_positions"]) >= cfg["max_concurrent"]:
        log(f"Max concurrent ({cfg['max_concurrent']}) reached, skip scan")
        return

    for sym in cfg["symbols"]:
        if any(p["symbol"] == sym for p in state["open_positions"]):
            continue
        if len(state["open_positions"]) >= cfg["max_concurrent"]:
            break

        try:
            df = fetcher.fetch_history(sym, cfg["timeframe"], total_candles=cfg["candles_lookback"])
            if df is None or len(df) < 200:
                continue
            df = add_indicators(df)
            df.attrs['symbol'] = sym

            i = len(df) - 1
            sig = signal_fn(df, i)
            if sig is None:
                continue

            row = df.iloc[i]
            entry_price = float(row["close"])
            side = "long" if sig["action"] == "open_long" else "short"
            lev = cfg["leverage_per_symbol"].get(sym, cfg["default_leverage"])
            margin_usd = cfg["margin_per_trade_usd"]
            tf_hours = 8 if cfg["timeframe"] == "8h" else 4
            max_bars = cfg.get("max_bars_hold", 6)
            time_stop = (datetime.now(timezone.utc) + timedelta(hours=max_bars * tf_hours)).isoformat()

            pos = {
                "symbol": sym, "side": side,
                "entry_ts": datetime.now(timezone.utc).isoformat(),
                "entry_price": entry_price,
                "margin_usd": margin_usd,
                "remaining_margin_usd": margin_usd,
                "leverage": lev,
                "notional_usd": margin_usd * lev,
                "tp_roe": cfg["tp_roe_pct"],
                "sl_roe": -cfg["sl_roe_pct"],
                "tp_hit": False,
                "peak_roe": 0,
                "realized_pnl_usd": 0,
                "time_stop_ts": time_stop,
                "reason": sig["reason"], "paper": cfg.get("paper_trade", True),
            }
            state["open_positions"].append(pos)
            with open(TRADE_LOG, "a") as f:
                f.write(json.dumps({"event": "open", **pos}, default=str) + "\n")

            mode = "📝 PAPER" if cfg.get("paper_trade", True) else "💸 LIVE"
            emoji = "📈" if side == "long" else "📉"
            tp_pct = cfg["tp_roe_pct"] / lev
            sl_pct = cfg["sl_roe_pct"] / lev
            tp_price = entry_price * (1 + (1 if side == "long" else -1) * tp_pct)
            sl_price = entry_price * (1 - (1 if side == "long" else -1) * sl_pct)
            msg = (f"{mode} {emoji} *OPEN* {sym} {side.upper()}\n"
                   f"Entry: {entry_price:.4g} | Lev: {lev}x | Margin: ${margin_usd}\n"
                   f"Notional: ${margin_usd*lev:.0f}\n"
                   f"TP +{cfg['tp_roe_pct']*100:.0f}% ROE: {tp_price:.4g} (close 75%, trail rest)\n"
                   f"SL {-cfg['sl_roe_pct']*100:.0f}% ROE: {sl_price:.4g}\n"
                   f"Reason: {sig['reason'][:120]}")
            log(f"OPEN {mode} {sym} {side} entry={entry_price:.4g} lev={lev}x")
            telegram(msg)

            # TODO real Nado place_order if not paper_trade
            if not cfg.get("paper_trade", True):
                log("LIVE MODE not yet wired — set paper_trade=true for now", "WARN")

            return  # one open per cycle
        except Exception as e:
            log(f"Error scanning {sym}: {e}\n{traceback.format_exc()[:400]}", "ERROR")


def check_circuit_breaker(state, cfg, balance_usd, telegram):
    threshold_usd = -balance_usd * cfg["daily_loss_circuit_pct"]
    if state["daily_pnl_usd"] <= threshold_usd:
        if not cfg.get("halt"):
            cfg["halt"] = True
            CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
            telegram(f"🛑 *CIRCUIT BREAKER*\n"
                    f"Daily loss: ${state['daily_pnl_usd']:.2f} (threshold ${threshold_usd:.2f})\n"
                    f"Halt 24h. Edit `halt: false` in config to resume.")
            log(f"CIRCUIT BREAKER: daily=${state['daily_pnl_usd']:.2f}", "WARN")
            return True
    return False


def status_summary(state, cfg):
    wr = (state["total_wins"] / state["total_trades"] * 100) if state["total_trades"] > 0 else 0
    s = get_stats()
    return (f"📊 *V10 Status*\n"
            f"Open: {len(state['open_positions'])}/{cfg['max_concurrent']}\n"
            f"Daily PnL: ${state['daily_pnl_usd']:+.4f}\n"
            f"Total: {state['total_trades']} trades, WR {wr:.1f}%, ${state['total_pnl_usd']:+.4f}\n"
            f"Stats: cand={s['candidates']} veto_mtf={s['veto_mtf']} veto_tv={s['veto_tv']} "
            f"veto_ai={s['veto_ai']} long={s['long']} short={s['short']}")


def main_loop():
    state = load_state()
    fetcher = NadoDataFetcher()

    # Load secrets
    for envfile in ["/root/nado-bot/secrets/llm.env", "/root/nado-bot/secrets/telegram.env"]:
        if Path(envfile).exists():
            for line in Path(envfile).read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")

    cfg = load_config()
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram = lambda msg: telegram_send(telegram_token, cfg["telegram_chat_id"], msg)

    mode = "📝 PAPER" if cfg.get("paper_trade", True) else "💸 LIVE"
    telegram(f"🤖 *V10 Bot Started* ({mode})\n"
            f"Symbols: {', '.join(cfg['symbols'])}\n"
            f"Margin: ${cfg['margin_per_trade_usd']}/trade × LEV MAX\n"
            f"BTC: 50x | SOL: 40x | BNB: 20x | DOGE: 10x\n"
            f"TP +{cfg['tp_roe_pct']*100:.0f}% ROE (close 75% + trail)\n"
            f"SL -{cfg['sl_roe_pct']*100:.0f}% ROE | Max concurrent: {cfg['max_concurrent']}\n"
            f"Strategy: V10 Multi-Layer (Stoch+EMA+MTF+TV+LLM)")
    log(f"V10 bot started in {mode} mode, lev max, margin ${cfg['margin_per_trade_usd']}")

    cycle = 0
    while True:
        try:
            cfg = load_config()
            if cfg.get("halt"):
                log("Halted by config, sleep 60s")
                time.sleep(60)
                continue

            reset_daily_if_needed(state)
            update_open_positions(state, fetcher, cfg, telegram)
            scan_signals(state, fetcher, cfg, telegram)
            check_circuit_breaker(state, cfg, 130.0, telegram)
            state["last_check_ts"] = datetime.now(timezone.utc).isoformat()
            save_state(state)

            cycle += 1
            if cycle % 24 == 0:  # every ~12h send status
                telegram(status_summary(state, cfg))
        except Exception as e:
            log(f"Loop err: {e}\n{traceback.format_exc()[:500]}", "ERROR")

        time.sleep(cfg.get("check_interval_minutes", 30) * 60)


if __name__ == "__main__":
    main_loop()
