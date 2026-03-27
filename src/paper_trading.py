"""
Phase 6 -- Paper Trading Engine
Runs every hour, fetches live data, generates signals, places demo orders.

Usage:
    python src/paper_trading.py

Runs indefinitely. Press Ctrl+C to stop.
"""

import os
import sys
import time
import json
import logging
import schedule
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.features import add_all_features, FEATURE_COLS
from src.signals  import TradingSignalEngine
from src.executor import PaperExecutor

# ── Paths ──────────────────────────────────────────────────
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

MODELS_DIR    = os.path.join(PROJECT_ROOT, "data", "models")
RL_DIR        = os.path.join(PROJECT_ROOT, "data", "rl")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LOGS_DIR      = os.path.join(PROJECT_ROOT, "logs")
TRADES_LOG    = os.path.join(LOGS_DIR, "trades.csv")
STATE_PATH    = os.path.join(LOGS_DIR, "state.json")
os.makedirs(LOGS_DIR, exist_ok=True)

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    handlers= [
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(LOGS_DIR, "paper_trading.log"),
            encoding="utf-8"   # explicit utf-8 to avoid Windows cp1252 issue
        ),
    ]
)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────
ASSETS          = ["btc", "eth"]
INITIAL_CAPITAL = 10_000.0
MAX_DRAWDOWN    = 0.15

SYMBOLS_CCXT = {
    "btc": "BTC/USDT:USDT",
    "eth": "ETH/USDT:USDT",
}

# ── State ───────────────────────────────────────────────────
state = {
    "btc": {
        "position"    : 0.0,
        "size"        : 0.0,
        "entry_price" : 0.0,
        "hold_hours"  : 0,
        "capital"     : INITIAL_CAPITAL / 2,
        "peak_capital": INITIAL_CAPITAL / 2,
    },
    "eth": {
        "position"    : 0.0,
        "size"        : 0.0,
        "entry_price" : 0.0,
        "hold_hours"  : 0,
        "capital"     : INITIAL_CAPITAL / 2,
        "peak_capital": INITIAL_CAPITAL / 2,
    }
}


def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df  = pd.DataFrame(
        raw, columns=["timestamp","open","high","low","close","volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def load_state() -> None:
    if not os.path.exists(STATE_PATH):
        return

    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            saved_state = json.load(f)

        for asset in ASSETS:
            if asset not in saved_state:
                continue
            for key in state[asset]:
                if key in saved_state[asset]:
                    state[asset][key] = saved_state[asset][key]

        logger.info("Loaded persisted bot state from disk.")
    except Exception as e:
        logger.warning(f"Failed to load saved state: {e}")


def save_state() -> None:
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save bot state: {e}")


def log_trade(asset: str, action: str, price: float,
              size: float, capital: float, signal: dict):
    row = {
        "timestamp" : datetime.now().isoformat(),
        "asset"     : asset.upper(),
        "action"    : action,
        "price"     : round(price, 4),
        "size_pct"  : size,
        "capital"   : round(capital, 2),
        "prob_buy"  : round(signal["prob_buy"],  4),
        "prob_sell" : round(signal["prob_sell"], 4),
        "prob_hold" : round(signal["prob_hold"], 4),
        "raw_action": signal["raw_action"],
    }
    df_row = pd.DataFrame([row])
    header = (not os.path.exists(TRADES_LOG)) or os.path.getsize(TRADES_LOG) == 0
    df_row.to_csv(TRADES_LOG, mode="a", header=header, index=False)


def check_drawdown(asset: str) -> bool:
    s        = state[asset]
    drawdown = (s["capital"] - s["peak_capital"]) / s["peak_capital"]
    if drawdown < -MAX_DRAWDOWN:
        logger.warning(
            f"DRAWDOWN LIMIT HIT for {asset.upper()}: "
            f"{drawdown*100:.1f}% -- bot paused for this asset"
        )
        return False
    return True


def order_succeeded(order: dict) -> bool:
    if not order:
        return False

    status = str(order.get("status", "")).lower()
    return status in {"open", "closed", "filled"} or order.get("id") is not None


def extract_order_fill(order: dict, fallback_price: float) -> tuple[float, float]:
    filled_size = float(order.get("filled") or order.get("amount") or 0.0)
    avg_price = float(order.get("average") or order.get("price") or fallback_price)
    return filled_size, avg_price


def run_trading_cycle(engine: TradingSignalEngine,
                      executor: PaperExecutor,
                      exchange) -> None:

    logger.info("=" * 55)
    logger.info(
        f"Trading cycle started at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    for asset in ASSETS:
        logger.info(f"\n--- {asset.upper()} ---")
        s      = state[asset]
        symbol = SYMBOLS_CCXT[asset]

        try:
            # ── Fetch live OHLCV ──
            df_1h = fetch_ohlcv(exchange, symbol, "1h",  limit=300)
            df_4h = fetch_ohlcv(exchange, symbol, "4h",  limit=100)
            df_1d = fetch_ohlcv(exchange, symbol, "1d",  limit=250)

            logger.info(f"  Data fetched: {len(df_1h)} 1h candles")

            # ── Compute features ──
            df_feat = add_all_features(df_1h, df_4h, df_1d)
            df_feat.dropna(subset=FEATURE_COLS, inplace=True)

            if len(df_feat) == 0:
                logger.warning(
                    f"  No clean features -- skipping {asset.upper()}"
                )
                continue

            # Use second-to-last candle (last one is still forming)
            latest      = df_feat.iloc[-2]
            feature_row = latest[FEATURE_COLS].values.astype(np.float32)
            curr_price  = float(df_1h["close"].iloc[-1])

            # ── Unrealized PnL ──
            unrealized_pnl = 0.0
            if s["entry_price"] > 0 and s["position"] != 0:
                if s["position"] == 1:
                    unrealized_pnl = (
                        (curr_price - s["entry_price"]) / s["entry_price"]
                    )
                else:
                    unrealized_pnl = (
                        (s["entry_price"] - curr_price) / s["entry_price"]
                    )

            capital_ratio = s["capital"] / (INITIAL_CAPITAL / 2)

            # ── Get RL signal ──
            signal = engine.get_signal(
                asset          = asset,
                feature_row    = feature_row,
                position       = s["position"],
                unrealized_pnl = unrealized_pnl,
                hold_hours     = s["hold_hours"],
                capital_ratio  = capital_ratio,
            )

            logger.info(
                f"  Signal: {signal['direction'].upper():6s} "
                f"size={signal['size']:.0%} | "
                f"P(sell)={signal['prob_sell']:.3f} "
                f"P(hold)={signal['prob_hold']:.3f} "
                f"P(buy)={signal['prob_buy']:.3f}"
            )

            # ── Drawdown check ──
            if not check_drawdown(asset):
                continue

            # ── Execute action ──
            direction = signal["direction"]
            size      = signal["size"]

            # Close if direction reversed
            if (direction == "long"  and s["position"] == -1) or \
               (direction == "short" and s["position"] ==  1) or \
               (direction == "close" and s["position"] !=  0):
                close_order = executor.close_position(
                    asset,
                    size=s.get("size", 0.0),
                    side="long" if s["position"] == 1 else "short",
                )
                if order_succeeded(close_order):
                    logger.info(
                        f"  CLOSED {asset.upper()} at ${curr_price:,.2f}"
                    )
                    log_trade(asset, "CLOSE", curr_price, 0, s["capital"], signal)
                    s["position"]    = 0.0
                    s["size"]        = 0.0
                    s["entry_price"] = 0.0
                    s["hold_hours"]  = 0
                    save_state()
                else:
                    logger.warning(
                        f"  Close order was not confirmed for {asset.upper()} -- keeping local position state unchanged"
                    )
                    continue

            # Open new position
            if direction == "long" and size > 0 and s["position"] == 0:
                usdt_amount = s["capital"] * size
                order = executor.place_long(asset, usdt_amount)
                if order_succeeded(order):
                    filled_size, fill_price = extract_order_fill(order, curr_price)
                    s["position"]    = 1.0
                    s["size"]        = filled_size
                    s["entry_price"] = fill_price
                    s["hold_hours"]  = 0
                    logger.info(
                        f"  LONG  {asset.upper()} "
                        f"${usdt_amount:,.2f} ({size:.0%}) "
                        f"at ${fill_price:,.2f}"
                    )
                    log_trade(
                        asset, "LONG", fill_price, size, s["capital"], signal
                    )
                    save_state()
                else:
                    logger.warning(
                        f"  Long order was rejected or not confirmed for {asset.upper()}"
                    )

            elif direction == "short" and size > 0 and s["position"] == 0:
                usdt_amount = s["capital"] * size
                order = executor.place_short(asset, usdt_amount)
                if order_succeeded(order):
                    filled_size, fill_price = extract_order_fill(order, curr_price)
                    s["position"]    = -1.0
                    s["size"]        = filled_size
                    s["entry_price"] = fill_price
                    s["hold_hours"]  = 0
                    logger.info(
                        f"  SHORT {asset.upper()} "
                        f"${usdt_amount:,.2f} ({size:.0%}) "
                        f"at ${fill_price:,.2f}"
                    )
                    log_trade(
                        asset, "SHORT", fill_price, size, s["capital"], signal
                    )
                    save_state()
                else:
                    logger.warning(
                        f"  Short order was rejected or not confirmed for {asset.upper()}"
                    )

            elif direction == "hold":
                if s["position"] != 0:
                    s["hold_hours"] += 1
                    pos_str = "LONG" if s["position"] == 1 else "SHORT"
                    logger.info(
                        f"  HOLD -- in {pos_str} for {s['hold_hours']}h "
                        f"| unrealized: {unrealized_pnl*100:+.2f}%"
                    )
                    save_state()
                else:
                    logger.info("  HOLD -- flat, no position")

            # ── Update capital from exchange ──
            live_balance = executor.get_balance()
            if live_balance > 0:
                s["capital"] = live_balance / len(ASSETS)
                if s["capital"] > s["peak_capital"]:
                    s["peak_capital"] = s["capital"]
                save_state()

        except ccxt.NetworkError as e:
            logger.error(f"  Network error for {asset.upper()}: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"  Exchange error for {asset.upper()}: {e}")
        except Exception as e:
            logger.error(
                f"  Unexpected error for {asset.upper()}: {e}",
                exc_info=True
            )

    # ── Portfolio summary ──
    total_capital = sum(state[a]["capital"] for a in ASSETS)
    if total_capital > 0:
        total_return = (
            (total_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        )
        logger.info(
            f"\n  Portfolio: ${total_capital:,.2f} "
            f"| Return: {total_return:+.2f}%"
        )
    else:
        logger.info(
            "\n  Portfolio: tracking unavailable (balance API issue)"
        )
    logger.info("=" * 55)


def print_status():
    logger.info("\n--- STATUS REPORT ---")
    for asset in ASSETS:
        s = state[asset]
        pos_map = {1.0: "LONG", -1.0: "SHORT", 0.0: "FLAT"}
        pos_str = pos_map.get(s["position"], "UNKNOWN")
        logger.info(
            f"  {asset.upper()}: {pos_str:5s} | "
            f"Capital: ${s['capital']:,.2f} | "
            f"Hold: {s['hold_hours']}h"
        )


def main():
    logger.info("Starting Paper Trading Engine -- Phase 6")
    logger.info(f"Assets: {[a.upper() for a in ASSETS]}")
    logger.info(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
    logger.info(f"Max drawdown limit: {MAX_DRAWDOWN*100:.0f}%")
    load_state()

    # ── Load API credentials ──
    api_key    = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    testnet    = os.getenv("TESTNET", "true").lower() == "true"

    if not api_key or not api_secret:
        raise ValueError(
            "API keys not found. "
            "Create a .env file with BINANCE_API_KEY and BINANCE_API_SECRET"
        )

    # ── Load models ──
    logger.info("Loading models...")
    engine = TradingSignalEngine(MODELS_DIR, RL_DIR, PROCESSED_DIR)

    # ── Connect to exchange ──
    logger.info("Connecting to exchange...")
    executor = PaperExecutor(api_key, api_secret, testnet=testnet)

    # Public exchange for OHLCV data (no auth needed)
    exchange = ccxt.binanceusdm({
        "enableRateLimit": True,
        "options"        : {"defaultType": "future"},
    })
    if testnet:
        exchange.set_sandbox_mode(True)

    logger.info("All systems ready. Starting trading loop.\n")

    # ── Run immediately on startup ──
    run_trading_cycle(engine, executor, exchange)

    # ── Schedule hourly at :01 ──
    schedule.every().hour.at(":01").do(
        run_trading_cycle, engine, executor, exchange
    )

    # Status report every 6 hours
    schedule.every(6).hours.do(print_status)

    logger.info(
        "Scheduler running. Next cycle at :01 of the next hour."
    )
    logger.info("Press Ctrl+C to stop.\n")

    while True:
        try:
            schedule.run_pending()
            time.sleep(30)
        except KeyboardInterrupt:
            logger.info("\nShutdown requested. Closing positions...")
            for asset in ASSETS:
                if state[asset]["position"] != 0:
                    try:
                        close_order = executor.close_position(
                            asset,
                            size=state[asset].get("size", 0.0),
                            side="long" if state[asset]["position"] == 1 else "short",
                        )
                        if order_succeeded(close_order):
                            state[asset]["position"] = 0.0
                            state[asset]["size"] = 0.0
                            state[asset]["entry_price"] = 0.0
                            state[asset]["hold_hours"] = 0
                            logger.info(f"  Closed {asset.upper()} position")
                        else:
                            logger.warning(
                                f"  Close order for {asset.upper()} was not confirmed during shutdown"
                            )
                    except Exception as e:
                        logger.error(
                            f"  Failed to close {asset.upper()}: {e}"
                        )
            save_state()
            logger.info("Paper trading stopped cleanly.")
            break


if __name__ == "__main__":
    main()
