#!/usr/bin/env python3
"""
Unified Trading Bot - Single or Multi-Stock Portfolio

Usage:
  # Single stock (auto-sets max-stocks to 1 when using -s)
  python runner.py -t 0.25 -s AAPL -m 100
  
  # Multi-stock auto (bot picks best 15 stocks - default)
  python runner.py -t 0.25 -m 1500
  
  # Multi-stock with forced picks
  python runner.py -t 0.25 -m 1500 --stocks TSLA AAPL --max-stocks 10
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import datetime as dt
from typing import List, Optional, Tuple, Dict
from dotenv import load_dotenv

import pytz
import requests
from alpaca_trade_api import REST
from alpaca_trade_api.rest import APIError
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

import config
from portfolio_manager import PortfolioManager

load_dotenv()

# ===== Logging Setup =====
LOG = logging.getLogger("paper_trading_bot")
LOG.setLevel(logging.INFO)

_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s"))
LOG.addHandler(_console)

FILE_LOG_PATH = config.LOG_PATH
DISABLE_FILE_LOG = os.getenv("BOT_TEE_LOG", "") in ("1", "true", "True")
SCHEDULED_TASK_MODE = os.getenv("SCHEDULED_TASK_MODE", "0") in ("1", "true", "True")

if not os.path.exists(FILE_LOG_PATH):
    open(FILE_LOG_PATH, "a").close()

def append_to_log_line(line: str):
    if DISABLE_FILE_LOG:
        return
    attempts = 3
    delay = 0.2
    for i in range(attempts):
        try:
            enforce_log_max_lines(99)
            with open(FILE_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            return
        except PermissionError:
            if i < attempts - 1:
                time.sleep(delay)
                delay *= 2

def enforce_log_max_lines(max_lines: int = 100):
    try:
        if not os.path.exists(FILE_LOG_PATH):
            return
        with open(FILE_LOG_PATH, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        init_line = None
        for ln in reversed(lines):
            if ln.startswith("INIT "):
                init_line = ln
                break
        last_header_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if "Starting bot for" in lines[i]:
                last_header_idx = i
                break
        header_line = lines[last_header_idx] if last_header_idx is not None else None
        
        if len(lines) <= max_lines:
            return
        
        kept = lines[-(max_lines - 2):]
        final_lines = []
        if init_line and init_line not in kept:
            final_lines.append(init_line)
        if header_line and header_line not in kept:
            final_lines.append(header_line)
        final_lines.extend(kept)
        
        with open(FILE_LOG_PATH, "w", encoding="utf-8") as fh:
            fh.writelines(final_lines)
    except Exception:
        pass

def log_info(msg: str):
    LOG.info(msg)
    append_to_log_line(msg)

def log_warn(msg: str):
    LOG.warning(msg)
    append_to_log_line(f"WARN: {msg}")

def log_error(msg: str):
    LOG.error(msg)
    append_to_log_line(f"ERROR: {msg}")

# ===== API Client =====
def make_client(allow_missing: bool = False, go_live: bool = False):
    key_id = config.ALPACA_API_KEY
    secret_key = config.ALPACA_SECRET_KEY
    base_url = config.ALPACA_BASE_URL
    
    if not all([key_id, secret_key, base_url]):
        if allow_missing:
            return None
        raise ValueError("Missing Alpaca credentials")
    
    if go_live:
        confirm = os.getenv("CONFIRM_GO_LIVE", "NO")
        if confirm != "YES":
            raise ValueError("Live trading requires CONFIRM_GO_LIVE=YES in .env")
    
    return REST(key_id, secret_key, base_url, api_version='v2')

# ===== Market Data =====
def snap_interval_to_supported_seconds(seconds: int) -> int:
    if seconds < 60:
        return 60
    elif seconds < 300:
        return 60
    elif seconds < 900:
        return 300
    elif seconds < 3600:
        return 900
    elif seconds < 14400:
        return 3600
    else:
        return 14400

def fetch_closes(client, symbol: str, interval_seconds: int, limit_bars: int) -> List[float]:
    try:
        snap = snap_interval_to_supported_seconds(interval_seconds)
        
        if snap == 60:
            tf = TimeFrame(1, TimeFrameUnit.Minute)
        elif snap == 300:
            tf = TimeFrame(5, TimeFrameUnit.Minute)
        elif snap == 900:
            tf = TimeFrame(15, TimeFrameUnit.Minute)
        elif snap == 3600:
            tf = TimeFrame(1, TimeFrameUnit.Hour)
        else:
            tf = TimeFrame(4, TimeFrameUnit.Hour)
        
        bars = client.get_bars(symbol, tf, limit=limit_bars).df
        if bars.empty:
            return []
        return list(bars['close'].values)
    except Exception:
        pass
    
    # Fallback to Polygon
    try:
        polygon_key = config.POLYGON_API_KEY
        if not polygon_key:
            return []
        
        snap = snap_interval_to_supported_seconds(interval_seconds)
        multiplier = 1
        timespan = "minute"
        
        if snap == 60:
            multiplier, timespan = 1, "minute"
        elif snap == 300:
            multiplier, timespan = 5, "minute"
        elif snap == 900:
            multiplier, timespan = 15, "minute"
        elif snap == 3600:
            multiplier, timespan = 1, "hour"
        else:
            multiplier, timespan = 4, "hour"
        
        end_date = dt.datetime.now(pytz.UTC)
        start_date = end_date - dt.timedelta(days=365)
        
        url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
               f"{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/"
               f"{end_date.strftime('%Y-%m-%d')}")
        
        resp = requests.get(url, params={"apiKey": polygon_key, "limit": limit_bars * 2}, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("results"):
                closes = [float(r["c"]) for r in data["results"]]
                return closes[-limit_bars:] if len(closes) > limit_bars else closes
    except Exception:
        pass
    
    return []

# ===== Trading Logic =====
def sma(closes: List[float], window: int) -> float:
    if len(closes) < window:
        return closes[-1] if closes else 0.0
    return sum(closes[-window:]) / window

def decide_action(closes: List[float], short_w: int, long_w: int) -> str:
    if len(closes) < max(short_w, long_w):
        return "hold"
    short_ma = sma(closes, short_w)
    long_ma = sma(closes, long_w)
    
    if short_ma > long_ma * 1.002:
        return "buy"
    elif short_ma < long_ma * 0.998:
        return "sell"
    return "hold"

def compute_confidence(closes: List[float]) -> float:
    if len(closes) < config.LONG_WINDOW:
        return 0.0
    short_ma = sma(closes, config.SHORT_WINDOW)
    long_ma = sma(closes, config.LONG_WINDOW)
    base_conf = abs((short_ma / long_ma) - 1.0)
    
    # Momentum boost
    if len(closes) >= 5:
        recent_change = (closes[-1] - closes[-5]) / closes[-5]
        signal = 1 if short_ma > long_ma else -1
        momentum_direction = 1 if recent_change > 0 else -1
        if signal == momentum_direction:
            base_conf *= 1.2
    
    return base_conf

def pct_stddev(closes: List[float]) -> float:
    if not closes:
        return 0.0
    mean = sum(closes) / len(closes)
    variance = sum((x - mean) ** 2 for x in closes) / len(closes)
    stddev = math.sqrt(variance)
    return stddev / mean if mean != 0 else 0.0

# ===== Position Management =====
def get_position(client, symbol: str):
    try:
        pos = client.get_position(symbol)
        return {
            "qty": float(pos.qty),
            "avg_entry_price": float(pos.avg_entry_price),
            "market_value": float(pos.market_value),
            "unrealized_pl": float(pos.unrealized_pl),
        }
    except:
        return None

def adjust_runtime_params(confidence: float, base_tp: float, base_sl: float, base_frac: float):
    c = max(0.0, min(1.0, confidence / 0.1))
    tp = base_tp * (1.0 + 0.5 * c)
    sl = base_sl * (1.0 + 0.2 * c)
    frac = base_frac * (1.0 + 0.3 * c)
    
    tp = max(config.MIN_TAKE_PROFIT_PERCENT, min(config.MAX_TAKE_PROFIT_PERCENT, tp))
    sl = max(config.MIN_STOP_LOSS_PERCENT, min(config.MAX_STOP_LOSS_PERCENT, sl))
    frac = max(config.MIN_TRADE_SIZE_FRAC, min(config.MAX_TRADE_SIZE_FRAC, frac))
    
    if config.RISKY_MODE_ENABLED:
        tp *= config.RISKY_TP_MULT
        sl *= config.RISKY_SL_MULT
        frac *= config.RISKY_FRAC_MULT
        frac = min(frac, config.RISKY_MAX_FRAC_CAP)
    
    return tp, sl, frac

def compute_order_qty_from_remaining(current_price: float, remaining_cap: float, fraction: float) -> int:
    usable = remaining_cap * fraction
    return int(usable / current_price)

def buy_flow(client, symbol: str, last_price: float, max_cap: float,
             confidence: float, base_frac: float, base_tp: float, base_sl: float,
             dynamic_enabled: bool = True, interval_seconds: int = None):
    
    pos = get_position(client, symbol)
    if pos:
        return (False, "Position exists")
    
    if confidence < config.MIN_CONFIDENCE_TO_TRADE:
        return (False, f"Low confidence: {confidence:.4f}")
    
    # Profitability check
    if interval_seconds:
        closes = fetch_closes(client, symbol, interval_seconds, config.LONG_WINDOW + 50)
        if closes:
            sim = simulate_signals_and_projection(closes, interval_seconds, override_cap_usd=max_cap)
            expected_daily = float(sim.get("expected_daily_usd", 0.0))
            if expected_daily < config.PROFITABILITY_MIN_EXPECTED_USD:
                return (False, f"Expected ${expected_daily:.2f}/day < ${config.PROFITABILITY_MIN_EXPECTED_USD}")
            
            # Volatility check
            vol_pct = pct_stddev(closes[-config.VOLATILITY_WINDOW:])
            if vol_pct > config.VOLATILITY_PCT_THRESHOLD:
                return (False, f"High volatility: {vol_pct*100:.1f}%")
    
    # Calculate params
    if dynamic_enabled:
        tp, sl, frac = adjust_runtime_params(confidence, base_tp, base_sl, base_frac)
    else:
        tp, sl, frac = base_tp, base_sl, base_frac
    
    qty = compute_order_qty_from_remaining(last_price, max_cap, frac)
    if qty < 1:
        return (False, "Insufficient capital")
    
    try:
        tp_price = last_price * (1 + tp / 100.0)
        sl_price = last_price * (1 - sl / 100.0)
        
        client.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='gtc',
            order_class='bracket',
            take_profit={'limit_price': round(tp_price, 2)},
            stop_loss={'stop_price': round(sl_price, 2)}
        )
        return (True, f"Bought {qty} @ ${last_price:.2f} (TP:{tp:.2f}% SL:{sl:.2f}%)")
    except Exception as e:
        return (False, f"Order failed: {e}")

def sell_flow(client, symbol: str, confidence: float = 0.0):
    pos = get_position(client, symbol)
    if not pos:
        return (False, "No position")
    
    qty = int(pos["qty"])
    realized_pl = pos["unrealized_pl"]
    
    try:
        # Cancel bracket orders
        orders = client.list_orders(status='open', symbols=[symbol])
        for order in orders:
            try:
                client.cancel_order(order.id)
            except:
                pass
        
        # Market sell
        client.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
        
        # Log to PnL ledger
        try:
            ledger_path = "pnl_ledger.json"
            ledger = []
            if os.path.exists(ledger_path):
                with open(ledger_path, "r") as f:
                    ledger = json.load(f)
            
            ledger.append({
                "timestamp": dt.datetime.now(pytz.UTC).isoformat(),
                "symbol": symbol,
                "realized_pl": realized_pl
            })
            
            with open(ledger_path, "w") as f:
                json.dump(ledger, f, indent=2)
        except:
            pass
        
        return (True, f"Sold {qty} shares (P&L: ${realized_pl:+.2f})")
    except Exception as e:
        return (False, f"Sell failed: {e}")

def enforce_safety(client, symbol: str):
    # Daily loss check
    try:
        account = client.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        
        if last_equity > 0:
            daily_pnl_pct = ((equity - last_equity) / last_equity) * 100
            
            if daily_pnl_pct < -config.MAX_DAILY_LOSS_PERCENT:
                log_warn(f"Daily loss limit hit: {daily_pnl_pct:.2f}%")
                sell_flow(client, symbol)
                sys.exit(1)
    except:
        pass

# ===== Simulation =====
def simulate_signals_and_projection(
    closes: List[float],
    interval_seconds: int,
    override_tp_pct: Optional[float] = None,
    override_sl_pct: Optional[float] = None,
    override_trade_frac: Optional[float] = None,
    override_cap_usd: Optional[float] = None
) -> dict:
    
    if len(closes) < config.LONG_WINDOW + 10:
        return {"win_rate": 0.0, "expected_trades_per_day": 0.0, "expected_daily_usd": 0.0}
    
    tp_pct = override_tp_pct if override_tp_pct is not None else config.TAKE_PROFIT_PERCENT
    sl_pct = override_sl_pct if override_sl_pct is not None else config.STOP_LOSS_PERCENT
    frac = override_trade_frac if override_trade_frac is not None else config.TRADE_SIZE_FRAC_OF_CAP
    cap = override_cap_usd if override_cap_usd is not None else config.MAX_CAP_USD
    
    wins = 0
    total_signals = 0
    
    for i in range(config.LONG_WINDOW, len(closes)):
        window = closes[:i+1]
        action = decide_action(window, config.SHORT_WINDOW, config.LONG_WINDOW)
        
        if action == "buy":
            total_signals += 1
            price = closes[i]
            
            tp_price = price * (1 + tp_pct / 100)
            sl_price = price * (1 - sl_pct / 100)
            
            for j in range(i + 1, len(closes)):
                if closes[j] >= tp_price:
                    wins += 1
                    break
                elif closes[j] <= sl_price:
                    break
        
        elif action == "sell":
            total_signals += 1
            price = closes[i]
            
            tp_price = price * (1 - tp_pct / 100)
            sl_price = price * (1 + sl_pct / 100)
            
            for j in range(i + 1, len(closes)):
                if closes[j] <= tp_price:
                    wins += 1
                    break
                elif closes[j] >= sl_price:
                    break
    
    win_rate = wins / total_signals if total_signals > 0 else 0.0
    simulation_bars = max(1.0, len(closes) - config.LONG_WINDOW)
    bars_per_day = (86400 / interval_seconds)
    days_simulated = simulation_bars / bars_per_day
    trades_per_day = total_signals / days_simulated if days_simulated > 0 else 0.0
    
    expected_return_per_trade = ((tp_pct / 100) * win_rate) - ((sl_pct / 100) * (1 - win_rate))
    usable_cap_per_trade = cap * frac
    expected_usd_per_trade = usable_cap_per_trade * expected_return_per_trade
    expected_daily_usd = expected_usd_per_trade * trades_per_day
    
    return {
        "win_rate": win_rate,
        "expected_trades_per_day": trades_per_day,
        "expected_daily_usd": expected_daily_usd
    }

# ===== Market Hours =====
def in_market_hours(client) -> bool:
    try:
        clock = client.get_clock()
        return clock.is_open
    except:
        now = dt.datetime.now(pytz.timezone('US/Eastern'))
        return now.weekday() < 5 and 9 <= now.hour < 16

def sleep_until_market_open(client):
    log_info("Market closed. Sleeping until open...")
    while not in_market_hours(client):
        if SCHEDULED_TASK_MODE:
            log_info("Market closed in scheduled task mode - exiting")
            sys.exit(0)
        time.sleep(300)

def prevent_system_sleep(enable: bool):
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        if enable:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        else:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except:
        pass

# ===== Portfolio Evaluation =====
def evaluate_portfolio_and_opportunities(
    client,
    current_symbols: List[str],
    forced_symbols: List[str],
    scan_universe: List[str],
    interval_seconds: int,
    cap_per_stock: float,
    max_positions: int
) -> Dict:
    # Import here to avoid circular dependency
    from stock_scanner import scan_stocks
    
    results = {
        "current_scores": {},
        "opportunities": [],
        "rebalance_suggestions": []
    }
    
    # Score current positions
    for symbol in current_symbols:
        try:
            closes = fetch_closes(client, symbol, interval_seconds, config.LONG_WINDOW + 50)
            if not closes:
                continue
            confidence = compute_confidence(closes)
            action = decide_action(closes, config.SHORT_WINDOW, config.LONG_WINDOW)
            
            score = confidence
            if action == "sell":
                score -= 0.05
            
            if symbol in forced_symbols:
                score += 999
            
            results["current_scores"][symbol] = score
        except:
            if symbol in forced_symbols:
                results["current_scores"][symbol] = 998
            else:
                results["current_scores"][symbol] = -999
    
    # Scan for opportunities
    if len(current_symbols) < max_positions or (results["current_scores"] and min(results["current_scores"].values()) < 0):
        try:
            scan_results = scan_stocks(
                symbols=[s for s in scan_universe if s not in current_symbols],
                interval_seconds=interval_seconds,
                cap_per_stock=cap_per_stock,
                max_results=5,
                verbose=False
            )
            results["opportunities"] = scan_results
        except Exception as e:
            log_warn(f"Stock scan failed: {e}")
    
    # Generate rebalance suggestions
    if results["opportunities"] and results["current_scores"]:
        replaceable_stocks = {s: score for s, score in results["current_scores"].items() 
                             if s not in forced_symbols}
        
        if replaceable_stocks:
            worst_held = min(replaceable_stocks.items(), key=lambda x: x[1])
            best_opportunity = results["opportunities"][0] if results["opportunities"] else None
            
            if best_opportunity and worst_held[1] < 0 and best_opportunity["score"] > worst_held[1] + 0.1:
                results["rebalance_suggestions"].append((worst_held[0], best_opportunity["symbol"]))
    
    return results

# ===== Main =====
def main():
    # Import here to avoid circular dependency
    from stock_scanner import scan_stocks, get_stock_universe
    
    parser = argparse.ArgumentParser(description="Unified trading bot (single or multi-stock)")
    parser.add_argument("-t", "--time", type=float, required=True,
                       help="Trading interval in hours (0.25=15min, 1.0=1hr)")
    parser.add_argument("-m", "--max-cap", type=float, required=True,
                       help="Total capital (for multi-stock) or max capital (for single-stock)")
    parser.add_argument("-s", "--symbol", type=str,
                       help="For single-stock mode: stock symbol. For multi-stock: leave blank or use --stocks")
    parser.add_argument("--stocks", nargs="+",
                       help="Force specific stocks in portfolio")
    parser.add_argument("--max-stocks", type=int, default=15,
                       help="Max positions (default: 15 for multi-stock, use 1 for single-stock)")
    parser.add_argument("--cap-per-stock", type=float,
                       help="Capital per stock (default: total/max)")
    parser.add_argument("--tp", type=float,
                       help="Take profit percent (overrides config)")
    parser.add_argument("--sl", type=float,
                       help="Stop loss percent (overrides config)")
    parser.add_argument("--frac", type=float,
                       help="Position size fraction (overrides config)")
    parser.add_argument("--no-dynamic", action="store_true",
                       help="Disable dynamic adjustments")
    parser.add_argument("--rebalance-every", type=int, default=4,
                       help="Rebalance every N intervals (multi-stock only)")
    parser.add_argument("--go-live", action="store_true",
                       help="Enable live trading")
    parser.add_argument("--allow-missing-keys", action="store_true",
                       help="Debug mode")
    
    args = parser.parse_args()

    # Determine mode
    # If user provides -s (single symbol), they want single-stock mode
    if args.symbol and not args.stocks:
        # Single-stock mode implied
        if args.max_stocks != 15:  # If user explicitly changed it
            pass  # Use their value
        else:
            # Default case with -s: force single-stock
            args.max_stocks = 1
    
    is_multi_stock = args.max_stocks > 1 or args.stocks
    
    if is_multi_stock:
        # Multi-stock mode
        forced_stocks = [s.upper() for s in args.stocks] if args.stocks else []
        
        if len(forced_stocks) > args.max_stocks:
            log_warn(f"Specified {len(forced_stocks)} stocks but max is {args.max_stocks}")
            return 1
        
        cap_per_stock = args.cap_per_stock or (args.max_cap / args.max_stocks)
        
        if cap_per_stock < 10:
            log_warn(f"Capital per stock is ${cap_per_stock:.2f} (very small)")
            # Auto-continue in scheduled/non-interactive mode
            if not SCHEDULED_TASK_MODE:
                response = input("Continue anyway? (y/n): ").strip().lower()
                if response != 'y':
                    return 1
            else:
                log_info("  Auto-continuing in scheduled mode...")
    else:
        # Single-stock mode
        if not args.symbol:
            log_warn("Single-stock mode requires -s/--symbol")
            return 1
        
        symbol = args.symbol.upper()
        cap_per_stock = args.max_cap
        forced_stocks = [symbol]
    
    # Setup
    interval_seconds = int(args.time * 3600)
    
    try:
        client = make_client(allow_missing=args.allow_missing_keys, go_live=args.go_live)
    except Exception as e:
        log_warn(f"Failed to create client: {e}")
        return 2
    
    # Initialize portfolio manager
    portfolio = PortfolioManager() if is_multi_stock else None
    
    # Log startup
    log_info(f"{'='*70}")
    log_info(f"UNIFIED TRADING BOT STARTING")
    log_info(f"{'='*70}")
    log_info(f"  Mode: {'Multi-Stock Portfolio' if is_multi_stock else 'Single-Stock'}")
    log_info(f"  Interval: {interval_seconds}s ({args.time}h)")
    log_info(f"  Total Capital: ${args.max_cap}")
    log_info(f"  Max Positions: {args.max_stocks}")
    log_info(f"  Cap Per Stock: ${cap_per_stock:.2f}")
    
    if is_multi_stock:
        log_info(f"  Forced Stocks: {len(forced_stocks)}")
        log_info(f"  Auto-Fill Slots: {args.max_stocks - len(forced_stocks)}")
        log_info(f"  Rebalance Every: {args.rebalance_every} intervals")
        
        # Sync with broker
        try:
            account = client.get_account()
            positions = client.list_positions()
            log_info(f"Broker: ${float(account.portfolio_value):.2f} total value")
            log_info(f"Found {len(positions)} existing positions")
            for pos in positions:
                portfolio.update_position(
                    pos.symbol,
                    float(pos.qty),
                    float(pos.avg_entry_price),
                    float(pos.market_value),
                    float(pos.unrealized_pl)
                )
        except Exception as e:
            log_warn(f"Could not sync: {e}")
        
        scan_universe = get_stock_universe()
    else:
        log_info(f"  Symbol: {symbol}")
    
    log_info(f"{'='*70}\n")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1

            if not in_market_hours(client):
                sleep_until_market_open(client)
                continue
            
            prevent_system_sleep(True)

            log_info(f"=== Iteration {iteration} ===")
            
            if is_multi_stock:
                # Multi-stock logic
                current_positions = portfolio.get_all_positions()
                held_symbols = list(current_positions.keys())
                log_info(f"Positions: {len(held_symbols)}/{args.max_stocks}")
                
                # Update positions
                for sym in held_symbols:
                    try:
                        pos = get_position(client, sym)
                        if pos:
                            portfolio.update_position(sym, pos["qty"], pos["avg_entry_price"], 
                                                    pos["market_value"], pos["unrealized_pl"])
                        else:
                            portfolio.remove_position(sym)
                    except:
                        pass
                
                # Rebalancing check
                should_rebalance = (iteration % args.rebalance_every == 0)
                
                if should_rebalance or len(held_symbols) < len(forced_stocks):
                    log_info("Evaluating portfolio...")
                    evaluation = evaluate_portfolio_and_opportunities(
                        client, held_symbols, forced_stocks, scan_universe,
                        interval_seconds, cap_per_stock, args.max_stocks
                    )
                    
                    # Execute rebalancing
                    for sell_sym, buy_sym in evaluation["rebalance_suggestions"]:
                        log_info(f"🔄 REBALANCE: Sell {sell_sym} → Buy {buy_sym}")
                        ok, msg = sell_flow(client, sell_sym)
                        if ok:
                            portfolio.remove_position(sell_sym)
                            log_info(f"✅ {msg}")
                
                # Determine stocks to trade
                stocks_to_evaluate = list(set(forced_stocks + held_symbols))
                
                # Add opportunities if room
                if len(stocks_to_evaluate) < args.max_stocks:
                    try:
                        opportunities = scan_stocks(
                            symbols=[s for s in scan_universe if s not in stocks_to_evaluate],
                            interval_seconds=interval_seconds,
                            cap_per_stock=cap_per_stock,
                            max_results=args.max_stocks - len(stocks_to_evaluate),
                            verbose=False
                        )
                        for opp in opportunities:
                            if opp["expected_daily"] >= config.PROFITABILITY_MIN_EXPECTED_USD:
                                stocks_to_evaluate.append(opp["symbol"])
                                log_info(f"  Adding {opp['symbol']} (${opp['expected_daily']:.2f}/day)")
                                if len(stocks_to_evaluate) >= args.max_stocks:
                                    break
                    except:
                        pass
                
                # Trade each stock
                log_info(f"\nTrading {len(stocks_to_evaluate)} stocks:")
                for i, sym in enumerate(stocks_to_evaluate, 1):
                    try:
                        log_info(f"[{i}/{len(stocks_to_evaluate)}] {sym}...")
                        closes = fetch_closes(client, sym, interval_seconds, config.LONG_WINDOW + 10)
                        if not closes:
                            log_info(f"  No data")
                            continue
                        
                        action = decide_action(closes, config.SHORT_WINDOW, config.LONG_WINDOW)
                        confidence = compute_confidence(closes)
                        last_price = closes[-1]
                        
                        log_info(f"  ${last_price:.2f} | {action.upper()} | conf={confidence:.4f}")
                        
                        enforce_safety(client, sym)
                        
                        if action == "buy":
                            ok, msg = buy_flow(
                                client, sym, last_price, cap_per_stock,
                                max(0.0, confidence),
                                args.frac or config.TRADE_SIZE_FRAC_OF_CAP,
                                args.tp or config.TAKE_PROFIT_PERCENT,
                                args.sl or config.STOP_LOSS_PERCENT,
                                dynamic_enabled=not args.no_dynamic,
                                interval_seconds=interval_seconds
                            )
                            log_info(f"  {'✅' if ok else '⚪'} {msg}")
                        elif action == "sell":
                            ok, msg = sell_flow(client, sym)
                            log_info(f"  {'✅' if ok else '⚪'} {msg}")
                    except Exception as e:
                        log_warn(f"  Error: {e}")
                
                # Portfolio summary
                total_value = portfolio.get_total_market_value()
                total_pl = portfolio.get_total_unrealized_pl()
                log_info(f"\nPortfolio: ${total_value:.2f} | P&L: ${total_pl:+.2f}\n")
            
            else:
                # Single-stock logic
                log_info(f"Trading {symbol}...")
                closes = fetch_closes(client, symbol, interval_seconds, config.LONG_WINDOW + 10)
                if not closes:
                    log_info("No data available")
                    time.sleep(interval_seconds)
                    continue

                action = decide_action(closes, config.SHORT_WINDOW, config.LONG_WINDOW)
                confidence = compute_confidence(closes)
                last_price = closes[-1]

                log_info(f"${last_price:.2f} | {action.upper()} | conf={confidence:.4f}")

                enforce_safety(client, symbol)

                if action == "buy":
                    ok, msg = buy_flow(
                        client, symbol, last_price, cap_per_stock,
                        max(0.0, confidence),
                        args.frac or config.TRADE_SIZE_FRAC_OF_CAP,
                        args.tp or config.TAKE_PROFIT_PERCENT,
                        args.sl or config.STOP_LOSS_PERCENT,
                        dynamic_enabled=not args.no_dynamic,
                        interval_seconds=interval_seconds
                    )
                    log_info(f"{'✅' if ok else '⚪'} {msg}")
                elif action == "sell":
                    ok, msg = sell_flow(client, symbol)
                    log_info(f"{'✅' if ok else '⚪'} {msg}\n")
            
            time.sleep(interval_seconds)
    
    except KeyboardInterrupt:
        log_info("Interrupted - shutting down")
        prevent_system_sleep(False)
        return 0
    except Exception as e:
        log_error(f"Fatal error: {e}")
        prevent_system_sleep(False)
        return 1


if __name__ == "__main__":
    sys.exit(main())
