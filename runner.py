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
LOG.propagate = False  # Prevent duplicate logging from parent loggers

# Only add handler if not already added (prevents duplicates on module re-import)
if not LOG.handlers:
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
        
        # Keep only non-empty lines
        lines = [ln for ln in lines if ln.strip()]
        
        if len(lines) <= max_lines:
            return
        
        # Find INIT line
        init_line = None
        for ln in lines:
            if ln.startswith("INIT "):
                init_line = ln
                break
        
        # Keep last N lines
        kept = lines[-max_lines:]
        
        # Ensure INIT line is at the top if not already included
        if init_line and init_line not in kept:
            kept = [init_line] + kept[1:]  # Replace first line with INIT
        
        with open(FILE_LOG_PATH, "w", encoding="utf-8") as fh:
            fh.writelines(kept)
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
    # Try yfinance first (FREE, unlimited, great history)
    try:
        import yfinance as yf
        
        # Map seconds to yfinance interval
        if interval_seconds <= 300:
            yf_interval = "5m"
            days = max(5, (limit_bars * 5 / 60 / 6.5) * 1.5)  # 5min bars, 6.5 hour day
        elif interval_seconds <= 900:
            yf_interval = "15m"
            days = max(10, (limit_bars * 15 / 60 / 6.5) * 1.5)  # 15min bars
        elif interval_seconds <= 3600:
            yf_interval = "1h"
            days = max(20, (limit_bars / 6.5) * 1.5)  # hourly bars
        else:
            yf_interval = "1d"
            days = min(365, limit_bars * 1.5)  # daily bars
        
        from datetime import datetime, timedelta
        import pytz
        end = datetime.now(pytz.UTC)
        start = end - timedelta(days=int(days))
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start, end=end, interval=yf_interval)
        
        if not hist.empty and 'Close' in hist.columns:
            closes = list(hist['Close'].values)
            # Return the most recent bars up to limit
            result = closes[-limit_bars:] if len(closes) > limit_bars else closes
            if len(result) >= 25:  # Got enough data
                return result
    except Exception as e:
        pass  # Fall back to Alpaca
    
    # Fallback to Alpaca
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
        
        # Try to get data from Alpaca
        bars = client.get_bars(symbol, tf, limit=limit_bars).df
        if not bars.empty:
            closes = list(bars['close'].values)
            # Return whatever we got - don't require exact count
            if len(closes) > 0:
                return closes
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
        frac *= config.RISKY_SIZE_MULT  # Fixed: was RISKY_FRAC_MULT
        frac = min(frac, config.RISKY_MAX_FRAC_CAP)
    
    return tp, sl, frac

def compute_order_qty_from_remaining(current_price: float, remaining_cap: float, fraction: float) -> float:
    """Calculate quantity to buy - supports fractional shares"""
    usable = remaining_cap * fraction
    qty = usable / current_price
    # Round to 6 decimal places (standard for fractional shares)
    return round(qty, 6)

def buy_flow(client, symbol: str, last_price: float, available_cap: float,
             confidence: float, base_frac: float, base_tp: float, base_sl: float,
             dynamic_enabled: bool = True, interval_seconds: int = None):
    """
    Buy a stock using available capital.
    available_cap: How much capital is available for THIS specific trade
    """
    pos = get_position(client, symbol)
    if pos:
        return (False, "Position exists")
    
    if confidence < config.MIN_CONFIDENCE_TO_TRADE:
        return (False, f"Low confidence: {confidence:.4f}")
    
    # Profitability check
    if interval_seconds:
        closes = fetch_closes(client, symbol, interval_seconds, config.LONG_WINDOW + 50)
        if closes:
            sim = simulate_signals_and_projection(closes, interval_seconds, override_cap_usd=available_cap)
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
    
    qty = compute_order_qty_from_remaining(last_price, available_cap, frac)
    if qty < 0.001:  # Minimum fractional share
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
        shares_text = f"{qty:.6f}".rstrip('0').rstrip('.') if qty < 1 else f"{qty:.2f}"
        return (True, f"Bought {shares_text} shares @ ${last_price:.2f} (TP:{tp:.2f}% SL:{sl:.2f}%)")
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
    
    # Lower minimum requirement - work with what we have
    min_bars = max(config.LONG_WINDOW + 2, 25)
    if len(closes) < min_bars:
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
    
    win_rate = wins / total_signals if total_signals > 0 else 0.55  # Assume 55% win rate if no history
    simulation_bars = max(1.0, len(closes) - config.LONG_WINDOW)
    bars_per_day = (86400 / interval_seconds)
    days_simulated = simulation_bars / bars_per_day
    trades_per_day = total_signals / days_simulated if days_simulated > 0 else 4.0  # Estimate 4 trades/day
    
    # If we have no historical signals, estimate based on typical SMA crossover frequency
    if total_signals == 0:
        win_rate = 0.55
        trades_per_day = 4.0  # Reasonable estimate for 15-min intervals
    
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
    # Show countdown info once
    try:
        clock = client.get_clock()
        if clock.next_open:
            now = dt.datetime.now(pytz.UTC)
            next_open = clock.next_open
            if next_open.tzinfo is None:
                next_open = pytz.UTC.localize(next_open)
            
            time_until_open = (next_open - now).total_seconds()
            hours = int(time_until_open // 3600)
            minutes = int((time_until_open % 3600) // 60)
            
            log_info(f"Market closed. Opens in {hours}h {minutes}m ({next_open.astimezone(pytz.timezone('US/Eastern')).strftime('%I:%M %p ET')})")
        else:
            log_info("Market closed. Waiting until open...")
    except:
        log_info("Market closed. Waiting until open...")
    
    # Exit immediately if in scheduled task mode
    if SCHEDULED_TASK_MODE:
        log_info("Scheduled task mode - exiting until next run")
        sys.exit(0)
    
    # Otherwise sleep silently and check periodically (no log spam)
    log_info("Sleeping silently until market opens...")
    while not in_market_hours(client):
        time.sleep(300)  # Check every 5 minutes, but don't log
    
    # Market opened
    log_info("Market is now open!")

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

# ===== Smart Capital Allocation =====
def allocate_capital_smartly(
    client,
    symbols: List[str],
    forced_symbols: List[str],
    total_capital: float,
    interval_seconds: int,
    min_cap_per_stock: float = 10.0
) -> Dict[str, float]:
    """
    Allocate capital based on expected profitability.
    Better opportunities get more capital.
    """
    allocations = {}
    
    # Score each stock
    scores = {}
    log_info(f"Scoring {len(symbols)} stocks for capital allocation...")
    for symbol in symbols:
        try:
            closes = fetch_closes(client, symbol, interval_seconds, config.LONG_WINDOW + 50)
            if not closes or len(closes) < config.LONG_WINDOW + 10:
                scores[symbol] = 0.0
                log_info(f"  {symbol}: No data (score: 0.0)")
                continue
            
            sim = simulate_signals_and_projection(closes, interval_seconds, override_cap_usd=100)
            expected_daily = sim.get("expected_daily_usd", 0.0)
            confidence = compute_confidence(closes)
            
            # Score = expected return + confidence boost
            # Use absolute value of expected_daily to consider both bullish and bearish opportunities
            score = abs(expected_daily) + confidence * 10
            
            # Forced stocks get minimum viable score
            if symbol in forced_symbols:
                score = max(score, 1.0)
            
            scores[symbol] = score
            log_info(f"  {symbol}: exp_daily=${expected_daily:.2f}, conf={confidence:.4f}, score={score:.2f}")
        except Exception as e:
            scores[symbol] = 1.0 if symbol in forced_symbols else 0.0
            log_info(f"  {symbol}: Error ({e}) - score: {scores[symbol]:.2f}")
    
    # Filter out stocks with very low scores (keep if score > 0.1 or forced)
    # Lowered threshold from 0 to allow marginal opportunities
    viable_symbols = [s for s in symbols if scores[s] > 0.1 or s in forced_symbols]
    
    log_info(f"Viable stocks: {len(viable_symbols)}/{len(symbols)}")
    
    if not viable_symbols:
        log_warn(f"No viable stocks found (all scores <= 0.1). Holding cash.")
        return {}  # Return empty dict = no allocation, hold cash
    
    # Aggressive mode: Amplify differences (winners get MORE, losers get LESS)
    # This concentrates capital on best performers for faster gains
    if len(viable_symbols) > 1:
        min_score = min(scores[s] for s in viable_symbols)
        max_score = max(scores[s] for s in viable_symbols)
        score_range = max_score - min_score if max_score > min_score else 1
        
        # Amplify score differences (best stocks get 50% bonus, worst get 0%)
        for s in viable_symbols:
            normalized = (scores[s] - min_score) / score_range if score_range > 0 else 0.5
            aggression_bonus = 0.5 * normalized  # Best +50%, worst +0%
            scores[s] = scores[s] * (1.0 + aggression_bonus)
    
    # Calculate proportional allocation
    total_score = sum(scores[s] for s in viable_symbols)
    
    if total_score == 0:
        # Equal split if all scores are 0
        cap_each = total_capital / len(viable_symbols)
        return {s: cap_each for s in viable_symbols}
    
    # Allocate proportionally
    for symbol in viable_symbols:
        proportion = scores[symbol] / total_score
        allocated = total_capital * proportion
        allocations[symbol] = max(min_cap_per_stock, allocated)
    
    # Normalize to stay within total_capital
    total_allocated = sum(allocations.values())
    if total_allocated > total_capital:
        scale_factor = total_capital / total_allocated
        for symbol in allocations:
            allocations[symbol] *= scale_factor
    
    return allocations

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
    
    # Quick check: Don't even start if market is closed and won't open soon
    if SCHEDULED_TASK_MODE:
        try:
            now = dt.datetime.now(pytz.timezone('US/Eastern'))
            # Only run if it's a weekday and either market is open or within 1 hour of opening
            is_weekday = now.weekday() < 5
            hour = now.hour
            minute = now.minute
            time_in_minutes = hour * 60 + minute
            market_open_minutes = 9 * 60 + 30  # 9:30 AM
            market_close_minutes = 16 * 60      # 4:00 PM
            
            # Exit if it's weekend OR if market closed and more than 1 hour away
            if not is_weekday:
                print("Weekend - exiting")
                return 0
            if time_in_minutes < (market_open_minutes - 60):  # Before 8:30 AM
                print(f"Too early ({now.strftime('%I:%M %p')}) - exiting")
                return 0
            if time_in_minutes > market_close_minutes:  # After 4:00 PM
                print(f"Market closed ({now.strftime('%I:%M %p')}) - exiting")
                return 0
        except:
            pass  # If check fails, continue anyway
    
    parser = argparse.ArgumentParser(description="Unified trading bot (single or multi-stock)")
    parser.add_argument("-t", "--time", type=float, required=True,
                       help="Trading interval in hours (0.25=15min, 1.0=1hr)")
    parser.add_argument("-m", "--max-cap", type=float, required=True,
                       help="Total capital (for multi-stock) or max capital (for single-stock)")
    parser.add_argument("-s", "--symbol", type=str,
                       help="For single-stock mode: stock symbol. For multi-stock: leave blank or use --stocks")
    parser.add_argument("--stocks", nargs="+",
                       help="Force specific stocks in portfolio")
    parser.add_argument("--max-stocks", type=int, default=5,
                       help="Max positions (default: 5 for concentrated gains, increase for more diversification)")
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
        
        # Use smart allocation by default, fallback to equal split if user specifies cap-per-stock
        cap_per_stock = args.cap_per_stock or (args.max_cap / args.max_stocks)
        use_smart_allocation = args.cap_per_stock is None  # Smart unless user forced specific cap
        
        # Only warn if user manually set cap-per-stock to a very small value
        if not use_smart_allocation and cap_per_stock < 10:
            log_warn(f"Manual cap per stock is ${cap_per_stock:.2f} (very small)")
            if not SCHEDULED_TASK_MODE:
                response = input("Continue anyway? (y/n): ").strip().lower()
                if response != 'y':
                    return 1
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
    
    if is_multi_stock:
        log_info(f"  Forced Stocks: {len(forced_stocks)}")
        log_info(f"  Auto-Fill Slots: {args.max_stocks - len(forced_stocks)}")
        log_info(f"  Rebalance Every: {args.rebalance_every} intervals")
        log_info(f"  Allocation: {'Smart (profit-based)' if use_smart_allocation else f'Equal (${cap_per_stock:.2f} each)'}")
        
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
        log_info(f"Broker: ${float(client.get_account().portfolio_value):.2f} total value" if client else "")
    
    log_info(f"{'='*70}\n")
    
    iteration = 0
    
    try:
        # Keep PC awake during market hours only
        prevent_system_sleep(False)  # Allow sleep initially
        
        while True:
            iteration += 1

            if not in_market_hours(client):
                prevent_system_sleep(False)  # Allow PC to sleep
                sleep_until_market_open(client)
                continue
            
            # Market is open - keep PC awake
            prevent_system_sleep(True)

            log_info(f"=== Iteration {iteration} ===")
            
            if is_multi_stock:
                # Multi-stock logic
                current_positions = portfolio.get_all_positions()
                held_symbols = list(current_positions.keys())
                total_invested = portfolio.get_total_market_value()
                
                log_info(f"Positions: {len(held_symbols)}/{args.max_stocks} | Invested: ${total_invested:.2f}/${args.max_cap:.2f}")
                
                # Safety check: If account value drops below max_cap, something is very wrong
                try:
                    account = client.get_account()
                    account_value = float(account.portfolio_value)
                    if account_value < args.max_cap * 0.5:  # Lost more than 50%
                        log_error(f"Account value ${account_value:.2f} dropped below 50% of max cap!")
                        log_error("Emergency shutdown - liquidating all positions")
                        for sym in held_symbols:
                            sell_flow(client, sym)
                        sys.exit(1)
                except:
                    pass
                
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
                        log_info(f"REBALANCE: Sell {sell_sym} -> Buy {buy_sym}")
                        ok, msg = sell_flow(client, sell_sym)
                        if ok:
                            portfolio.remove_position(sell_sym)
                            log_info(f"OK: {msg}")
                
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
                            verbose=True  # Enable to see what's failing
                        )
                        log_info(f"Scanned {len(opportunities)} opportunities (threshold: ${config.PROFITABILITY_MIN_EXPECTED_USD:.2f}/day)")
                        for opp in opportunities:
                            log_info(f"  {opp['symbol']}: ${opp['expected_daily']:.2f}/day (score: {opp['score']:.2f})")
                            if opp["expected_daily"] >= config.PROFITABILITY_MIN_EXPECTED_USD:
                                stocks_to_evaluate.append(opp["symbol"])
                                log_info(f"    -> ADDED to portfolio")
                                if len(stocks_to_evaluate) >= args.max_stocks:
                                    break
                            else:
                                log_info(f"    -> SKIPPED (below ${config.PROFITABILITY_MIN_EXPECTED_USD:.2f} threshold)")
                    except Exception as e:
                        log_error(f"Error scanning stocks: {e}")
                        import traceback
                        log_error(traceback.format_exc())
                        # Continue anyway - don't let scanner errors stop trading
                        stocks_to_evaluate = []
                
                # Smart allocation: calculate capital per stock dynamically
                if use_smart_allocation and stocks_to_evaluate:
                    log_info("Calculating smart capital allocation...")
                    stock_allocations = allocate_capital_smartly(
                        client, stocks_to_evaluate, forced_stocks, 
                        args.max_cap, interval_seconds
                    )
                    
                    # Check if allocation is empty (no good opportunities)
                    if not stock_allocations:
                        log_info("No profitable opportunities found. Holding cash.")
                        continue  # Skip to next iteration
                    
                    # Identify potential sells: stocks we hold but aren't in optimal allocation
                    held_not_optimal = [s for s in held_symbols if s not in stock_allocations]
                    
                    # Consider selling underperformers if we need capital for better stocks
                    available_capital = args.max_cap - total_invested
                    needed_capital = sum(stock_allocations.values())
                    
                    if needed_capital > available_capital and held_not_optimal:
                        log_info(f"  Need ${needed_capital:.2f}, have ${available_capital:.2f} available")
                        log_info(f"  May sell: {', '.join(held_not_optimal)} (not in optimal portfolio)")
                        
                        # Sell weakest positions to free up capital
                        for sym in held_not_optimal:
                            pos = get_position(client, sym)
                            if pos:
                                log_info(f"  REBALANCE: Selling {sym} to reallocate capital")
                                ok, msg = sell_flow(client, sym)
                                if ok:
                                    portfolio.remove_position(sym)
                                    log_info(f"  OK: {msg}")
                                    available_capital += pos["market_value"]
                    
                    # Show allocation breakdown
                    log_info("Target Capital Allocation:")
                    for sym in sorted(stock_allocations.keys(), key=lambda x: stock_allocations[x], reverse=True):
                        has_position = "[HELD]" if sym in held_symbols else "[NEW] "
                        log_info(f"  {has_position} {sym}: ${stock_allocations[sym]:.2f}")
                else:
                    stock_allocations = {sym: cap_per_stock for sym in stocks_to_evaluate}
                
                # Trade each stock
                log_info(f"\nTrading {len(stocks_to_evaluate)} stocks:")
                for i, sym in enumerate(stocks_to_evaluate, 1):
                    try:
                        stock_cap = stock_allocations.get(sym, cap_per_stock)
                        
                        # Check if we already have this position
                        existing_pos = get_position(client, sym)
                        existing_value = existing_pos["market_value"] if existing_pos else 0
                        
                        # Calculate how much MORE capital we can use for this stock
                        additional_cap = max(0, stock_cap - existing_value)
                        
                        if existing_pos:
                            log_info(f"[{i}/{len(stocks_to_evaluate)}] {sym} (holding: ${existing_value:.2f}, target: ${stock_cap:.2f})...")
                        else:
                            log_info(f"[{i}/{len(stocks_to_evaluate)}] {sym} (target: ${stock_cap:.2f})...")
                        
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
                            if existing_pos:
                                if additional_cap >= 10:  # Room to add more
                                    log_info(f"  -- Already holding, could add ${additional_cap:.2f} more")
                                else:
                                    log_info(f"  -- Already at target allocation")
                            else:
                                # New position
                                ok, msg = buy_flow(
                                    client, sym, last_price, stock_cap,
                                    max(0.0, confidence),
                                    args.frac or config.TRADE_SIZE_FRAC_OF_CAP,
                                    args.tp or config.TAKE_PROFIT_PERCENT,
                                    args.sl or config.STOP_LOSS_PERCENT,
                                    dynamic_enabled=not args.no_dynamic,
                                    interval_seconds=interval_seconds
                                )
                                log_info(f"  {'OK' if ok else '--'} {msg}")
                        elif action == "sell":
                            ok, msg = sell_flow(client, sym)
                            log_info(f"  {'OK' if ok else '--'} {msg}")
                        else:
                            if existing_pos:
                                log_info(f"  -- Holding (no signal)")
                            else:
                                log_info(f"  -- No signal")
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
                    log_info(f"{'OK' if ok else '--'} {msg}")
                elif action == "sell":
                    ok, msg = sell_flow(client, symbol)
                    log_info(f"{'OK' if ok else '--'} {msg}\n")
            
            time.sleep(interval_seconds)
    
    except KeyboardInterrupt:
        log_info("Interrupted - shutting down")
        prevent_system_sleep(False)  # Allow PC to sleep
        return 0
    except Exception as e:
        log_error(f"Fatal error: {e}")
        prevent_system_sleep(False)  # Allow PC to sleep
        return 1
    finally:
        prevent_system_sleep(False)  # Always allow PC to sleep on exit


if __name__ == "__main__":
    sys.exit(main())
