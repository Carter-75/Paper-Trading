#!/usr/bin/env python3
"""
Smart Paper Trading Bot - Main Runner
Orchestrates the Decision, Allocation, and Execution engines.
"""

import time
import logging
import sys
import argparse
import traceback
import os
from typing import List, Any

# Import our new modules
from config_validated import get_config
from strategies.decision_engine import DecisionEngine
import stock_scanner # Phase 13: Dynamic Universe
from risk.allocation_engine import AllocationEngine
from execution.orders import OrderExecutor
from portfolio_manager import PortfolioManager
from utils.helpers import log_info, log_error, log_warn

# Import data fetching and cache from the old runner logic (preserved here for simplicity)
# In a full refactor, these would go to utils/market_data.py
from runner_data_utils import PriceCache, fetch_closes_with_volume, make_client
from utils.process_lock import ProcessLock
import json
import datetime
import pytz

class SmartTradingBot:
    def __init__(self):
        self.config = get_config()
        self.pm = PortfolioManager()
        
        # Initialize Engines
        self.decision_engine = DecisionEngine()
        self.allocation_engine = AllocationEngine(self.pm)
        
        # Connect to Alpaca
        try:
            self.api = make_client(go_live=self.config.wants_live_mode())
            self.order_executor = OrderExecutor(self.api)
            log_info("Connected to Alpaca API successfully.")
        except Exception as e:
            log_error(f"Failed to connect to Alpaca: {e}")
            sys.exit(1)

        self.schedule = {}
        self.active_universe = [] # Dynamic List
        self.last_universe_refresh = 0
        self._account_cache = None  # (ts, account)
        self._account_cache_ttl = 60  # seconds
        self._error_throttle = {}  # throttle noisy per-symbol exceptions
        
        # Initialize Universe immediatley
        self.refresh_universe()

    def run(self):
        """Main Loop"""
        # Lock check
        self.lock = ProcessLock()
        if not self.lock.acquire(force_kill=True):
            log_warn("Another bot instance appears to be running (lock held). Exiting this instance.")
            sys.exit(0)
            
        log_info(f"Starting Smart Trading Bot v17.1 (Highlander Details)")
        log_info("RUNNER_PATCH=20260204_throttle_cache")
        
        # 1. Dynamic Stock Universe
        # Initial population of schedule based on the refreshed universe
        self.schedule = {
            sym: time.time() + (i * 0.5) for i, sym in enumerate(self.active_universe)
        }
        
        api_call_timestamps = []
        
        while True:
            try:
                # 0. Check Pause
                if os.path.exists("bot.pause"):
                    log_info("Bot is PAUSED. (Waiting for resume...)")
                    self._dump_dashboard_state(0, "PAUSED", None)
                    time.sleep(5)
                    continue

                now = time.time()
                
                # Check Market Hours
                if not self.config.wants_live_mode():
                     # Paper trading: We can simulate hours or just run 24/7.
                     # For now, let's respect "Opening Guard" even in paper to test it.
                     pass 
                
                # Market Hours Logic (Eastern Time)
                ny_time = datetime.datetime.now(pytz.timezone('US/Eastern'))
                
                # 1. Closed at Night (8pm - 4am) - Sleep to save API
                if ny_time.hour >= 20 or ny_time.hour < 4:
                     # log_info("Market closed (Night Mode). Sleeping 60s...") 
                     time.sleep(60)
                if ny_time.hour >= 20 or ny_time.hour < 4:
                     # log_info("Market closed (Night Mode). Sleeping 60s...") 
                     time.sleep(60)
                     continue
                
                # Morning Routine (4 AM - 5 AM): Check if we need to learn
                if ny_time.hour == 4 and ny_time.minute < 10:
                    # Only try once per day (could use a flag, but check_and_retrain checks file age so it's safe to call often)
                    # We'll just call it. If it retrains, it takes 2 mins.
                    # To avoid spamming it every loop for 10 mins, we rely on check_and_retrain returning fast if fresh.
                    try:
                        if self.decision_engine.ml_predictor.check_and_retrain():
                            log_info("Morning Training Complete! ????")
                    except Exception as e:
                        log_error(f"Morning Training check failed: {e}")
                
                # 2. Opening Guard (9:30 - 9:45 AM)
                # Volatility is extreme. We pause or reduce size.
                is_opening_chaos = (ny_time.hour == 9 and 30 <= ny_time.minute < 45)
                if is_opening_chaos:
                     # Option A: Skip trading
                     log_info("Opening Guard Active (9:30-9:45 AM). Pausing for volatility settlement.")
                     time.sleep(60)
                     continue
                     # Option B: Trade with reduced size (implemented in AllocationEngine if we pass a flag)
                
                # 1. Clean API rate limit history (Keep only last 60s)
                api_call_timestamps = [t for t in api_call_timestamps if now - t < 60]
                
                # 2. Find next symbol due
                # Sort by timestamp (asc)
                due_symbols = sorted(self.schedule.items(), key=lambda x: x[1])
                next_sym, next_time = due_symbols[0]
                
                # 3. Wait if too early
                if next_time > now:
                    sleep_needed = next_time - now
                    time.sleep(min(sleep_needed, 1.0)) 
                    continue
                    
                    
                # 4. Check Rate Limit
                if len(api_call_timestamps) >= self.config.max_api_calls_per_min:
                    log_warn("Rate limit approaching. Throttling...")
                    time.sleep(1)
                    continue
                    
                # Phase 13: Periodic Universe Refresh (every 4 hours)
                if time.time() - self.last_universe_refresh > (4 * 3600):
                    self.refresh_universe()
                    # Re-populate schedule for new symbols
                    for sym in self.active_universe:
                        if sym not in self.schedule:
                            self.schedule[sym] = 0 # ASAP

                # 5. Process Symbol start
                # (Existing logic...)
                log_info(f"Processing {next_sym} (Lag: {now - next_time:.1f}s)")
                    
                # 5. Process Symbol
                api_call_timestamps.append(now)
                self.process_symbol_adaptive(next_sym)
                
            except KeyboardInterrupt:
                log_info("Bot stopped by user.")
                self.lock.release()
                break
            except Exception as e:
                # Exponential Backoff for Network/API errors
                import math
                consecutive_errors = locals().get('consecutive_errors', 0) + 1
                wait_time = min(300, 10 * (2 ** (consecutive_errors - 1)))
                
                log_error(f"Unhandled exception in main loop: {e}")
                log_error(f"Retrying in {wait_time}s... (Error #{consecutive_errors})")
                log_info(traceback.format_exc())
                time.sleep(wait_time) 
                
            else:
                # Reset error count on success
                consecutive_errors = 0

    def process_symbol_adaptive(self, symbol: str):
        # A. Account Update
        try:
            account = self._get_account_cached()
            equity = float(account.equity)
            if self.config.virtual_account_size:
                equity = self.config.virtual_account_size
        except Exception:
            equity = 100000.0 # Fallback

        # B. Data Fetching
        try:
            closes, volumes = fetch_closes_with_volume(
                self.api, symbol, 
                interval_seconds=60, 
                limit_bars=200
            )
            if closes:
                current_price = float(closes[-1])
            else:
                return
        except Exception as e:
            log_error(f"Data fetch failed for {symbol}: {e}")
            return

        # --- RISK A: BODYGUARD CHECK ---
        current_pos = self.pm.get_position(symbol)
        
        # Calculate ATR safe
        atr = 0.0
        try:
            raw_atr = self.decision_engine.calculate_atr(closes)
            if isinstance(raw_atr, (list, tuple, bytes, str)):
                atr = float(raw_atr[0])
            else:
                atr = float(raw_atr)
        except:
            atr = current_price * 0.02 # Fallback 2%

        if current_pos and current_price > 0:
            avg_entry = current_pos.get('avg_entry', 0.0)
            qty = current_pos.get('qty', 0)
            
            if qty > 0 and avg_entry > 0:
                pnl_pct = (current_price - avg_entry) / avg_entry
                
                stop_price = avg_entry * (1.0 - (self.config.stop_loss_percent / 100.0))
                
                if atr > 0:
                    vol_stop = avg_entry - (2.0 * atr)
                    stop_price = min(stop_price, vol_stop)
                    
                if pnl_pct > 0.02:
                    be_stop = avg_entry * 1.001
                    stop_price = max(stop_price, be_stop)
                    
                reason = ""
                if current_price < stop_price:
                     reason = f"STOP LOSS HIT"
                elif pnl_pct > (self.config.take_profit_percent / 100.0):
                     reason = f"TAKE PROFIT HIT"
                
                if reason:
                    log_warn(f"??????? BODYGUARD TRIGGER: {symbol} -> {reason}")
                    if self.order_executor.liquidate(symbol, reason):
                        self.pm.log_closed_trade(symbol, avg_entry, current_price, qty, reason)
                        self.pm.remove_position(symbol)
                        self.schedule[symbol] = time.time() + 300 
                        return

        # C. Analyze Strategy
        try:
            signal = None
            if closes and len(closes) > 50:
                signal = self.decision_engine.analyze(symbol, closes, volumes)
                
                if signal.action != "hold":
                    log_info(f"SIGNAL {symbol}: {signal.action.upper()} (Conf: {signal.confidence:.2f})")
                    allocation = self.allocation_engine.calculate_allocation(signal, current_price, equity)
                    
                    if allocation.is_allowed and allocation.target_quantity > 0:
                        self.order_executor.execute_allocation(allocation)
                        self.pm.update_position(symbol, allocation.target_quantity, current_price, allocation.target_value, 0.0, confidence=signal.confidence)
                        
            # D. Determine Next Interval
            new_interval = self.config.default_interval_seconds
            computed_interval = new_interval  # default; will be updated
            
            if self.pm.get_position(symbol):
                new_interval = 30
                
            # High Volatility Check (Safe)
            if atr > (current_price * 0.01):
                 new_interval = min(new_interval, 15)
            
            # Update Schedule
            computed_interval = new_interval # ALIAS for Ghost Compatibility
            self.schedule[symbol] = time.time() + computed_interval
            
            self._dump_dashboard_state(equity, symbol, signal, next_updates=self.schedule)
            
        except Exception as e:
            # Throttled error logging so we can debug without log spam
            if not hasattr(self, '_error_throttle'): self._error_throttle = {}
            key = f"{symbol}:{type(e).__name__}:{str(e)[:80]}"
            now_ts = time.time()
            last_ts = self._error_throttle.get(key, 0)
            if now_ts - last_ts > 300:
                self._error_throttle[key] = now_ts
                try:
                    log_error(f"Error adaptive processing {symbol}: {e}")
                    log_error(traceback.format_exc())
                    try: log_error(f"Context {symbol}: len(closes)={len(closes) if closes else None} len(volumes)={len(volumes) if volumes else None}")
                    except Exception: pass
                except Exception: pass
            self.schedule[symbol] = time.time() + 60
    def _get_account_cached(self):
        """Return Alpaca account with a short TTL to avoid rate limits."""
        now = time.time()
        if self._account_cache and (now - self._account_cache[0]) < self._account_cache_ttl:
            return self._account_cache[1]
        try:
            acct = self.api.get_account()
        except Exception as e:
            # If rate limited or transient error, keep running and reuse stale cache if present
            try:
                msg = str(e)
                log_warn(f"Alpaca get_account failed: {msg}")
            except Exception:
                pass
            if self._account_cache:
                return self._account_cache[1]
            raise
        self._account_cache = (now, acct)
        return acct

    def refresh_universe(self):
        """Phase 13: Update stock list from scanner"""
        try:
            log_info("Refreshing Top 50 Stock Universe...")
            # Fetch dynamic list (cached internally by scanner for 24h)
            self.active_universe = stock_scanner.get_stock_universe(use_top_100=True, force_refresh=False)
            # Limit to 50 for performance (Phase 13 scalability step 1)
            self.active_universe = self.active_universe[:50]
            self.last_universe_refresh = time.time()
            log_info(f"Universe Updated: {len(self.active_universe)} symbols loaded.")
        except Exception as e:
            log_error(f"Failed to refresh universe: {e}")
            # Fallback
            self.active_universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "SPY", "QQQ"]

    def run_cycle(self):
        """Single Trading Cycle"""
        log_info("=== Starting Trading Cycle ===")
        
        # 0. Check Pause
        if os.path.exists("bot.pause"):
            log_info("Bot is PAUSED. (Waiting for resume...)")
            self._dump_dashboard_state(0, "PAUSED", None) # Update state to show paused
            return

        # 1. Update Portfolio
        account = self._get_account_cached()
        equity = float(account.equity)
        cash = float(account.cash)
        
        if self.config.virtual_account_size:
            log_info(f"Using Virtual Account Size: ${self.config.virtual_account_size:.2f} (Real: ${equity:.2f})")
            equity = self.config.virtual_account_size
            
        log_info(f"Account Equity: ${equity:.2f} (Cash: ${cash:.2f})")
        
        # 2. Get Universe
        # Phase 13: Use Dynamic Active Universe
        if not self.active_universe:
             self.refresh_universe()
        
        universe = self.active_universe
        
        # 3. Process Each Symbol
        for symbol in universe:
            try:
                self.process_symbol(symbol, equity)
            except Exception as e:
                log_error(f"Error processing {symbol}: {e}")
    
    def process_symbol(self, symbol: str, total_equity: float):
        # A. Fetch Data
        # We need ~150 bars for good technicals + ML
        closes, volumes = fetch_closes_with_volume(
            self.api, 
            symbol, 
            interval_seconds=int(self.config.default_interval_seconds), 
            limit_bars=200
        )
        
        if not closes or len(closes) < 50:
            return

        # B. Make Decision (The "Brain")
        signal = self.decision_engine.analyze(symbol, closes, volumes)
        
        if signal.action == "hold":
            # Optional: Log hold reasoning if verbose
            return
            
        log_info(f"SIGNAL {symbol}: {signal.action.upper()} (Conf: {signal.confidence:.2f})")
        for r in signal.reasoning:
            log_info(f"  > {r}")

        # C. Calculate Allocation (The "Banker")
        current_price = closes[-1]
        allocation = self.allocation_engine.calculate_allocation(signal, current_price, total_equity)
        
        if not allocation.is_allowed:
            log_info(f"  Allocation denied: {allocation.reason}")
            return
            
        # D. Execute (The "Hands")
        if allocation.target_quantity > 0:
            self.order_executor.execute_allocation(allocation)
            
            # Update local portfolio state
            self.pm.update_position(
                symbol, 
                allocation.target_quantity, 
                current_price, 
                allocation.target_value, 
                0.0 # PnL update happens next cycle
            )
        
        # E. Dump Dashboard State
        self._dump_dashboard_state(total_equity, symbol, signal)

    def _dump_dashboard_state(self, equity: float, last_symbol: str = "", last_signal: Any = None, next_updates: dict = None):
        """Write current state to JSON for web dashboard"""
        try:
            # Find closest next update
            next_time_iso = ""
            if next_updates:
                 soonest = min(next_updates.values())
                 next_time_iso = datetime.datetime.fromtimestamp(soonest).isoformat()
            
            state = {
                "timestamp": datetime.datetime.now().isoformat(),
                "equity": equity,
                "virtual_account_size": self.config.virtual_account_size,
                "next_cycle_time": next_time_iso,
                "last_symbol": last_symbol,
                "last_action": last_signal.action if last_signal else "idle",
                "last_confidence": last_signal.confidence if last_signal else 0.0,
                "active_cycle": not os.path.exists("bot.pause"), # Real status
                "positions": self.pm.get_all_positions()
            }
            with open("dashboard_state.json", "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log_error(f"Failed to dump dashboard state: {e}")

if __name__ == "__main__":
    bot = SmartTradingBot()
    bot.run()







