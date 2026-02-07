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
        self.config = get_config(reload=True)
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

        # --- Equity cache for market-closed hours (keeps last known total on dashboard) ---
        self._equity_cache_path = 'equity_cache.json'
        self._last_equity = 0.0
        self._last_equity_ts = ''
        self._high_water_mark = 0.0 # Track max equity seen
        self._restricted_mode = False
        self._restricted_mode_start = 0.0
        self._load_equity_cache()
        
        # Initialize Universe immediatley
        self.refresh_universe()

    def _market_is_open(self) -> bool:
        """Return True if market is open.

        Prefer Alpaca clock (handles holidays/half-days). If Alpaca fails,
        fall back to a simple 9:30-16:00 ET weekday window.
        """
        # 1) Alpaca clock (best)
        try:
            clock = self.api.get_clock()
            return bool(getattr(clock, 'is_open', False))
        except Exception:
            pass

        # 2) Fallback time window (ET)
        try:
            tz = pytz.timezone('US/Eastern')
            now_et = datetime.datetime.now(tz)
            # weekday: Mon=0 .. Sun=6
            if now_et.weekday() >= 5:
                return False
            open_min = 9 * 60 + 30
            close_min = 16 * 60
            now_min = now_et.hour * 60 + now_et.minute
            return open_min <= now_min < close_min
        except Exception:
            # If even fallback fails, be safe: treat as closed.
            return False

    def _load_equity_cache(self):
        try:
            if os.path.exists(self._equity_cache_path):
                data = json.load(open(self._equity_cache_path, 'r'))
                self._last_equity = float(data.get('equity', 0.0) or 0.0)
                self._last_equity_ts = str(data.get('timestamp', '') or '')
                self._high_water_mark = float(data.get('high_water_mark', 0.0) or 0.0)
                self._restricted_mode = bool(data.get('restricted_mode', False))
                self._restricted_mode_start = float(data.get('restricted_mode_start', 0.0) or 0.0)
        except Exception:
            pass

    def _save_equity_cache(self, equity: float):
        try:
            self._last_equity = float(equity or 0.0)
            self._last_equity_ts = datetime.datetime.now().isoformat()
            
            # Simple persistence of HWM/restricted state along with equity
            state = {
                'timestamp': self._last_equity_ts,
                'equity': self._last_equity,
                'high_water_mark': self._high_water_mark,
                'restricted_mode': self._restricted_mode,
                'restricted_mode_start': self._restricted_mode_start
            }
            
            with open(self._equity_cache_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def _get_equity_for_dashboard(self) -> float:
        """Best-effort equity for display.

        When market is closed, Alpaca quotes may not update, but account equity is still valid.
        If API fails, fall back to last cached equity.
        """
        try:
            account = self._get_account_cached()
            equity = float(account.equity)
            if self.config.virtual_account_size:
                equity = float(self.config.virtual_account_size)
            self._save_equity_cache(equity)
            return float(equity)
        except Exception:
            try:
                return float(self._last_equity or 0.0)
            except Exception:
                return 0.0

    def liquidate_all(self, reason: str):
        """Emergency: Close all positions immediately."""
        log_error(f"!!! LIQUIDATE ALL TRIGGERED: {reason} !!!")
        try:
            # Check market open? For kill switch, we try regardless.
            positions = self.pm.get_all_positions()
            for symbol in list(positions.keys()): # Copy keys
                log_warn(f"Liquidating {symbol}...")
                self.order_executor.liquidate(symbol, reason)
                self.pm.remove_position(symbol)
                time.sleep(0.5) # Avoid rate limits slightly
        except Exception as e:
            log_error(f"Failed to fully liquidate: {e}")

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
                    try:
                        eq = self._get_equity_for_dashboard()
                    except Exception:
                        eq = 0.0
                    self._dump_dashboard_state(eq, "PAUSED", None)
                    time.sleep(5)
                    continue

                now = time.time()

                # Hard gate: if configured for market-hours-only and market is closed, idle until open.
                # Exiting here leaves the dashboard stuck on MARKET_CLOSED until the *scheduled task* runs again.
                # Idling fixes that and keeps the bot ready to resume as soon as Alpaca reports open.
                if getattr(self.config, 'enable_market_hours_only', True):
                    if not self._market_is_open():
                        log_info('Market closed - idling until open (still updating dashboard state).')
                        try:
                            eq = self._get_equity_for_dashboard()
                            self._dump_dashboard_state(eq, 'MARKET_CLOSED', None)
                        except Exception:
                            pass

                        # Sleep smart: use Alpaca clock.next_open when available; otherwise default to 60s.
                        sleep_s = 60
                        try:
                            clock = self.api.get_clock()
                            ts = getattr(clock, 'timestamp', None)
                            nxt = getattr(clock, 'next_open', None)
                            if ts and nxt:
                                delta = (nxt - ts).total_seconds()
                                # Wake up more frequently as we approach open.
                                sleep_s = int(max(15, min(delta, 300)))
                        except Exception:
                            pass

                        time.sleep(sleep_s)
                        continue

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
            equity = self._get_equity_for_dashboard()
        except Exception:
            equity = 100000.0  # Fallback

        # B. Data Fetching
        try:
            closes, volumes = fetch_closes_with_volume(
                self.api, symbol, 
                interval_seconds=60, 
                limit_bars=200
            )
            # Defensive: ensure closes and volumes are aligned
            try:
                if closes is not None and volumes is not None and len(closes) != len(volumes):
                    n = min(len(closes), len(volumes))
                    closes = closes[-n:]
                    volumes = volumes[-n:]
            except Exception:
                pass

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
                    allocation = self.allocation_engine.calculate_allocation(signal, current_price, equity)

                    # Track desired vs executed action for dashboard/UI
                    try:
                        self._desired_action = str(getattr(signal, 'action', '')).lower()
                    except Exception:
                        self._desired_action = ''
                    self._executed_action = ''
                    self._blocked_reason = ''


                    # Always log allocation decision; prevents misleading BUY logs
                    try:
                        notional = float(getattr(allocation, "target_notional", 0.0) or 0.0)
                        log_info(
                            f"ALLOCATION {symbol}: desired_action={signal.action.upper()} "
                            f"allowed={allocation.is_allowed} qty={allocation.target_quantity} notional=${notional:.2f} "
                            f"value=${allocation.target_value:.2f} reason={allocation.reason}"
                        )
                    except Exception:
                        notional = 0.0

                    can_trade = bool(
                        allocation.is_allowed
                        and ((allocation.target_quantity and allocation.target_quantity > 0) or (notional and notional > 0.0))
                    )

                    if not can_trade:
                        try:
                            # Final pre-exec decision is HOLD (blocked by fees/risk/constraints)
                            self._desired_action = "hold"
                            self._executed_action = "hold"
                            self._blocked_reason = getattr(allocation, "reason", "")
                        except Exception:
                            pass
                        # Treat as HOLD so dashboard/logs don't claim we bought when fees/constraints blocked it
                        signal.action = "hold"
                    else:
                        log_info(f"SIGNAL {symbol}: {signal.action.upper()} (Conf: {signal.confidence:.2f})")
                        try:
                            self._desired_action = str(getattr(signal, 'action', '')).lower()
                            self._executed_action = self._desired_action
                            self._blocked_reason = ''
                        except Exception:
                            pass
                        try:
                            self._executed_action = str(getattr(signal, "action", "")).lower()
                        except Exception:
                            pass
                        placed = self.order_executor.execute_allocation(allocation)
                        if placed:
                            # Update portfolio from Alpaca position if available (avoid assuming instant fills)
                            try:
                                pos = self.api.get_position(symbol)
                                qty = float(pos.qty)
                                avg_entry = float(getattr(pos, 'avg_entry_price', current_price))
                                mv = float(getattr(pos, 'market_value', allocation.target_value))
                                upl = float(getattr(pos, 'unrealized_pl', 0.0))
                                self.pm.update_position(symbol, qty, avg_entry, mv, upl, confidence=signal.confidence)
                            except Exception:
                                # If position isn't visible yet, record intended allocation
                                intended_value = allocation.target_value if allocation.target_value else notional
                                intended_qty = allocation.target_quantity if allocation.target_quantity else (notional / current_price if current_price else 0.0)
                                self.pm.update_position(symbol, intended_qty, current_price, intended_value, 0.0, confidence=signal.confidence)
                        else:
                            log_warn(f"ORDER NOT PLACED for {symbol} (see earlier warnings).")
                            try:
                                self._executed_action = "hold"
                                self._blocked_reason = "order_not_placed"
                            except Exception:
                                pass

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
            
            try:
                self._desired_action = str(getattr(signal, "action", "")).lower()
                self._executed_action = self._desired_action
            except Exception:
                pass
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
            

        # Track High Water Mark (only goes up)
        if equity > self._high_water_mark:
            self._high_water_mark = equity
            # If we were in restricted mode and recovered, check resumption
        
        # Define Kill Levels based on HWM (15% Hard, 10% Soft)
        # Assuming HWM starts at initial capital (e.g. 100k) or grows.
        if self._high_water_mark <= 0.1:
             self._high_water_mark = equity # Initialize if 0
             
        hard_kill_2_level = self._high_water_mark * 0.85 # 15% Drawdown
        dynamic_floor = self._high_water_mark * 0.90 # 10% Drawdown
        
        # Kill switch logic
        try:
            # 1. Hard Kill 1: Fixed Floor (Original)
            fixed_floor = float(getattr(self.config, 'max_cap_usd', 0.0) or 0.0)
            if fixed_floor > 0 and equity < fixed_floor:
                self.liquidate_all(f"Hard Kill 1: Equity ${equity:.2f} < Fixed Floor ${fixed_floor:.2f}")
                self._dump_dashboard_state(equity, 'KILL_SWITCH_FIXED', None)
                raise SystemExit(2)

            # 2. Hard Kill 2: 15% Drawdown (Dynamic)
            if equity < hard_kill_2_level:
                self.liquidate_all(f"Hard Kill 2: Equity ${equity:.2f} < 15% Drawdown (Limit: ${hard_kill_2_level:.2f})")
                self._dump_dashboard_state(equity, 'KILL_SWITCH_DYNAMIC', None)
                raise SystemExit(2)

            # 3. Soft Kill: 10% Drawdown -> Restricted Mode
            if equity < dynamic_floor and not self._restricted_mode:
                self.liquidate_all(f"Soft Kill: Equity ${equity:.2f} < 10% Drawdown (Limit: ${dynamic_floor:.2f})")
                self._restricted_mode = True
                self._restricted_mode_start = time.time()
                log_warn("ENTERING RESTRICTED MODE. Pausing for 5 minutes.")
                self._save_equity_cache(equity) # Save state immediately
                
            # 4. Restricted Mode Logic
            if self._restricted_mode:
                # Check for recovery (must be > floor + 25 buffer to avoid oscillation)
                # Note: "floor" here refers to the 10% drawdown level.
                if equity > (dynamic_floor + 25.0):
                    self._restricted_mode = False
                    log_info("Restricted Mode: RECOVERED! Resuming normal operation.")
                    self._save_equity_cache(equity)
                else:
                    # Cooldown check
                    elapsed = time.time() - self._restricted_mode_start
                    if elapsed < 300: # 5 minutes
                         log_info(f"Restricted Mode: Warming up... {300 - elapsed:.0f}s remaining.")
                         self._dump_dashboard_state(equity, "RESTRICTED_COOLDOWN", None)
                         return # Skip trading cycle
                    
                    # If cooldown done, we proceed but with Max Exposure Cap
                    log_warn(f"Restricted Mode Active: Trading capped at 10% of Equity (${equity * 0.10:.2f})")

        except SystemExit:
            raise
        except Exception as e:
            log_error(f"Kill switch error: {e}")
            
        log_info(f"Account Equity: ${equity:.2f} (Cash: ${cash:.2f}) | HWM: ${self._high_water_mark:.2f}")
        
        # 2. Get Universe
        # Phase 13: Use Dynamic Active Universe
        if not self.active_universe:
             self.refresh_universe()
        
        universe = self.active_universe
        
        # 3. Process Each Symbol
        # Calculate restricted cap if needed
        max_exposure_cap = None
        if self._restricted_mode:
            max_exposure_cap = equity * 0.10 # 10% of current equity

        for symbol in universe:
            try:
                self.process_symbol(symbol, equity, max_exposure_cap)
            except Exception as e:
                log_error(f"Error processing {symbol}: {e}")
    
    def process_symbol(self, symbol: str, total_equity: float, max_exposure_cap: float = None):
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
        allocation = self.allocation_engine.calculate_allocation(signal, current_price, total_equity, max_exposure_cap=max_exposure_cap)
        
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
                "runner_patch": "RUNNER_PATCH_20260205_103716",
                "equity": equity,
                "last_equity_timestamp": getattr(self, "_last_equity_ts", ""),
                "virtual_account_size": self.config.virtual_account_size,
                "kill_switch_floor_usd": float(getattr(self.config, "max_cap_usd", 0.0) or 0.0),
                "next_cycle_time": next_time_iso,
                "last_symbol": last_symbol,
                "last_action": last_signal.action if last_signal else "idle",
                "last_confidence": last_signal.confidence if last_signal else 0.0,
                "active_cycle": not os.path.exists("bot.pause"), # Real status
                "positions": self.pm.get_all_positions(),
                "high_water_mark": self._high_water_mark,
                "dynamic_floor_level": self._high_water_mark * 0.90,
                "hard_kill_2_level": self._high_water_mark * 0.85,
                "restricted_mode": self._restricted_mode,
                "restricted_mode_start": self._restricted_mode_start
            }
            with open("dashboard_state.json", "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log_error(f"Failed to dump dashboard state: {e}")

if __name__ == "__main__":
    bot = SmartTradingBot()
    bot.run()













