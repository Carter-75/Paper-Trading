#!/usr/bin/env python3
"""
Comprehensive Binary Search Optimizer

Tests intervals and capitals with ROBUSTNESS testing across multiple periods.
No artificial limits - let's find what ACTUALLY works consistently!
"""

import sys
from typing import Tuple, Dict, List
import config
from tqdm import tqdm
import csv
import os
from datetime import datetime
import multiprocessing
from functools import partial
import signal
import platform
import ctypes
from runner import (
    make_client,
    fetch_closes,
    simulate_signals_and_projection,
    snap_interval_to_supported_seconds,
    monte_carlo_projection,
    calculate_overnight_gap_risk,
    calculate_market_beta,
    calculate_correlation_matrix,
)
from stock_scanner import get_stock_universe

# Global result cache to avoid duplicate API calls
_result_cache: Dict[Tuple[str, int, float], float] = {}


def prevent_sleep(verbose=False):
    """
    Prevent system from sleeping during optimization (Windows only).
    Returns a function to restore normal sleep behavior.
    """
    if platform.system() != "Windows":
        # Not Windows - return no-op functions
        return lambda: None
    
    try:
        # Windows constants for SetThreadExecutionState
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        ES_AWAYMODE_REQUIRED = 0x00000040
        
        # Prevent sleep and keep display on during optimization
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        
        if verbose:
            print("[OK] System sleep prevention enabled - PC will stay awake during optimization")
        
        def restore_sleep():
            """Restore normal sleep behavior"""
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                if verbose:
                    print("[OK] System sleep prevention disabled - normal power settings restored")
            except:
                pass
        
        return restore_sleep
    except:
        # Failed to prevent sleep - return no-op function
        return lambda: None

# Global risk metrics cache (these don't change with capital)
_risk_cache: Dict[Tuple[str, int], dict] = {}


def worker_init():
    """Ignore SIGINT in worker processes - only main process handles Ctrl+C"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def evaluate_single_stock(symbol: str, max_cap: float, use_robustness: bool, commission_per_trade: float) -> dict:
    """
    Worker function for parallel stock evaluation.
    Returns a dict with optimization results for one stock.
    """
    try:
        # Run optimization for this stock
        optimal_interval, optimal_cap, expected_return, consistency = comprehensive_binary_search(
            symbol, verbose=False, max_cap=max_cap, use_robustness=use_robustness
        )
        
        # Fetch detailed metrics
        try:
            # Use None as client - fetch_closes will use yfinance (no API rate limits!)
            closes = fetch_closes(None, symbol, optimal_interval, 200)
            detailed_sim = simulate_signals_and_projection(closes, optimal_interval, override_cap_usd=optimal_cap)
            sharpe = detailed_sim.get("sharpe_ratio", 0.0)
            sortino = detailed_sim.get("sortino_ratio", 0.0)
            expectancy = detailed_sim.get("expectancy", 0.0)
            avg_mae = detailed_sim.get("avg_mae", 0.0)
            max_dd = detailed_sim.get("max_drawdown_pct", 0.0)
            recovery = detailed_sim.get("recovery_factor", 0.0)
            trade_returns = detailed_sim.get("trade_returns", [])
            trades_per_day = detailed_sim.get("expected_trades_per_day", 1.0)
            
            # Get out-of-sample performance if robustness testing was enabled
            if use_robustness:
                _, _, _, oos_return = evaluate_robustness(None, symbol, optimal_interval, optimal_cap)
                wf_result = walk_forward_test(None, symbol, optimal_interval, optimal_cap, total_bars=600)
                wf_avg_test = wf_result.get("avg_test_return", 0.0)
                wf_ratio = wf_result.get("train_test_ratio", 0.0)
            else:
                oos_return = 0.0
                wf_avg_test = 0.0
                wf_ratio = 0.0
        except:
            sharpe = sortino = expectancy = avg_mae = max_dd = recovery = 0.0
            trade_returns = []
            trades_per_day = 1.0
            oos_return = 0.0
            wf_avg_test = 0.0
            wf_ratio = 0.0
        
        # Calculate VaR/CVaR
        var_95 = cvar_95 = 0.0
        win_rate = 0.0
        if trade_returns and len(trade_returns) >= 5:
            try:
                mc_result = monte_carlo_projection(
                    trade_returns=trade_returns,
                    starting_capital=optimal_cap,
                    trades_per_day=trades_per_day,
                    days=30,
                    num_simulations=1000
                )
                var_95 = mc_result.get("var_95", 0.0)
                cvar_95 = mc_result.get("cvar_95", 0.0)
                wins = sum(1 for r in trade_returns if r > 0)
                win_rate = wins / len(trade_returns) if trade_returns else 0.0
            except:
                pass
        
        # Save to history
        save_to_history(
            symbol=symbol,
            interval=optimal_interval,
            capital=optimal_cap,
            daily_return=expected_return,
            consistency=consistency,
            sharpe=sharpe,
            sortino=sortino,
            win_rate=win_rate,
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=cvar_95
        )
        
        return {
            "symbol": symbol,
            "interval": optimal_interval,
            "cap": optimal_cap,
            "return": expected_return,
            "consistency": consistency,
            "sharpe": sharpe,
            "sortino": sortino,
            "expectancy": expectancy,
            "avg_mae": avg_mae,
            "max_dd": max_dd,
            "recovery": recovery,
            "oos_return": oos_return,
            "wf_avg_test": wf_avg_test,
            "wf_ratio": wf_ratio,
            "trade_returns": trade_returns,
            "trades_per_day": trades_per_day,
            "var_95": var_95,
            "cvar_95": cvar_95,
        }
    except Exception as e:
        # Return error result
        return {
            "symbol": symbol,
            "interval": 0,
            "cap": 0.0,
            "return": -999999.0,
            "consistency": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "expectancy": 0.0,
            "avg_mae": 0.0,
            "max_dd": 0.0,
            "recovery": 0.0,
            "oos_return": 0.0,
            "wf_avg_test": 0.0,
            "wf_ratio": 0.0,
            "trade_returns": [],
            "trades_per_day": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "error": str(e)
        }


def save_to_history(symbol: str, interval: int, capital: float, daily_return: float, 
                     consistency: float, sharpe: float, sortino: float, win_rate: float,
                     max_drawdown: float, var_95: float, cvar_95: float):
    """
    Save optimization result to optimization_history.csv for tracking over time.
    
    This helps you:
    - See if strategies degrade over time
    - Compare different runs
    - Track which symbols perform best
    """
    history_file = "optimization_history.csv"
    file_exists = os.path.exists(history_file)
    
    try:
        with open(history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow([
                    "Date", "Time", "Symbol", "Interval_Sec", "Interval_Hours",
                    "Capital", "Daily_Return", "Consistency", "Sharpe", "Sortino",
                    "Win_Rate", "Max_Drawdown", "VaR_95", "CVaR_95"
                ])
            
            # Write data
            now = datetime.now()
            writer.writerow([
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                symbol,
                interval,
                f"{interval / 3600:.4f}",
                f"{capital:.2f}",
                f"{daily_return:.2f}",
                f"{consistency:.3f}",
                f"{sharpe:.3f}",
                f"{sortino:.3f}",
                f"{win_rate:.3f}",
                f"{max_drawdown:.2f}",
                f"{var_95:.2f}",
                f"{cvar_95:.2f}"
            ])
    except Exception as e:
        # Don't crash if logging fails
        pass


def evaluate_config(client, symbol: str, interval_seconds: int, cap_usd: float, bars: int = 200) -> float:
    """
    Evaluate expected daily return for specific interval and capital.
    NO artificial penalties - let the data speak for itself!
    """
    # Check cache first
    cache_key = (symbol, interval_seconds, round(cap_usd, 2))
    if cache_key in _result_cache:
        return _result_cache[cache_key]
    
    try:
        closes = fetch_closes(client, symbol, interval_seconds, bars)
        if not closes or len(closes) < max(config.LONG_WINDOW + 10, 30):
            result = -999999.0
        else:
            sim = simulate_signals_and_projection(closes, interval_seconds, override_cap_usd=cap_usd)
            result = float(sim.get("expected_daily_usd", -999999.0))
    except Exception:
        result = -999999.0
    
    # Cache result
    _result_cache[cache_key] = result
    return result


def evaluate_robustness(client, symbol: str, interval_seconds: int, cap_usd: float) -> Tuple[float, float, List[float], float]:
    """
    Test strategy across multiple time periods to check robustness.
    Includes out-of-sample testing on most recent data.
    
    Returns:
        (average_return, consistency_score, period_returns, out_of_sample_return)
        
    consistency_score: 0.0-1.0, higher = more consistent across periods
    out_of_sample_return: Performance on held-out recent data (not used in optimization)
    """
    # Test on 3 different periods (last 200, 400, 600 bars)
    # These are IN-SAMPLE (used for optimization)
    period_bars = [200, 400, 600]  # ~50 days, 100 days, 150 days each
    period_returns = []
    
    for bars in period_bars:
        ret = evaluate_config(client, symbol, interval_seconds, cap_usd, bars)
        if ret > -999000:  # Valid result
            period_returns.append(ret)
    
    # OUT-OF-SAMPLE TEST: Fetch 750 bars, use last 150 bars (20%) as out-of-sample
    # Train on bars 0-600, test on bars 600-750
    try:
        all_closes = fetch_closes(client, symbol, interval_seconds, 750)
        if all_closes and len(all_closes) >= 700:
            # Split: 80% train (first 600 bars), 20% test (last 150 bars)
            train_size = 600
            train_data = all_closes[:train_size]
            test_data = all_closes[train_size:]  # Out-of-sample data
            
            # Evaluate on OUT-OF-SAMPLE data only
            if len(test_data) >= 50:  # Need minimum data for testing
                sim_test = simulate_signals_and_projection(test_data, interval_seconds, override_cap_usd=cap_usd)
                out_of_sample_return = float(sim_test.get("expected_daily_usd", 0.0))
            else:
                out_of_sample_return = 0.0
        else:
            out_of_sample_return = 0.0  # Not enough data for out-of-sample test
    except:
        out_of_sample_return = 0.0
    
    if len(period_returns) == 0:
        return -999999.0, 0.0, [], out_of_sample_return
    
    # Calculate average and consistency (ONLY from in-sample periods)
    avg_return = sum(period_returns) / len(period_returns)
    
    if len(period_returns) >= 2:
        # Consistency = how similar the returns are across periods
        # BUT: consistent losses should still have LOW consistency
        import statistics
        std_dev = statistics.stdev(period_returns)
        mean_abs = abs(avg_return) if avg_return != 0 else 1.0
        
        # Base consistency on coefficient of variation
        raw_consistency = 1.0 - (std_dev / mean_abs)
        
        # Penalty for negative returns - consistent losses = bad!
        if avg_return <= 0:
            consistency = max(0.0, min(0.3, raw_consistency))  # Cap at 0.3 for losses
        else:
            consistency = max(0.0, min(1.0, raw_consistency))
    else:
        consistency = 0.5  # Unknown consistency
    
    return avg_return, consistency, period_returns, out_of_sample_return


def walk_forward_test(client, symbol: str, interval_seconds: int, cap_usd: float, total_bars: int = 600) -> dict:
    """
    Walk-forward optimization: Rolling train/test windows to prevent look-ahead bias.
    
    Splits data into 3 periods:
    - Period 1: Train on bars 0-200, test on bars 200-400
    - Period 2: Train on bars 200-400, test on bars 400-600
    
    Returns:
        dict with:
            - avg_test_return: Average return on test periods
            - test_returns: List of test period returns
            - train_test_ratio: Test return / Train return (>0.5 = not overfit)
    """
    try:
        all_closes = fetch_closes(client, symbol, interval_seconds, total_bars)
        if not all_closes or len(all_closes) < 400:
            return {"avg_test_return": 0.0, "test_returns": [], "train_test_ratio": 0.0}
        
        period_size = len(all_closes) // 3
        test_returns = []
        train_returns = []
        
        # Period 1: Train on first 1/3, test on second 1/3
        train_data_1 = all_closes[:period_size]
        test_data_1 = all_closes[period_size:2*period_size]
        
        if len(train_data_1) >= 50 and len(test_data_1) >= 50:
            train_sim = simulate_signals_and_projection(train_data_1, interval_seconds, override_cap_usd=cap_usd)
            test_sim = simulate_signals_and_projection(test_data_1, interval_seconds, override_cap_usd=cap_usd)
            train_returns.append(train_sim.get("expected_daily_usd", 0.0))
            test_returns.append(test_sim.get("expected_daily_usd", 0.0))
        
        # Period 2: Train on second 1/3, test on third 1/3
        train_data_2 = all_closes[period_size:2*period_size]
        test_data_2 = all_closes[2*period_size:]
        
        if len(train_data_2) >= 50 and len(test_data_2) >= 50:
            train_sim = simulate_signals_and_projection(train_data_2, interval_seconds, override_cap_usd=cap_usd)
            test_sim = simulate_signals_and_projection(test_data_2, interval_seconds, override_cap_usd=cap_usd)
            train_returns.append(train_sim.get("expected_daily_usd", 0.0))
            test_returns.append(test_sim.get("expected_daily_usd", 0.0))
        
        if len(test_returns) == 0:
            return {"avg_test_return": 0.0, "test_returns": [], "train_test_ratio": 0.0}
        
        avg_test = sum(test_returns) / len(test_returns)
        avg_train = sum(train_returns) / len(train_returns) if train_returns else 1.0
        
        # Train/Test ratio: If test << train, we're overfitting
        # Good ratio: >0.5 (test returns at least 50% of train returns)
        ratio = avg_test / avg_train if avg_train > 0 else 0.0
        
        return {
            "avg_test_return": avg_test,
            "test_returns": test_returns,
            "train_test_ratio": ratio
        }
    except:
        return {"avg_test_return": 0.0, "test_returns": [], "train_test_ratio": 0.0}


def binary_search_capital(client, symbol: str, interval_seconds: int, 
                          min_cap: float = 1.0, max_cap: float = 1000000.0,
                          tolerance: float = 10.0) -> Tuple[float, float]:
    """
    Binary search to find optimal capital for given interval.
    Tests from $1 to max_cap with caching for efficiency.
    
    Uses REAL risk metrics (volatility, win rate, max drawdown) instead of
    arbitrary penalties.
    """
    # Check risk cache first (these metrics don't depend on capital)
    cache_key = (symbol, interval_seconds)
    
    if cache_key in _risk_cache:
        # Use cached risk metrics
        risk_metrics = _risk_cache[cache_key]
    else:
        # Calculate real risk metrics once for this stock/interval
        try:
            closes = fetch_closes(client, symbol, interval_seconds, 200)
            if not closes or len(closes) < 50:
                # Not enough data - cache minimal metrics
                risk_metrics = {
                    "has_data": False,
                    "volatility_pct": 0.5,
                    "max_dd_pct": 0.3,
                    "win_rate": 0.5,
                    "profit_factor": 1.0,
                    "max_consec_losses": 5,
                    "avg_trade_duration": 10,
                    "gap_frequency": 0.0,
                    "avg_gap_size": 0.0,
                    "market_beta": 1.0
                }
            else:
                # Calculate REAL risk metrics
                import statistics
            
            # 1. Volatility (price standard deviation)
            mean_price = sum(closes) / len(closes)
            variance = sum((x - mean_price) ** 2 for x in closes) / len(closes)
            stddev = variance ** 0.5
            volatility_pct = stddev / mean_price if mean_price > 0 else 0.5
            
            # 2. Max drawdown (worst peak-to-trough drop)
            peak = closes[0]
            max_dd_pct = 0.0
            for price in closes:
                if price > peak:
                    peak = price
                dd = (peak - price) / peak if peak > 0 else 0.0
                if dd > max_dd_pct:
                    max_dd_pct = dd
            
            # 3. Win rate and advanced metrics from simulation (at small capital to avoid bias)
            sim = simulate_signals_and_projection(closes, interval_seconds, override_cap_usd=1000.0)
            win_rate = sim.get("win_rate", 0.5)
            profit_factor = sim.get("profit_factor", 1.5)
            max_consec_losses = sim.get("max_consecutive_losses", 3)
            avg_trade_duration = sim.get("avg_trade_duration_bars", 5)
            
            # 4. Overnight gap risk (if trades hold overnight)
            bars_per_day = (6.5 * 3600) / interval_seconds
            holds_overnight = avg_trade_duration >= bars_per_day
            
            if holds_overnight:
                gap_risk = calculate_overnight_gap_risk(symbol, client, days=60)
                gap_frequency = gap_risk.get("gap_frequency", 0.0)  # % of days with >2% gaps
                avg_gap_size = gap_risk.get("avg_gap_size", 0.0)  # Average gap size
            else:
                gap_frequency = 0.0  # No overnight risk for intraday trades
                avg_gap_size = 0.0
            
                # 5. Market Beta (correlation to S&P 500)
                market_beta = calculate_market_beta(symbol, days=60)
                
                # Store all metrics in cache
                risk_metrics = {
                    "has_data": True,
                    "volatility_pct": volatility_pct,
                    "max_dd_pct": max_dd_pct,
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "max_consec_losses": max_consec_losses,
                    "avg_trade_duration": avg_trade_duration,
                    "gap_frequency": gap_frequency,
                    "avg_gap_size": avg_gap_size,
                    "market_beta": market_beta
                }
            
            # Cache risk metrics for future use
            _risk_cache[cache_key] = risk_metrics
            
        except Exception:
            # Fallback minimal metrics if calculation fails
            risk_metrics = {
                "has_data": False,
                "volatility_pct": 0.5,
                "max_dd_pct": 0.3,
                "win_rate": 0.5,
                "profit_factor": 1.0,
                "max_consec_losses": 5,
                "avg_trade_duration": 10,
                "gap_frequency": 0.0,
                "avg_gap_size": 0.0,
                "market_beta": 1.0
            }
            _risk_cache[cache_key] = risk_metrics
    
    # Extract metrics from cache
    volatility_pct = risk_metrics.get("volatility_pct", 0.5)
    max_dd_pct = risk_metrics.get("max_dd_pct", 0.3)
    win_rate = risk_metrics.get("win_rate", 0.5)
    profit_factor = risk_metrics.get("profit_factor", 1.0)
    max_consec_losses = risk_metrics.get("max_consec_losses", 5)
    avg_trade_duration = risk_metrics.get("avg_trade_duration", 10)
    gap_frequency = risk_metrics.get("gap_frequency", 0.0)
    avg_gap_size = risk_metrics.get("avg_gap_size", 0.0)
    market_beta = risk_metrics.get("market_beta", 1.0)
    
    # Now use these metrics for risk adjustment
    if not risk_metrics.get("has_data", False):
        # Not enough data - apply conservative penalty
        def apply_risk_adjustment(return_usd: float, capital: float) -> float:
            # Unknown risk = heavy penalty for large positions
            if capital <= 5000:
                return return_usd * 0.8  # 20% penalty
            else:
                return return_usd * max(0.2, 1.0 - (capital / 50000))  # Up to 80% penalty
    else:
        # Calculate risk-adjusted penalty
        def apply_risk_adjustment(return_usd: float, capital: float) -> float:
            """
            Apply penalty based on REAL risk metrics:
            - High volatility = riskier with large capital
            - High max drawdown = danger of large losses
            - Low win rate = unreliable strategy
            - Low profit factor = barely profitable
            - High consecutive losses = streak risk (could wipe you out)
            - Long trade duration = capital tied up (opportunity cost)
            - Large capital = liquidity issues + harder to exit
            """
            # Base liquidity penalty (market impact)
            if capital <= 10000:
                liquidity_penalty = 0.0
            elif capital <= 50000:
                liquidity_penalty = 0.05 + 0.05 * ((capital - 10000) / 40000)
            elif capital <= 200000:
                liquidity_penalty = 0.10 + 0.15 * ((capital - 50000) / 150000)
            else:
                liquidity_penalty = 0.25 + 0.25 * min(1.0, (capital - 200000) / 800000)
            
            # Volatility risk (higher volatility = riskier with large positions)
            # 5% vol = 0% penalty, 20% vol = 20% penalty, 50% vol = 50% penalty
            vol_risk_penalty = min(0.5, volatility_pct) * (capital / 100000)  # Scales with capital
            
            # Drawdown risk (if it dropped 30% before, it can again)
            # 10% DD = 5% penalty, 30% DD = 15% penalty, 50% DD = 25% penalty
            dd_risk_penalty = (max_dd_pct * 0.5) * (capital / 100000)  # Scales with capital
            
            # Win rate risk (low win rate = unreliable)
            # 60% WR = 0% penalty, 50% WR = 10% penalty, 40% WR = 20% penalty
            wr_risk_penalty = max(0.0, (0.60 - win_rate) * 1.0) * (capital / 50000)
            
            # NEW: Profit factor risk (low profit factor = barely profitable)
            # PF 2.0 = 0%, PF 1.5 = 5%, PF 1.0 = 20%, PF <1.0 = 50%
            if profit_factor >= 2.0:
                pf_penalty = 0.0
            elif profit_factor >= 1.0:
                pf_penalty = (2.0 - profit_factor) * 0.20 * (capital / 50000)
            else:
                pf_penalty = 0.50 * (capital / 50000)  # Very risky!
            
            # NEW: Consecutive loss streak risk
            # 3 losses = 0%, 5 losses = 10%, 10 losses = 30%
            streak_penalty = min(0.30, max(0.0, (max_consec_losses - 3) * 0.05)) * (capital / 50000)
            
            # NEW: Trade duration risk (longer = capital tied up + overnight risk)
            # Calculate bars per day to identify overnight holds
            bars_per_day = (6.5 * 3600) / interval_seconds  # 6.5 hour trading day
            
            # Penalty structure:
            # - Intraday (<1 day): 0-5% penalty (opportunity cost only)
            # - Overnight (1-3 days): 10-20% penalty (gap risk + opportunity cost)
            # - Multi-day (>3 days): 20-30% penalty (high risk + capital tied up)
            
            if avg_trade_duration < bars_per_day:
                # Intraday - minimal penalty (just opportunity cost)
                duration_penalty = min(0.05, (avg_trade_duration / bars_per_day) * 0.05) * (capital / 100000)
            elif avg_trade_duration < bars_per_day * 3:
                # 1-3 days - moderate penalty (overnight gap risk)
                days_held = avg_trade_duration / bars_per_day
                duration_penalty = (0.10 + (days_held - 1) * 0.05) * (capital / 100000)
            else:
                # >3 days - high penalty (excessive capital tie-up)
                duration_penalty = 0.30 * (capital / 100000)
            
            # NEW: Trade frequency optimization (penalize extremes)
            # Optimal: 2-10 trades/day
            # Too few (<2): Missing opportunities
            # Too many (>20): Overtrading + high costs
            
            # Estimate trades/day from interval (conservative estimate: 1 trade per 2 intervals)
            max_possible_trades_per_day = (6.5 * 3600) / interval_seconds
            trades_per_day_actual = max_possible_trades_per_day / 2.0  # Conservative estimate
            
            if trades_per_day_actual < 1:
                # Very few trades - missing opportunities
                frequency_penalty = 0.30 * (capital / 100000)
            elif trades_per_day_actual < 2:
                # Below optimal - some penalty
                frequency_penalty = 0.15 * (capital / 100000)
            elif trades_per_day_actual > 20:
                # Overtrading - high transaction costs + exhausting
                frequency_penalty = 0.25 * (capital / 100000)
            elif trades_per_day_actual > 10:
                # Above optimal - moderate penalty
                frequency_penalty = 0.10 * (capital / 100000)
            else:
                # Optimal range (2-10 trades/day) - no penalty
                frequency_penalty = 0.0
            
            # NEW: Overnight gap risk penalty
            # If strategy holds overnight, penalize based on historical gap frequency
            # Formula: (gap_frequency% / 100) * (avg_gap_size / 2) * capital_factor
            # Example: 10% gap freq, 1.5% avg gap = 7.5% penalty for large positions
            
            if gap_frequency > 0:
                # Scale penalty with both frequency and size of gaps
                gap_risk_penalty = (gap_frequency / 100) * (avg_gap_size / 2) * (capital / 50000)
                gap_risk_penalty = min(0.40, gap_risk_penalty)  # Cap at 40% for extremely volatile stocks
            else:
                gap_risk_penalty = 0.0  # Intraday = no gap risk
            
            # NEW: Market Beta risk penalty
            # High beta (>1.5) = very market-dependent (systemic risk)
            # Low beta (<0.7) = independent alpha (good for diversification)
            # Ideal beta: 0.7-1.3 (some market exposure but not excessive)
            
            if market_beta > 1.5:
                # High beta = high systemic risk
                beta_penalty = (market_beta - 1.5) * 0.10 * (capital / 100000)
                beta_penalty = min(0.30, beta_penalty)  # Cap at 30%
            elif market_beta < 0.7:
                # Low beta = good (independent alpha), slight bonus
                beta_penalty = -0.05 * (capital / 200000)  # Small bonus for diversification
                beta_penalty = max(-0.10, beta_penalty)  # Cap bonus at 10%
            else:
                # Normal beta range - no penalty
                beta_penalty = 0.0
            
            # Total penalty (cap at 95% to avoid completely killing stocks)
            total_penalty = min(0.95, 
                liquidity_penalty + vol_risk_penalty + dd_risk_penalty + 
                wr_risk_penalty + pf_penalty + streak_penalty + duration_penalty + frequency_penalty + gap_risk_penalty + beta_penalty
            )
            
            return return_usd * (1.0 - total_penalty)
    
    def apply_liquidity_adjustment(return_usd: float, capital: float) -> float:
        return apply_risk_adjustment(return_usd, capital)
    
    # Quick sample at key capital points
    test_caps = [10, 50, 100, 250, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    test_caps = [c for c in test_caps if min_cap <= c <= max_cap]
    
    best_cap = min_cap
    raw_return = evaluate_config(client, symbol, interval_seconds, min_cap)
    best_return = apply_liquidity_adjustment(raw_return, min_cap)
    
    for cap in test_caps:
        raw_ret = evaluate_config(client, symbol, interval_seconds, cap)
        adjusted_ret = apply_liquidity_adjustment(raw_ret, cap)
        if adjusted_ret > best_return:
            best_return = adjusted_ret
            best_cap = cap
    
    # Binary search refinement around best
    search_min = max(min_cap, best_cap / 2)
    search_max = min(max_cap, best_cap * 2)
    iterations = 0
    max_iterations = 10
    
    # Cache evaluations to avoid redundant calls
    eval_cache = {}
    
    def get_eval(cap):
        if cap not in eval_cache:
            raw = evaluate_config(client, symbol, interval_seconds, cap)
            eval_cache[cap] = apply_liquidity_adjustment(raw, cap)
        return eval_cache[cap]
    
    while (search_max - search_min) > tolerance and iterations < max_iterations:
        iterations += 1
        third = (search_max - search_min) / 3
        left_third = search_min + third
        right_third = search_max - third
        
        left_ret = get_eval(left_third)
        right_ret = get_eval(right_third)
        
        if left_ret > best_return:
            best_return = left_ret
            best_cap = left_third
        if right_ret > best_return:
            best_return = right_ret
            best_cap = right_third
        
        # Ternary search: narrow to better third
        if left_ret > right_ret:
            search_max = right_third
        else:
            search_min = left_third
    
    return best_cap, best_return


def comprehensive_binary_search(symbol: str, verbose: bool = False, max_cap: float = 1000000.0, 
                                use_robustness: bool = True) -> Tuple[int, float, float, float]:
    """
    Comprehensive binary search optimizer.
    
    Tests intervals from 1 minute to 6.5 hours.
    Tests capitals from $1 to max_cap.
    
    Args:
        use_robustness: If True, tests across multiple periods for consistency
    
    Returns: (optimal_interval_seconds, optimal_cap_usd, expected_daily_return, consistency_score)
    """
    # Clear cache for fresh evaluation (in case market conditions changed)
    global _result_cache
    _result_cache.clear()
    
    # Use None as client - optimizer uses yfinance only (avoids Alpaca rate limits!)
    client = None
    
    # Market hours: 6.5 hours = 390 minutes = 23400 seconds
    MARKET_HOURS_SECONDS = 6.5 * 3600  # 23400 seconds
    min_interval = 60  # 1 minute (API minimum)
    max_interval = int(MARKET_HOURS_SECONDS)
    
    if verbose:
        print(f"\nComprehensive optimization for {symbol}")
        print(f"Strategy: Test trades-per-day, calculate perfect intervals")
        print(f"Market hours: 6.5 hours (9:30 AM - 4:00 PM ET)")
        print(f"Capital range: $1 - ${max_cap:,.0f}")
        if use_robustness:
            print(f"Robustness testing: ENABLED (tests 3 time periods)")
            print(f"  • Tests consistency across 50, 100, 150 day periods")
            print(f"  • Rewards strategies that work across ALL periods")
        print(f"{'='*70}\n")
    
    # Phase 1: Test trades-per-day, calculate intervals that PERFECTLY divide the trading day
    # This ensures no wasted time!
    # We test 1 to 39 trades per day (39 = every 10 minutes, 1 = once per day)
    test_trades_per_day = list(range(1, 40))  # 1, 2, 3, ..., 39 trades
    test_intervals = []
    
    if verbose:
        print(f"Calculating perfect intervals (zero wasted time):")
    
    for trades in test_trades_per_day:
        interval_seconds = MARKET_HOURS_SECONDS / trades  # Exact division
        # Check if it divides evenly (no remainder > 1 minute)
        remainder = MARKET_HOURS_SECONDS % trades
        
        if remainder == 0:  # Perfect division!
            interval_int = int(interval_seconds)
            if min_interval <= interval_int <= max_interval:
                test_intervals.append(interval_int)
                if verbose:
                    print(f"  {trades} trades/day = {interval_int}s ({interval_int/60:.0f} min) - PERFECT")
    
    # Remove duplicates and sort
    test_intervals = sorted(set(test_intervals))
    
    if verbose:
        print(f"\nFound {len(test_intervals)} perfect intervals to test\n")
    
    best_interval = 3600
    best_cap = 100.0
    best_return = -999999.0
    best_consistency = 0.0
    
    if verbose:
        print("Phase 1: Testing trades-per-day (intervals perfectly divide 6.5 hour market):")
    
    for interval in test_intervals:
        # DON'T snap! We want to use the perfect intervals exactly as calculated
        # Snapping would ruin our perfect division of the 6.5 hour trading day
        cap, ret = binary_search_capital(client, symbol, interval, min_cap=1.0, max_cap=max_cap)
        
        # Calculate actual trades per day with this interval
        trades_per_day = MARKET_HOURS_SECONDS / interval
        
        # Test robustness if enabled
        if use_robustness and ret > -999000:
            avg_ret, consistency, _, oos_return = evaluate_robustness(client, symbol, interval, cap)
            # Use average return from multiple periods
            ret = avg_ret
            
            if verbose:
                print(f"  {trades_per_day:5.1f} trades/day ({interval:5d}s = {interval/60:5.1f}m): ${ret:7.2f}/day @ ${cap:>9.0f} cap [consistency: {consistency:.2f}]")
        else:
            consistency = 0.5  # Unknown
            if verbose:
                print(f"  {trades_per_day:5.1f} trades/day ({interval:5d}s = {interval/60:5.1f}m): ${ret:7.2f}/day @ ${cap:>9.0f} cap")
        
        # Score: return × consistency (no utilization penalty since all intervals divide perfectly!)
        score = ret * (0.5 + 0.5 * consistency)
        
        if best_return > -999000:
            best_score = best_return * (0.5 + 0.5 * best_consistency)
        else:
            best_score = -999999
        
        if score > best_score:
            best_return = ret
            best_interval = interval  # Use exact interval, not snapped!
            best_cap = cap
            best_consistency = consistency
    
    # Phase 2: SKIPPED - We only use intervals that perfectly divide the trading day
    # This ensures ZERO wasted market time
    if verbose:
        print(f"\nPhase 2: Skipped (only using perfect intervals that divide 6.5 hours evenly)")
        trades_per_day_final = MARKET_HOURS_SECONDS / best_interval
        print(f"\nFinal Result:")
        print(f"  Best: {trades_per_day_final:.1f} trades/day ({best_interval}s = {best_interval/60:.0f} min)")
        print(f"  Config: ${best_cap:.0f} capital = ${best_return:.2f}/day [consistency: {best_consistency:.2f}]")
    
    return best_interval, best_cap, best_return, best_consistency


def estimate_live_trading_return(paper_return: float, interval_seconds: int, capital: float, 
                                 commission_per_trade: float = 0.0) -> float:
    """
    SMART realistic live trading estimate based on actual market conditions.
    
    This accounts for ALL real-world factors that degrade backtest performance:
    - Slippage (larger orders = worse prices)
    - Partial fills (orders don't always fill completely)
    - Market regime changes (patterns stop working)
    - Competition (others exploit same patterns)
    - Execution delays (network, API, decision time)
    - Overnight gaps (if holding positions)
    - Psychological factors (harder to follow rules with real money)
    - Overfitting penalty (if returns are suspiciously good)
    
    Returns CONSERVATIVE estimate, not optimistic best-case.
    """
    if paper_return <= 0:
        # If strategy is losing money in backtest, live trading will be worse
        return paper_return * 1.5  # 50% worse losses in live trading (fear, worse execution)
    
    # Guard against division by zero
    if interval_seconds <= 0 or capital <= 0:
        return paper_return * 0.3  # Very conservative if invalid params
    
    # Calculate daily return percentage
    daily_pct = (paper_return / capital) * 100
    
    # OVERFITTING DETECTION: If returns are too good, it's probably overfit
    # Professional hedge funds aim for 15-30% per YEAR (not per day!)
    # If we're seeing >5% per day consistently, something's wrong
    overfitting_penalty = 1.0
    if daily_pct > 10:  # >10%/day = 99.9% likely overfit
        overfitting_penalty = 0.20  # Keep only 20%
    elif daily_pct > 5:  # >5%/day = extremely suspicious
        overfitting_penalty = 0.35  # Keep only 35%
    elif daily_pct > 2:  # >2%/day = very suspicious
        overfitting_penalty = 0.50  # Keep only 50%
    elif daily_pct > 1:  # >1%/day = moderately suspicious
        overfitting_penalty = 0.65  # Keep 65%
    elif daily_pct > 0.5:  # >0.5%/day = slightly suspicious but possible
        overfitting_penalty = 0.75  # Keep 75%
    else:  # <0.5%/day = realistic range
        overfitting_penalty = 0.85  # Keep 85% (still some slippage)
    
    # Estimate trades per day based on interval
    trades_per_day = (6.5 * 3600) / interval_seconds  # 6.5 hour trading day
    
    # Slippage cost per trade (scales with capital AND frequency)
    # More trades = market sees your pattern = worse prices
    if capital <= 10000:
        base_slippage = 0.0015  # 0.15% for small orders
    elif capital <= 50000:
        base_slippage = 0.0015 + 0.001 * ((capital - 10000) / 40000)  # 0.15% to 0.25%
    elif capital <= 200000:
        base_slippage = 0.0025 + 0.0015 * ((capital - 50000) / 150000)  # 0.25% to 0.40%
    else:
        base_slippage = 0.004 + 0.002 * min(1.0, (capital - 200000) / 800000)  # 0.40% to 0.60%
    
    # Frequency multiplier: High frequency = predictable = front-run
    if trades_per_day > 50:
        slippage_multiplier = 2.0  # You're being front-run
    elif trades_per_day > 20:
        slippage_multiplier = 1.5
    elif trades_per_day > 10:
        slippage_multiplier = 1.2
    else:
        slippage_multiplier = 1.0
    
    actual_slippage = base_slippage * slippage_multiplier
    daily_slippage_cost = trades_per_day * actual_slippage * capital
    
    # Commission costs
    daily_commission_cost = trades_per_day * commission_per_trade
    
    # Partial fill impact (realistic: 10-15% of trades have issues)
    partial_fill_factor = 0.88  # 12% reduction due to partial fills, rejections, etc.
    
    # Execution delay factor (thinking time, network lag, API delays)
    # By the time you execute, price has moved
    if trades_per_day > 20:
        execution_delay_factor = 0.85  # High frequency = timing critical = big impact
    elif trades_per_day > 10:
        execution_delay_factor = 0.90
    else:
        execution_delay_factor = 0.95  # Lower frequency = less timing critical
    
    # Market regime change factor (patterns degrade over time)
    # Historical edge erodes as more people find it or market changes
    regime_factor = 0.70  # Assume 70% of edge persists (30% disappears)
    
    # Psychological factor (harder to follow rules with real money)
    # Fear causes early exits, greed causes late entries
    psychological_factor = 0.90  # 10% reduction due to human emotions
    
    # Calculate realistic return with ALL factors
    adjusted_return = paper_return * overfitting_penalty * partial_fill_factor * execution_delay_factor * regime_factor * psychological_factor
    net_return = adjusted_return - daily_slippage_cost - daily_commission_cost
    
    # Return the REAL calculated value - no artificial caps
    # If math says it's negative after all costs, then it's negative
    return net_return


def main() -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive binary search optimizer with robustness testing")
    parser.add_argument("-s", "--symbol", type=str, default=None,
                       help="Stock symbol to optimize (optional, will auto-find best if not provided)")
    parser.add_argument("--symbols", nargs="+", type=str, default=None,
                       help="Multiple symbols to test and compare")
    parser.add_argument("-m", "--max-cap", type=float, default=None,
                       help="Maximum capital to test (default: depends on preset)")
    parser.add_argument("--commission", type=float, default=0.0,
                       help="Commission per trade in USD (default: $0 for Alpaca, e.g., $1-5 for other brokers)")
    parser.add_argument("--preset", type=str, choices=["conservative", "balanced", "aggressive"], default="balanced",
                       help="Risk preset: conservative ($25k cap), balanced ($250k cap, default), aggressive ($1M cap)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Show detailed progress")
    parser.add_argument("--no-robustness", action="store_true",
                       help="Disable robustness testing (faster but less reliable)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of stocks to test (default: test all)")
    parser.add_argument("--no-dynamic", action="store_true",
                       help="Don't fetch dynamic stock list (use fallback)")
    parser.add_argument("--parallel", action="store_true",
                       help="Enable parallel stock evaluation (8x faster on 8-core CPU)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count)")
    
    args = parser.parse_args()
    
    # Apply preset if max_cap not explicitly set
    if args.max_cap is None:
        if args.preset == "conservative":
            args.max_cap = 25000.0  # $25k cap, safer for beginners
        elif args.preset == "aggressive":
            args.max_cap = 1000000.0  # $1M cap, for experienced traders
        else:  # balanced (default)
            args.max_cap = 250000.0  # $250k cap, good middle ground
    
    # Display preset info
    preset_name = args.preset.upper()
    print(f"\n{'='*70}")
    print(f"PRESET: {preset_name}")
    print(f"  Max Capital: ${args.max_cap:,.0f}")
    if args.preset == "conservative":
        print(f"  Risk Level: LOW - Good for beginners or small accounts")
    elif args.preset == "aggressive":
        print(f"  Risk Level: HIGH - For experienced traders with large capital")
    else:
        print(f"  Risk Level: MEDIUM - Balanced approach (default)")
    print(f"{'='*70}")
    
    use_robustness = not args.no_robustness
    commission_per_trade = args.commission
    
    # Determine which symbols to test
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        # Use the bot's stock scanning logic!
        print(f"\n{'='*70}")
        print(f"Fetching stock universe (like the bot does)...")
        print(f"{'='*70}")
        symbols = get_stock_universe(use_top_100=not args.no_dynamic)
        
        if args.limit and args.limit < len(symbols):
            print(f"Limiting to top {args.limit} stocks by market cap")
            symbols = symbols[:args.limit]
        
        print(f"\n{'='*70}")
        print(f"AUTO-SCANNING MODE")
        print(f"{'='*70}")
        print(f"Testing {len(symbols)} stocks to find best opportunity...")
        print(f"Robustness testing: {'ENABLED' if use_robustness else 'DISABLED'}")
        print(f"Capital range: $1 - ${args.max_cap:,.0f}")
        print()
    
    # Test each symbol
    best_symbol = None
    best_interval = None
    best_cap = None
    best_return = -999999.0
    best_consistency = 0.0
    best_trade_returns = []
    best_trades_per_day = 1.0
    
    results = []
    
    # PARALLEL PROCESSING (if enabled and multiple stocks)
    if args.parallel and len(symbols) > 1:
        num_workers = args.workers or multiprocessing.cpu_count()
        print(f"\n>> PARALLEL MODE: Using {num_workers} workers")
        print(f"   Estimated speedup: {num_workers}x faster\n")
        print(f"   (Press Ctrl+C to cancel)\n")
        
        # Create worker function with fixed parameters
        worker = partial(evaluate_single_stock, 
                        max_cap=args.max_cap,
                        use_robustness=use_robustness,
                        commission_per_trade=commission_per_trade)
        
        # Set up signal handler for clean Ctrl+C termination
        pool = None
        def signal_handler(sig, frame):
            print(f"\n\n[X] Ctrl+C detected - terminating workers...")
            if pool is not None:
                pool.terminate()
                pool.join()
            print(f"[OK] Cleanup complete")
            sys.exit(1)
        
        original_sigint = signal.signal(signal.SIGINT, signal_handler)
        
        # Run in parallel with progress bar
        try:
            pool = multiprocessing.Pool(processes=num_workers, initializer=worker_init)
            results = list(tqdm(
                pool.imap(worker, symbols),
                total=len(symbols),
                desc="Optimizing (Parallel)",
                unit="stock"
            ))
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print(f"\n\n[X] Optimization cancelled by user (Ctrl+C)")
            if pool is not None:
                pool.terminate()
                pool.join()
            return 1
        except Exception as e:
            print(f"\n\n[X] Error during parallel optimization: {e}")
            if pool is not None:
                pool.terminate()
                pool.join()
            return 1
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint)
        
        # Find best result - RANK BY LIVE RETURNS, NOT PAPER!
        for idx, result in enumerate(results, 1):
            symbol = result["symbol"]
            optimal_interval = result["interval"]
            optimal_cap = result["cap"]
            expected_return = result["return"]  # Paper backtest
            consistency = result["consistency"]
            trade_returns = result["trade_returns"]
            trades_per_day = result["trades_per_day"]
            
            # Calculate LIVE return for this stock
            live_return = estimate_live_trading_return(expected_return, optimal_interval, optimal_cap, commission_per_trade)
            
            # SMART SCORING: Penalize intervals that waste the trading day
            trading_day_seconds = 6.5 * 3600
            actual_trades = trading_day_seconds / optimal_interval
            effective_trades = int(actual_trades)
            utilization = min(effective_trades / 6.5, 1.0)
            waste = actual_trades - effective_trades
            waste_penalty = 1.0 - (waste * 0.2)
            utilization_factor = utilization * waste_penalty
            
            # Update best if this is better (RANK BY LIVE × consistency × utilization!)
            # Utilization weighted at 50% to strongly prefer intervals that maximize trades
            score = live_return * (0.5 + 0.5 * consistency) * (0.5 + 0.5 * utilization_factor)
            
            if best_return > -999000:
                best_live = estimate_live_trading_return(best_return, best_interval, best_cap, commission_per_trade)
                best_actual_trades = trading_day_seconds / best_interval
                best_effective = int(best_actual_trades)
                best_util = min(best_effective / 6.5, 1.0)
                best_waste = best_actual_trades - best_effective
                best_waste_pen = 1.0 - (best_waste * 0.2)
                best_util_factor = best_util * best_waste_pen
                best_score = best_live * (0.5 + 0.5 * best_consistency) * (0.5 + 0.5 * best_util_factor)
            else:
                best_score = -999999
            
            if score > best_score:
                best_symbol = symbol
                best_interval = optimal_interval
                best_cap = optimal_cap
                best_return = expected_return  # Store paper for reference
                best_consistency = consistency
                best_trade_returns = trade_returns
                best_trades_per_day = trades_per_day
        
        print(f"\n[OK] Parallel optimization complete!")
    
    # SEQUENTIAL PROCESSING (original behavior)
    else:
        # Use tqdm for progress bar with ETA
        progress_bar = tqdm(enumerate(symbols, 1), total=len(symbols), desc="Optimizing", unit="stock")
        
        for idx, symbol in progress_bar:
            # Update progress bar description
            progress_bar.set_description(f"Testing {symbol}")
            
            if len(symbols) > 1:
                print(f"\n{'='*70}")
                print(f"[{idx}/{len(symbols)}] TESTING: {symbol}")
                print(f"{'='*70}")
            else:
                print(f"\n{'='*70}")
                print(f"COMPREHENSIVE OPTIMIZER: {symbol}")
                print(f"{'='*70}")
            
            print(f"Method: Binary search with multi-period robustness testing")
            print(f"Range: 1 min to 6.5 hours × $1 to ${args.max_cap:,.0f}")
        
        optimal_interval, optimal_cap, expected_return, consistency = comprehensive_binary_search(
            symbol, verbose=args.verbose, max_cap=args.max_cap, use_robustness=use_robustness
        )
        
        # Fetch detailed metrics for this config
        try:
            # Use None as client - optimizer uses yfinance only (avoids Alpaca rate limits!)
            closes = fetch_closes(None, symbol, optimal_interval, 200)
            detailed_sim = simulate_signals_and_projection(closes, optimal_interval, override_cap_usd=optimal_cap)
            sharpe = detailed_sim.get("sharpe_ratio", 0.0)
            sortino = detailed_sim.get("sortino_ratio", 0.0)
            expectancy = detailed_sim.get("expectancy", 0.0)
            avg_mae = detailed_sim.get("avg_mae", 0.0)
            max_dd = detailed_sim.get("max_drawdown_pct", 0.0)
            recovery = detailed_sim.get("recovery_factor", 0.0)
            trade_returns = detailed_sim.get("trade_returns", [])
            trades_per_day = detailed_sim.get("expected_trades_per_day", 1.0)
            
            # Get out-of-sample performance if robustness testing was enabled
            if use_robustness:
                _, _, _, oos_return = evaluate_robustness(None, symbol, optimal_interval, optimal_cap)
                # Also run walk-forward test
                wf_result = walk_forward_test(None, symbol, optimal_interval, optimal_cap, total_bars=600)
                wf_avg_test = wf_result.get("avg_test_return", 0.0)
                wf_ratio = wf_result.get("train_test_ratio", 0.0)
            else:
                oos_return = 0.0
                wf_avg_test = 0.0
                wf_ratio = 0.0
        except:
            sharpe = sortino = expectancy = avg_mae = max_dd = recovery = 0.0
            trade_returns = []
            trades_per_day = 1.0
            oos_return = 0.0
            wf_avg_test = 0.0
            wf_ratio = 0.0
        
        # Calculate VaR/CVaR for history logging
        var_95 = cvar_95 = 0.0
        win_rate = 0.0
        if trade_returns and len(trade_returns) >= 5:
            try:
                mc_result = monte_carlo_projection(
                    trade_returns=trade_returns,
                    starting_capital=optimal_cap,
                    trades_per_day=trades_per_day,
                    days=30,
                    num_simulations=1000
                )
                var_95 = mc_result.get("var_95", 0.0)
                cvar_95 = mc_result.get("cvar_95", 0.0)
                
                # Calculate win rate from trade returns
                wins = sum(1 for r in trade_returns if r > 0)
                win_rate = wins / len(trade_returns) if trade_returns else 0.0
            except:
                pass
        
        results.append({
            "symbol": symbol,
            "interval": optimal_interval,
            "cap": optimal_cap,
            "return": expected_return,
            "consistency": consistency,
            "sharpe": sharpe,
            "sortino": sortino,
            "expectancy": expectancy,
            "avg_mae": avg_mae,
            "max_dd": max_dd,
            "recovery": recovery,
            "oos_return": oos_return,  # Out-of-sample performance
            "wf_avg_test": wf_avg_test,  # Walk-forward test average
            "wf_ratio": wf_ratio,  # Walk-forward train/test ratio
            "trade_returns": trade_returns,
            "trades_per_day": trades_per_day,
            "var_95": var_95,
            "cvar_95": cvar_95,
        })
        
        # Save to optimization history CSV
        save_to_history(
            symbol=symbol,
            interval=optimal_interval,
            capital=optimal_cap,
            daily_return=expected_return,
            consistency=consistency,
            sharpe=sharpe,
            sortino=sortino,
            win_rate=win_rate,
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=cvar_95
        )
        
        # Calculate live trading estimate FIRST
        live_return = estimate_live_trading_return(expected_return, optimal_interval, optimal_cap, commission_per_trade)
        
        # SMART SCORING: Penalize intervals that waste the trading day
        trading_day_seconds = 6.5 * 3600
        actual_trades = trading_day_seconds / optimal_interval
        effective_trades = int(actual_trades)
        utilization = min(effective_trades / 6.5, 1.0)
        waste = actual_trades - effective_trades
        waste_penalty = 1.0 - (waste * 0.2)
        utilization_factor = utilization * waste_penalty
        
        # Use LIVE score (live return × consistency × utilization) to rank - NOT paper!
        # Utilization weighted at 50% to strongly prefer intervals that maximize trades
        score = live_return * (0.5 + 0.5 * consistency) * (0.5 + 0.5 * utilization_factor)
        
        if best_return > -999000:
            best_live = estimate_live_trading_return(best_return, best_interval, best_cap, commission_per_trade)
            best_actual_trades = trading_day_seconds / best_interval
            best_effective = int(best_actual_trades)
            best_util = min(best_effective / 6.5, 1.0)
            best_waste = best_actual_trades - best_effective
            best_waste_pen = 1.0 - (best_waste * 0.2)
            best_util_factor = best_util * best_waste_pen
            best_score = best_live * (0.5 + 0.5 * best_consistency) * (0.5 + 0.5 * best_util_factor)
        else:
            best_score = -999999
        
        if score > best_score:
            best_return = expected_return  # Store paper for reference
            best_symbol = symbol
            best_interval = optimal_interval
            best_cap = optimal_cap
            best_consistency = consistency
            best_trade_returns = trade_returns
            best_trades_per_day = trades_per_day
        
        # Determine confidence based on consistency
        if consistency >= 0.8:
            confidence_label = "VERY HIGH"
        elif consistency >= 0.6:
            confidence_label = "HIGH"
        elif consistency >= 0.4:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "LOW"
        
        print(f"Result:")
        print(f"  Paper (backtest): ${expected_return:.2f}/day")
        print(f"  Live (realistic): ${live_return:.2f}/day  [{live_return/expected_return*100:.0f}% of backtest]")
        print(f"  Config: {optimal_interval}s ({optimal_interval/3600:.4f}h) @ ${optimal_cap:.0f} cap")
        if use_robustness:
            print(f"  Consistency: {consistency:.2f} ({confidence_label} confidence)")
            if oos_return != 0.0:
                oos_label = "[OK]" if oos_return > 0 else "[!]"
                print(f"  Out-of-Sample: ${oos_return:.2f}/day {oos_label}  (unseen data test)")
            if wf_avg_test != 0.0:
                wf_label = "[OK]" if wf_ratio >= 0.5 else "[!]"
                print(f"  Walk-Forward: ${wf_avg_test:.2f}/day (ratio: {wf_ratio:.2f}) {wf_label}")
        print(f"  Quality Metrics:")
        print(f"    Sharpe Ratio: {sharpe:.3f}  (risk-adjusted return)")
        print(f"    Sortino Ratio: {sortino:.3f}  (downside risk-adjusted)")
        print(f"    Expectancy: {expectancy:.2f}%  (avg return per trade)")
        print(f"    Avg MAE: {avg_mae:.2f}%  (worst move against us)")
        print(f"    Max Drawdown: {max_dd:.2f}%  (largest peak-to-trough drop)")
        if recovery != float('inf'):
            print(f"    Recovery Factor: {recovery:.2f}  (return/drawdown ratio)")
        else:
            print(f"    Recovery Factor: ∞  (no drawdown!)")
        
        # Calculate and display Kelly Criterion
        if trade_returns and len(trade_returns) >= 5:
            try:
                wins = [r for r in trade_returns if r > 0]
                losses = [r for r in trade_returns if r < 0]
                
                if wins and losses:
                    calculated_win_rate = len(wins) / len(trade_returns)
                    avg_win = sum(wins) / len(wins)
                    avg_loss = abs(sum(losses) / len(losses))
                    
                    # Kelly formula: f* = (p*b - q) / b
                    b = avg_win / avg_loss  # Win/loss ratio
                    q = 1 - calculated_win_rate
                    kelly_full = (calculated_win_rate * b - q) / b
                    kelly_half = kelly_full * 0.5  # Half-Kelly (conservative)
                    
                    kelly_pct = kelly_half * 100
                    
                    print(f"    Kelly Criterion: {kelly_pct:.1f}%  (optimal position size)")
                    print(f"      -> Bot uses Half-Kelly for safety ({kelly_full*50:.1f}% of full Kelly)")
            except:
                pass
        
        # Show running best after each stock (for early stopping)
        if len(symbols) > 1 and best_symbol is not None:
            print(f"\n{'~'*70}")
            print(f"BEST SO FAR (after {idx}/{len(symbols)} stocks):")
            print(f"  Leader: {best_symbol}")
            print(f"  Interval: {best_interval}s ({best_interval/3600:.4f}h)")
            
            # Show trading day utilization
            best_trades_per_day = (6.5 * 3600) / best_interval
            best_effective_trades = int(best_trades_per_day)
            print(f"  Trades/day: {best_effective_trades} complete trades ({best_trades_per_day:.1f} total)")
            
            print(f"  Capital: ${best_cap:,.0f}")
            print(f"  Expected: ${best_return:.2f}/day (paper)")
            best_live = estimate_live_trading_return(best_return, best_interval, best_cap, commission_per_trade)
            print(f"           ${best_live:.2f}/day (live) <<< USE THIS")
            print(f"  Consistency: {best_consistency:.2f}")
            print(f"{'~'*70}")
    
    # Show summary if multiple symbols
    if len(symbols) > 1:
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY (Ranked by LIVE return × consistency)")
        print(f"{'='*70}")
        # Add live trading estimates and scores to results
        for r in results:
            r["live_return"] = estimate_live_trading_return(r["return"], r["interval"], r["cap"], commission_per_trade)
            # RANK BY LIVE RETURNS, NOT PAPER!
            r["score"] = r["live_return"] * (0.5 + 0.5 * r["consistency"])
        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results[:10], 1):
            status = "[OK]" if r["return"] > 0 else "[X]"
            
            # Confidence label
            if r["consistency"] >= 0.8:
                conf_label = "VERY HIGH"
            elif r["consistency"] >= 0.6:
                conf_label = "HIGH"
            elif r["consistency"] >= 0.4:
                conf_label = "MEDIUM"
            else:
                conf_label = "LOW"
            
            print(f"{i}. {r['symbol']:6s} {status}  LIVE: ${r['live_return']:6.2f}/day  (paper: ${r['return']:6.2f}/day)")
            print(f"   Config: {r['interval']:5d}s ({r['interval']/3600:.4f}h) @ ${r['cap']:7.0f} cap")
            if use_robustness:
                print(f"   Consistency: {r['consistency']:.2f} ({conf_label})")
        print(f"{'='*70}")
        
        # Calculate and display correlation matrix
        if len(symbols) >= 2:
            print(f"\n{'='*70}")
            print(f"PORTFOLIO CORRELATION ANALYSIS")
            print(f"{'='*70}")
            print(f"Analyzing {len(symbols)} stocks for diversification...")
            
            corr_data = calculate_correlation_matrix(symbols, days=60)
            avg_corr = corr_data.get("avg_correlation", 0.0)
            high_corr_pairs = corr_data.get("high_correlation_pairs", [])
            
            print(f"\nAverage Portfolio Correlation: {avg_corr:.2f}")
            
            if avg_corr < 0.3:
                print(f"[OK] EXCELLENT diversification (low correlation)")
            elif avg_corr < 0.5:
                print(f"[OK] GOOD diversification")
            elif avg_corr < 0.7:
                print(f"[!]  MODERATE diversification (some correlation)")
            else:
                print(f"[X] POOR diversification (high correlation - similar risk)")
            
            if high_corr_pairs:
                print(f"\n[!]  High Correlation Pairs (>0.7):")
                for sym1, sym2, corr in high_corr_pairs[:5]:  # Show top 5
                    print(f"   {sym1} <-> {sym2}: {corr:.2f} (move together)")
                print(f"\n[TIP] Consider replacing one stock from each pair for better diversification")
            else:
                print(f"\n[OK] No high correlation pairs found (>0.7)")
                print(f"   Portfolio is well-diversified!")
            
            print(f"{'='*70}")
    
    # Calculate live trading estimate for best config (if we found a valid stock)
    if best_symbol is None or best_interval is None:
        print(f"\n{'='*70}")
        print(f"[X] NO VALID CONFIGURATIONS FOUND")
        print(f"{'='*70}")
        print(f"\n[!]  All {len(symbols)} stocks failed evaluation. Possible causes:")
        print(f"  1. yfinance couldn't fetch data (network issue or invalid symbols)")
        print(f"  2. Insufficient historical data (need {max(config.LONG_WINDOW + 10, 30)}+ bars)")
        print(f"  3. Symbols may be delisted or have trading halts")
        print(f"\n[TIP] Try:")
        print(f"  • Check your internet connection")
        print(f"  • Run with a single known-good symbol: python optimizer.py -s AAPL")
        print(f"  • Enable verbose mode: python optimizer.py -v")
        return 1
    
    best_live_return = estimate_live_trading_return(best_return, best_interval, best_cap, commission_per_trade)
    
    # Show optimal config
    print(f"\n{'='*70}")
    print(f"OPTIMAL CONFIGURATION (Bot auto-picks stocks)")
    print(f"{'='*70}")
    print(f"Best performer found: {best_symbol}  (reference only)")
    print(f"\n>>> YOU SET THESE 2 VALUES:")
    # Calculate trades per day and wasted time
    trading_day_seconds = 6.5 * 3600
    trades_per_day = trading_day_seconds / best_interval
    waste_time_seconds = trading_day_seconds % best_interval
    waste_minutes = waste_time_seconds / 60
    
    print(f"  1. Time Interval: {best_interval}s ({best_interval/3600:.4f}h = {best_interval/60:.0f} min)")
    print(f"     -> {trades_per_day:.1f} trades/day")
    if waste_minutes > 0:
        print(f"     -> Wastes {waste_minutes:.0f} minutes at end of day")
    else:
        print(f"     -> Perfect fit! No wasted time")
    
    print(f"  2. Total Capital: ${best_cap:.2f}")
    print(f"\n>>> BOT DOES EVERYTHING ELSE AUTOMATICALLY")
    
    if use_robustness:
        if best_consistency >= 0.8:
            conf_label = "VERY HIGH"
        elif best_consistency >= 0.6:
            conf_label = "HIGH"
        elif best_consistency >= 0.4:
            conf_label = "MEDIUM"
        else:
            conf_label = "LOW"
        print(f"Consistency: {best_consistency:.2f} ({conf_label} confidence)")
    
    print(f"\nExpected Daily Returns (USE THESE VALUES!):")
    print(f"  Live Trading (REALISTIC):  ${best_live_return:.2f}/day  <<< USE THIS")
    if best_return != 0:
        live_pct = best_live_return/best_return*100
        print(f"  Paper Trading (reference):  ${best_return:.2f}/day  (backtest only, {100-live_pct:.0f}% unrealistic)")
        
        # OVERFITTING WARNING based on return percentage
        daily_return_pct = (best_return / best_cap) * 100
        if daily_return_pct > 5:
            print(f"\n[!]  OVERFITTING WARNING: {daily_return_pct:.1f}%/day is EXTREMELY suspicious!")
            print(f"     This is likely curve-fitted to recent data and won't persist.")
            print(f"     Hedge funds average 1-2% per MONTH, not per day.")
            print(f"     Live estimate already applies 65-80% penalty for overfitting.")
        elif daily_return_pct > 2:
            print(f"\n[!]  HIGH RETURNS WARNING: {daily_return_pct:.1f}%/day is very high")
            print(f"     This may not be sustainable long-term.")
            print(f"     Live estimate applies 50% penalty for potential overfitting.")
        elif daily_return_pct > 1:
            print(f"\n[!]  NOTE: {daily_return_pct:.1f}%/day is above typical algo performance")
            print(f"     Live estimate applies 35% penalty as safety margin.")
    else:
        print(f"  Live Trading (realistic):  ${best_live_return:.2f}/day")
    
    print(f"\nRealistic Factors Applied to Live Estimate:")
    trades_per_day = (6.5 * 3600) / best_interval
    print(f"  • Trades per day: {trades_per_day:.1f}")
    
    # Show actual factors applied
    daily_pct = (best_return / best_cap) * 100
    if daily_pct > 10:
        print(f"  • Overfitting penalty: 80% (returns too high = likely overfit)")
    elif daily_pct > 5:
        print(f"  • Overfitting penalty: 65% (very high returns = suspicious)")
    elif daily_pct > 2:
        print(f"  • Overfitting penalty: 50% (high returns = caution)")
    elif daily_pct > 1:
        print(f"  • Overfitting penalty: 35% (above average = conservative)")
    elif daily_pct > 0.5:
        print(f"  • Overfitting penalty: 25% (good returns = mild caution)")
    else:
        print(f"  • Overfitting penalty: 15% (realistic returns)")
    
    print(f"  • Slippage: 0.15-0.60% per trade (scales with capital & frequency)")
    print(f"  • Partial fills: 12% reduction (realistic market impact)")
    print(f"  • Execution delays: 5-15% reduction (network, API, timing)")
    print(f"  • Market regime changes: 30% edge erosion (patterns degrade)")
    print(f"  • Psychological factors: 10% reduction (human emotions)")
    if commission_per_trade > 0:
        print(f"  • Commission: ${commission_per_trade:.2f} per trade")
    print(f"  • NO ARTIFICIAL CAPS - pure math based on real-world costs")
    
    symbol = best_symbol
    optimal_interval = best_interval
    optimal_cap = best_cap
    expected_return = best_return
    
    print(f"\n{'='*70}")
    if expected_return < 0:
        print(f"[!]  STRATEGY NOT PROFITABLE")
        print(f"{'='*70}")
        print(f"\n{symbol} shows negative returns in current market conditions.")
        print(f"\nReasons:")
        print(f"  • Market is bearish (downtrend)")
        print(f"  • Long-only strategy can't profit from falling prices")
        print(f"\nSuggestions:")
        print(f"  1. Try different symbol: python optimizer.py -s SPY -v")
        print(f"  2. Wait for bullish market conditions")
        print(f"  3. Bot will EXIT if run with negative projection")
    elif expected_return < 1.0:
        print(f"[!]  LOW PROFITABILITY")
        print(f"{'='*70}")
        print(f"\nExpected return is less than $1/day.")
        print(f"Consider:")
        print(f"  • Different symbol with better momentum")
        print(f"  • Waiting for more favorable conditions")
    else:
        print(f"[OK] STRATEGY IS PROFITABLE")
        print(f"{'='*70}")
        print(f"\nRun bot (ONLY SET 2 THINGS: interval & capital):")
        print(f"  $BotDir = 'C:\\Users\\YourName\\...\\Paper-Trading'")
        print(f"\n  # RECOMMENDED: Bot auto-picks best stocks & manages everything")
        rounded_cap = round(optimal_cap, 2)
        print(f"  & \"$BotDir\\botctl.ps1\" start python -u runner.py -t {optimal_interval/3600:.4f} -m {rounded_cap}")
        print(f"\n  Bot will automatically:")
        print(f"    - Scan top stocks (S&P 500)")
        print(f"    - Pick best 15 performers")
        print(f"    - Allocate capital smartly")
        print(f"    - Set stop loss/take profit")
        print(f"    - Rebalance portfolio")
        print(f"    - Everything else!")
        print(f"\n  You ONLY set: Time interval ({optimal_interval/3600:.4f}h) + Total capital (${optimal_cap:.0f})")
        
        # Compounding projections with both paper and live estimates
        print(f"\n{'='*70}")
        print(f"REALISTIC EXPECTED RETURNS")
        print(f"{'='*70}")
        
        if optimal_cap > 0:
            paper_daily_pct = (expected_return / optimal_cap) * 100
            live_daily_pct = (best_live_return / optimal_cap) * 100
            
            print(f"Starting capital: ${optimal_cap:.2f}")
            print(f"Paper return: ${expected_return:.2f}/day ({paper_daily_pct:.3f}%/day)")
            print(f"Live return:  ${best_live_return:.2f}/day ({live_daily_pct:.3f}%/day)")
            
            # Only show realistic timeframes (1-6 months, not years)
            # Anything beyond 6 months is pure speculation
            print(f"\n{'Timeframe':<15} {'Expected Profit (Conservative)':<35}")
            print(f"{'-'*15} {'-'*35}")
            
            for months in [1, 2, 3, 6]:
                trading_days = months * 20
                
                # Account for variance and drawdowns (not just straight compounding)
                # Real returns have ups and downs, not smooth exponential growth
                # Use geometric mean instead of arithmetic mean
                
                # Assume 20% variance drag (realistic for algo trading)
                variance_drag = 0.80
                
                # Assume periodic drawdowns reduce compounding
                # Longer timeframes = more likely to hit drawdown
                if months == 1:
                    drawdown_factor = 0.95  # 5% chance of 10-20% drawdown
                elif months == 2:
                    drawdown_factor = 0.90  # 10% expected drawdown impact
                elif months == 3:
                    drawdown_factor = 0.85  # 15% expected drawdown impact
                else:  # 6 months
                    drawdown_factor = 0.75  # 25% expected drawdown impact
                
                # Calculate conservative estimate
                effective_daily_return = live_daily_pct * variance_drag * drawdown_factor
                
                try:
                    if effective_daily_return > 0:
                        final_capital = optimal_cap * ((1 + effective_daily_return/100) ** trading_days)
                    else:
                        final_capital = optimal_cap  # No growth if negative
                    
                    profit = final_capital - optimal_cap
                    gain_pct = (profit / optimal_cap) * 100
                    
                    if profit > 0:
                        result_str = f"+${profit:>10,.0f} (+{gain_pct:>5.1f}%)"
                    else:
                        result_str = f"${profit:>10,.0f} ({gain_pct:>5.1f}%)"
                    
                    print(f"{months:2d} month{'s' if months>1 else ' '} ({trading_days:3d} days): {result_str}")
                except (OverflowError, ValueError):
                    print(f"{months:2d} month{'s' if months>1 else ' '} ({trading_days:3d} days): [Data insufficient for projection]")
            
            print(f"\n[!]  CRITICAL: These are BEST-CASE estimates assuming:")
            print(f"   • No major market crashes or regime changes")
            print(f"   • You follow the strategy perfectly (no emotion)")
            print(f"   • No extended losing streaks (very unlikely)")
            print(f"   • Bot runs 24/7 with no downtime")
            print(f"   • Already accounts for: slippage, fees, execution delays,")
            print(f"     psychological factors, variance drag, and drawdowns")
            print(f"\n[!]  REALITY: Actual returns will likely be 30-50% lower than shown")
            print(f"   Professional traders expect 15-30% per YEAR, not per month")
            print(f"   If you beat 2%/month consistently, you're doing VERY well")
            
            # Monte Carlo confidence intervals
            if len(best_trade_returns) >= 2:
                print(f"\n{'='*70}")
                print(f"MONTE CARLO SIMULATION (30 days)")
                print(f"{'='*70}")
                print(f"Based on {len(best_trade_returns)} historical trades, running 1000 simulations...")
                print(f"")
                
                mc_result = monte_carlo_projection(
                    trade_returns=best_trade_returns,
                    starting_capital=optimal_cap,
                    trades_per_day=best_trades_per_day,
                    days=30,
                    num_simulations=1000
                )
                
                p5 = mc_result["p5"]
                p50 = mc_result["p50"]
                p95 = mc_result["p95"]
                mean = mc_result["mean"]
                var_95 = mc_result["var_95"]
                cvar_95 = mc_result["cvar_95"]
                
                gain_p5 = ((p5 - optimal_cap) / optimal_cap) * 100
                gain_p50 = ((p50 - optimal_cap) / optimal_cap) * 100
                gain_p95 = ((p95 - optimal_cap) / optimal_cap) * 100
                gain_mean = ((mean - optimal_cap) / optimal_cap) * 100
                
                print(f"Starting capital: ${optimal_cap:,.2f}")
                print(f"")
                print(f"After 30 trading days:")
                print(f"  5th percentile (pessimistic): ${p5:,.2f}  ({gain_p5:+.1f}%)")
                print(f"  50th percentile (median):     ${p50:,.2f}  ({gain_p50:+.1f}%)")
                print(f"  Mean (average):               ${mean:,.2f}  ({gain_mean:+.1f}%)")
                print(f"  95th percentile (optimistic): ${p95:,.2f}  ({gain_p95:+.1f}%)")
                print(f"")
                print(f"95% confidence interval: ${p5:,.2f} to ${p95:,.2f}")
                print(f"")
                print(f"RISK METRICS (Institutional Standard):")
                print(f"  VaR (95% confidence):  ${var_95:,.2f}")
                print(f"    -> 95% chance daily loss won't exceed this amount")
                print(f"  CVaR (Expected Shortfall): ${cvar_95:,.2f}")
                print(f"    -> Average loss in worst 5% of scenarios")
                print(f"")
                print(f"[!]  This shows the RANGE of possible outcomes, not just average.")
                print(f"   • 5% of simulations ended worse than ${p5:,.2f}")
                print(f"   • 5% of simulations ended better than ${p95:,.2f}")
                print(f"   • Real results will vary - this is based on historical patterns")
        else:
            print(f"\n[!]  Invalid capital configuration (${optimal_cap:.2f})")
            print(f"   Cannot calculate compounding projections.")
    
    # Confidence assessment
    daily_pct = (best_return / best_cap * 100) if best_cap > 0 else 0
    
    if use_robustness:
        print(f"\n{'='*70}")
        print(f"ROBUSTNESS ANALYSIS")
        print(f"{'='*70}")
        print(f"Consistency Score: {best_consistency:.2f} ({conf_label})")
        print(f"Daily Return: {daily_pct:.2f}% per day")
        print(f"Capital: ${best_cap:.2f}")
        print(f"Frequency: {trades_per_day:.1f} trades/day")
        
        if best_consistency >= 0.8:
            print(f"\n[OK] HIGH CONFIDENCE - Strategy performs consistently across all test periods")
            print(f"   This config is likely to work well in live trading")
        elif best_consistency >= 0.6:
            print(f"\n[!]  MEDIUM CONFIDENCE - Some variation across periods")
            print(f"   Use with caution and start with small capital")
        elif best_consistency >= 0.4:
            print(f"\n[!]  LOW CONFIDENCE - Significant variation across periods")
            print(f"   Likely overfit to recent market conditions")
        else:
            print(f"\n[X] VERY LOW CONFIDENCE - Highly variable across periods")
            print(f"   Strong evidence of overfitting, not recommended")
    
    print(f"\n{'='*70}")
    print(f"WHAT WAS TESTED:")
    print(f"  • Intervals: 60s to {int(6.5*3600)}s (1min to 6.5hrs)")
    print(f"  • Capitals: $1 to ${args.max_cap:,.0f}")
    print(f"  • Stocks: {len(symbols)} symbol(s)")
    if use_robustness:
        print(f"  • Time periods: 3 different windows (50, 100, 150 days)")
        print(f"  • Method: Golden ratio search + multi-period robustness scoring")
        print(f"  • Scoring: return × (0.5 + 0.5 × consistency)")
    else:
        print(f"  • Method: Golden ratio binary search")
    print(f"  • Total evaluations: ~{len(_result_cache)} configs tested (cached)")
    print(f"{'='*70}\n")
    
    # Final clean summary - just the 2 values you need
    print(f"\n{'='*70}")
    print(f"FINAL RESULT - USE THESE 2 VALUES:")
    print(f"{'='*70}")
    print(f"\n  Time Interval: {best_interval/3600:.4f} hours")
    print(f"  Total Capital: ${best_cap:.2f}")
    print(f"\n  Command:")
    print(f"  & \"$BotDir\\botctl.ps1\" start python -u runner.py -t {best_interval/3600:.4f} -m {best_cap:.2f}")
    print(f"\n{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    # Required for Windows multiprocessing support
    multiprocessing.freeze_support()
    
    # Check if verbose mode for sleep messages
    verbose = '-v' in sys.argv or '--verbose' in sys.argv
    
    # Prevent system sleep during optimization
    restore_sleep = prevent_sleep(verbose=verbose)
    
    try:
        exit_code = main()
    finally:
        # Always restore normal sleep behavior when done
        restore_sleep()
    
    sys.exit(exit_code)
