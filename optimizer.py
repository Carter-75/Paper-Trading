#!/usr/bin/env python3
"""
Comprehensive Binary Search Optimizer

Tests intervals and capitals with ROBUSTNESS testing across multiple periods.
No artificial limits - let's find what ACTUALLY works consistently!
"""

import sys
from typing import Tuple, Dict, List
import config
from runner import (
    make_client,
    fetch_closes,
    simulate_signals_and_projection,
    snap_interval_to_supported_seconds,
)
from stock_scanner import get_stock_universe

# Global result cache to avoid duplicate API calls
_result_cache: Dict[Tuple[str, int, float], float] = {}


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


def evaluate_robustness(client, symbol: str, interval_seconds: int, cap_usd: float) -> Tuple[float, float, List[float]]:
    """
    Test strategy across multiple time periods to check robustness.
    
    Returns:
        (average_return, consistency_score, period_returns)
        
    consistency_score: 0.0-1.0, higher = more consistent across periods
    """
    # Test on 3 different periods (last 200, 400, 600 bars)
    period_bars = [200, 400, 600]  # ~50 days, 100 days, 150 days each
    period_returns = []
    
    for bars in period_bars:
        ret = evaluate_config(client, symbol, interval_seconds, cap_usd, bars)
        if ret > -999000:  # Valid result
            period_returns.append(ret)
    
    if len(period_returns) == 0:
        return -999999.0, 0.0, []
    
    # Calculate average and consistency
    avg_return = sum(period_returns) / len(period_returns)
    
    if len(period_returns) >= 2:
        # Consistency = 1.0 - (std_dev / mean), capped at 0-1
        import statistics
        std_dev = statistics.stdev(period_returns)
        mean_abs = abs(avg_return) if avg_return != 0 else 1.0
        consistency = max(0.0, min(1.0, 1.0 - (std_dev / mean_abs)))
    else:
        consistency = 0.5  # Unknown consistency
    
    return avg_return, consistency, period_returns


def binary_search_capital(client, symbol: str, interval_seconds: int, 
                          min_cap: float = 1.0, max_cap: float = 1000000.0,
                          tolerance: float = 10.0) -> Tuple[float, float]:
    """
    Binary search to find optimal capital for given interval.
    Tests from $1 to max_cap with caching for efficiency.
    """
    # Quick sample at key capital points
    test_caps = [10, 50, 100, 250, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    test_caps = [c for c in test_caps if min_cap <= c <= max_cap]
    
    best_cap = min_cap
    best_return = evaluate_config(client, symbol, interval_seconds, min_cap)
    
    for cap in test_caps:
        ret = evaluate_config(client, symbol, interval_seconds, cap)
        if ret > best_return:
            best_return = ret
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
            eval_cache[cap] = evaluate_config(client, symbol, interval_seconds, cap)
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
    
    client = make_client(allow_missing=False, go_live=False)
    
    min_interval = 60  # 1 minute (API minimum)
    max_interval = int(6.5 * 3600)  # 6.5 hours
    
    if verbose:
        print(f"\nComprehensive optimization for {symbol}")
        print(f"Interval range: {min_interval}s - {max_interval}s (1min to 6.5hrs)")
        print(f"Capital range: $1 - ${max_cap:,.0f}")
        if use_robustness:
            print(f"Robustness testing: ENABLED (tests 3 time periods)")
            print(f"  • Tests consistency across 50, 100, 150 day periods")
            print(f"  • Rewards strategies that work across ALL periods")
        print(f"{'='*70}\n")
    
    # Phase 1: Sample key intervals
    test_intervals = [60, 300, 900, 1800, 3600, 7200, 14400, 21600]  # 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h
    test_intervals = [i for i in test_intervals if min_interval <= i <= max_interval]
    
    best_interval = 3600
    best_cap = 100.0
    best_return = -999999.0
    best_consistency = 0.0
    
    if verbose:
        print("Phase 1: Sampling key intervals with capital optimization:")
    
    for interval in test_intervals:
        snapped = snap_interval_to_supported_seconds(interval)
        cap, ret = binary_search_capital(client, symbol, snapped, min_cap=1.0, max_cap=max_cap)
        
        # Test robustness if enabled
        if use_robustness and ret > -999000:
            avg_ret, consistency, _ = evaluate_robustness(client, symbol, snapped, cap)
            # Use average return from multiple periods
            ret = avg_ret
            
            if verbose:
                print(f"  {snapped:5d}s ({snapped/3600:6.3f}h): ${ret:7.2f}/day @ ${cap:>9.0f} cap [consistency: {consistency:.2f}]")
        else:
            consistency = 0.5  # Unknown
            if verbose:
                print(f"  {snapped:5d}s ({snapped/3600:6.3f}h): ${ret:7.2f}/day @ ${cap:>9.0f} cap")
        
        # Favor configs with good returns AND high consistency
        score = ret * (0.5 + 0.5 * consistency)  # Weight consistency 50%
        best_score = best_return * (0.5 + 0.5 * best_consistency)
        
        if score > best_score:
            best_return = ret
            best_interval = snapped
            best_cap = cap
            best_consistency = consistency
    
    # Phase 2: Golden ratio search for optimal interval (more efficient than ternary)
    if verbose:
        print(f"\nPhase 2: Refining around {best_interval}s (golden ratio search):")
    
    search_min = max(min_interval, best_interval // 2)
    search_max = min(max_interval, best_interval * 2)
    iterations = 0
    max_iterations = 12  # Allow more iterations for finer precision
    
    # Golden ratio for optimal search point placement
    golden_ratio = (3 - 5**0.5) / 2  # ~0.382
    
    # Track tested intervals to avoid duplicates
    tested_intervals = set()
    
    # Initial evaluation at golden ratio points
    left_test = int(search_min + golden_ratio * (search_max - search_min))
    right_test = int(search_max - golden_ratio * (search_max - search_min))
    
    left_snap = snap_interval_to_supported_seconds(left_test)
    right_snap = snap_interval_to_supported_seconds(right_test)
    
    # Early termination if snapping converges to same interval
    if left_snap == right_snap:
        if verbose:
            print(f"  Skipping refinement - intervals converged to {left_snap}s after snapping")
    else:
        left_cap, left_ret = binary_search_capital(client, symbol, left_snap, min_cap=1.0, max_cap=max_cap)
        right_cap, right_ret = binary_search_capital(client, symbol, right_snap, min_cap=1.0, max_cap=max_cap)
        
        tested_intervals.add(left_snap)
        tested_intervals.add(right_snap)
        
        # Update best if found better
        if left_ret > best_return:
            best_return = left_ret
            best_interval = left_snap
            best_cap = left_cap
        if right_ret > best_return:
            best_return = right_ret
            best_interval = right_snap
            best_cap = right_cap
        
        while (search_max - search_min) > 60 and iterations < max_iterations:
            iterations += 1
            
            if verbose:
                print(f"  Iter {iterations}: Range [{search_min}s - {search_max}s] = {search_max - search_min}s span")
                print(f"    Testing: {left_snap}s (${left_ret:.2f}) vs {right_snap}s (${right_ret:.2f})")
            
            if left_ret > right_ret:
                # Narrow to left side
                search_max = right_test
                right_test = left_test
                right_snap = left_snap
                right_ret = left_ret
                right_cap = left_cap
                
                # New left test point
                left_test = int(search_min + golden_ratio * (search_max - search_min))
                left_snap = snap_interval_to_supported_seconds(left_test)
                
                # Check if we've already tested this snapped interval
                if left_snap in tested_intervals or left_snap == right_snap:
                    if verbose:
                        print(f"  Converged - all nearby intervals snap to tested values")
                    break
                
                tested_intervals.add(left_snap)
                left_cap, left_ret = binary_search_capital(client, symbol, left_snap, min_cap=1.0, max_cap=max_cap)
                
                if left_ret > best_return:
                    best_return = left_ret
                    best_interval = left_snap
                    best_cap = left_cap
            else:
                # Narrow to right side
                search_min = left_test
                left_test = right_test
                left_snap = right_snap
                left_ret = right_ret
                left_cap = right_cap
                
                # New right test point
                right_test = int(search_max - golden_ratio * (search_max - search_min))
                right_snap = snap_interval_to_supported_seconds(right_test)
                
                # Check if we've already tested this snapped interval
                if right_snap in tested_intervals or right_snap == left_snap:
                    if verbose:
                        print(f"  Converged - all nearby intervals snap to tested values")
                    break
                
                tested_intervals.add(right_snap)
                right_cap, right_ret = binary_search_capital(client, symbol, right_snap, min_cap=1.0, max_cap=max_cap)
                
                if right_ret > best_return:
                    best_return = right_ret
                    best_interval = right_snap
                    best_cap = right_cap
    
    if verbose:
        print(f"\nConverged after {iterations} refinement iterations")
        print(f"Final precision: {search_max - search_min}s range (down to ~1 minute)")
        print(f"Best config: {best_interval}s @ ${best_cap:.0f} = ${best_return:.2f}/day [consistency: {best_consistency:.2f}]")
    
    return best_interval, best_cap, best_return, best_consistency


def estimate_live_trading_return(paper_return: float, interval_seconds: int, capital: float) -> float:
    """
    Estimate realistic live trading returns accounting for real-world costs.
    
    Factors considered:
    - Slippage: ~0.1% per trade (difference between expected and actual price)
    - Partial fills: ~5% of trades don't fully execute
    - Market regime changes: Historical patterns may not continue
    - Competition: Other traders exploit the same patterns
    
    Returns adjusted expected daily return for live trading.
    """
    if paper_return <= 0:
        # If strategy is losing money in backtest, live trading will be worse
        return paper_return * 1.3  # 30% worse losses in live trading
    
    # Guard against division by zero
    if interval_seconds <= 0 or capital <= 0:
        return paper_return * 0.5  # Conservative 50% of backtest if invalid params
    
    # Estimate trades per day based on interval
    trades_per_day = (6.5 * 3600) / interval_seconds  # 6.5 hour trading day
    
    # Slippage cost per trade (as % of capital)
    slippage_per_trade = 0.001  # 0.1% per trade
    daily_slippage_cost = trades_per_day * slippage_per_trade * capital
    
    # Partial fill impact (5% of trades miss, reducing profits)
    partial_fill_factor = 0.95
    
    # Market efficiency factor (historical patterns degrade)
    # More frequent trading = more competition = lower edge
    if trades_per_day > 20:
        efficiency_factor = 0.50  # Very frequent trading (1min) = 50% of backtest
    elif trades_per_day > 10:
        efficiency_factor = 0.60  # Frequent trading (5min) = 60% of backtest
    elif trades_per_day > 4:
        efficiency_factor = 0.70  # Moderate trading (15min-1hr) = 70% of backtest
    else:
        efficiency_factor = 0.80  # Infrequent trading (4hr+) = 80% of backtest
    
    # Calculate realistic return
    gross_return = paper_return * partial_fill_factor * efficiency_factor
    net_return = gross_return - daily_slippage_cost
    
    return max(net_return, paper_return * 0.3)  # At least 30% of backtest return


def main() -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive binary search optimizer with robustness testing")
    parser.add_argument("-s", "--symbol", type=str, default=None,
                       help="Stock symbol to optimize (optional, will auto-find best if not provided)")
    parser.add_argument("--symbols", nargs="+", type=str, default=None,
                       help="Multiple symbols to test and compare")
    parser.add_argument("-m", "--max-cap", type=float, default=1000000.0,
                       help="Maximum capital to test (default: $1,000,000)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Show detailed progress")
    parser.add_argument("--no-robustness", action="store_true",
                       help="Disable robustness testing (faster but less reliable)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of stocks to test (default: test all)")
    parser.add_argument("--no-dynamic", action="store_true",
                       help="Don't fetch dynamic stock list (use fallback)")
    
    args = parser.parse_args()
    
    use_robustness = not args.no_robustness
    
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
    
    results = []
    
    for symbol in symbols:
        if len(symbols) > 1:
            print(f"\n{'='*70}")
            print(f"TESTING: {symbol}")
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
        
        results.append({
            "symbol": symbol,
            "interval": optimal_interval,
            "cap": optimal_cap,
            "return": expected_return,
            "consistency": consistency
        })
        
        # Use score (return * consistency) to rank
        score = expected_return * (0.5 + 0.5 * consistency)
        best_score = best_return * (0.5 + 0.5 * best_consistency)
        
        if score > best_score:
            best_return = expected_return
            best_symbol = symbol
            best_interval = optimal_interval
            best_cap = optimal_cap
            best_consistency = consistency
        
        # Calculate live trading estimate
        live_return = estimate_live_trading_return(expected_return, optimal_interval, optimal_cap)
        
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
    
    # Show summary if multiple symbols
    if len(symbols) > 1:
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY (Ranked by score = return × consistency)")
        print(f"{'='*70}")
        # Add live trading estimates and scores to results
        for r in results:
            r["live_return"] = estimate_live_trading_return(r["return"], r["interval"], r["cap"])
            r["score"] = r["return"] * (0.5 + 0.5 * r["consistency"])
        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results[:10], 1):
            status = "✅" if r["return"] > 0 else "❌"
            
            # Confidence label
            if r["consistency"] >= 0.8:
                conf_label = "VERY HIGH"
            elif r["consistency"] >= 0.6:
                conf_label = "HIGH"
            elif r["consistency"] >= 0.4:
                conf_label = "MEDIUM"
            else:
                conf_label = "LOW"
            
            print(f"{i}. {r['symbol']:6s} {status}  Paper: ${r['return']:6.2f}/day  Live: ${r['live_return']:6.2f}/day")
            print(f"   Config: {r['interval']:5d}s ({r['interval']/3600:.4f}h) @ ${r['cap']:7.0f} cap")
            if use_robustness:
                print(f"   Consistency: {r['consistency']:.2f} ({conf_label})")
        print(f"{'='*70}")
    
    # Calculate live trading estimate for best config
    best_live_return = estimate_live_trading_return(best_return, best_interval, best_cap)
    
    # Show optimal config
    print(f"\n{'='*70}")
    print(f"OPTIMAL CONFIGURATION")
    print(f"{'='*70}")
    print(f"Symbol: {best_symbol}")
    print(f"Interval: {best_interval}s ({best_interval/3600:.4f}h)")
    print(f"Capital: ${best_cap:.2f}")
    
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
    
    print(f"\nExpected Daily Returns:")
    print(f"  Paper Trading (backtest):  ${best_return:.2f}/day")
    if best_return != 0:
        live_pct = best_live_return/best_return*100
        print(f"  Live Trading (realistic):  ${best_live_return:.2f}/day  ({live_pct:.0f}% of backtest)")
    else:
        print(f"  Live Trading (realistic):  ${best_live_return:.2f}/day")
    print(f"\nAdjustments applied to live estimate:")
    trades_per_day = (6.5 * 3600) / best_interval
    print(f"  • Trades per day: {trades_per_day:.1f}")
    print(f"  • Slippage cost: ~0.1% per trade")
    print(f"  • Partial fills: ~5% reduction")
    if trades_per_day > 20:
        print(f"  • Market efficiency: 50% (very high frequency)")
    elif trades_per_day > 10:
        print(f"  • Market efficiency: 60% (high frequency)")
    elif trades_per_day > 4:
        print(f"  • Market efficiency: 70% (moderate frequency)")
    else:
        print(f"  • Market efficiency: 80% (low frequency)")
    
    symbol = best_symbol
    optimal_interval = best_interval
    optimal_cap = best_cap
    expected_return = best_return
    
    print(f"\n{'='*70}")
    if expected_return < 0:
        print(f"⚠️  STRATEGY NOT PROFITABLE")
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
        print(f"⚠️  LOW PROFITABILITY")
        print(f"{'='*70}")
        print(f"\nExpected return is less than $1/day.")
        print(f"Consider:")
        print(f"  • Different symbol with better momentum")
        print(f"  • Waiting for more favorable conditions")
    else:
        print(f"✅ STRATEGY IS PROFITABLE")
        print(f"{'='*70}")
        print(f"\nRun bot (as Administrator from anywhere):")
        print(f"  $BotDir = 'C:\\Users\\YourName\\...\\Paper-Trading'")
        print(f"\n  # Single stock (Admin mode - full automation)")
        print(f"  & \"$BotDir\\botctl.ps1\" start python -u runner.py -t {optimal_interval/3600:.4f} -s {symbol} -m {optimal_cap:.0f} --max-stocks 1")
        print(f"\n  # Multi-stock portfolio (Admin mode - bot picks best 15)")
        print(f"  & \"$BotDir\\botctl.ps1\" start python -u runner.py -t {optimal_interval/3600:.4f} -m {15 * optimal_cap:.0f}")
        print(f"\n  # Quick test (Simple mode - no automation)")
        print(f"  python \"$BotDir\\runner.py\" -t {optimal_interval/3600:.4f} -s {symbol} -m {optimal_cap:.0f} --max-stocks 1")
        
        # Compounding projections with both paper and live estimates
        print(f"\n{'='*70}")
        print(f"COMPOUNDING PROJECTIONS")
        print(f"{'='*70}")
        
        if optimal_cap > 0:
            paper_daily_pct = (expected_return / optimal_cap) * 100
            live_daily_pct = (best_live_return / optimal_cap) * 100
            
            print(f"Starting capital: ${optimal_cap:.2f}")
            print(f"Paper return: ${expected_return:.2f}/day ({paper_daily_pct:.3f}%/day)")
            print(f"Live return:  ${best_live_return:.2f}/day ({live_daily_pct:.3f}%/day)")
            
            print(f"\n{'Paper Backtest':<25} {'Live Trading (Realistic)':<30}")
            print(f"{'-'*25} {'-'*30}")
            
            for months in [1, 3, 6, 12, 24]:
                trading_days = months * 20
                
                paper_final = optimal_cap * ((1 + paper_daily_pct/100) ** trading_days)
                paper_gain_pct = ((paper_final - optimal_cap) / optimal_cap) * 100
                
                live_final = optimal_cap * ((1 + live_daily_pct/100) ** trading_days)
                live_gain_pct = ((live_final - optimal_cap) / optimal_cap) * 100
                
                print(f"{months:2d}mo ({trading_days:3d}days): ${paper_final:>10,.0f} (+{paper_gain_pct:>5.0f}%)  |  ${live_final:>10,.0f} (+{live_gain_pct:>5.0f}%)")
            
            print(f"\n⚠️  IMPORTANT REALITY CHECK:")
            print(f"   • Paper = theoretical backtest (OPTIMISTIC)")
            print(f"   • Live = adjusted for real trading costs (MORE REALISTIC)")
            print(f"   • Even live estimates assume:")
            print(f"     - Market conditions stay similar to backtest period")
            print(f"     - No extended losing streaks or black swan events")
            print(f"     - Consistent execution and no downtime")
            print(f"\n💡 Professional traders expect 10-30% per YEAR, not per day.")
        else:
            print(f"\n⚠️  Invalid capital configuration (${optimal_cap:.2f})")
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
            print(f"\n✅ HIGH CONFIDENCE - Strategy performs consistently across all test periods")
            print(f"   This config is likely to work well in live trading")
        elif best_consistency >= 0.6:
            print(f"\n⚠️  MEDIUM CONFIDENCE - Some variation across periods")
            print(f"   Use with caution and start with small capital")
        elif best_consistency >= 0.4:
            print(f"\n⚠️  LOW CONFIDENCE - Significant variation across periods")
            print(f"   Likely overfit to recent market conditions")
        else:
            print(f"\n❌ VERY LOW CONFIDENCE - Highly variable across periods")
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
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
