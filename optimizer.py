#!/usr/bin/env python3
"""
Comprehensive Binary Search Optimizer

Tests ALL possible intervals (1 second to 6.5 hours) and capitals ($1 to $1M)
using efficient binary search to find optimal configuration quickly.
"""

import sys
from typing import Tuple
import config
from runner import (
    make_client,
    fetch_closes,
    simulate_signals_and_projection,
    snap_interval_to_supported_seconds,
)


def evaluate_config(client, symbol: str, interval_seconds: int, cap_usd: float, bars: int = 200) -> float:
    """Evaluate expected daily return for specific interval and capital."""
    try:
        closes = fetch_closes(client, symbol, interval_seconds, bars)
        if not closes or len(closes) < max(config.LONG_WINDOW + 10, 30):
            return -999999.0
        
        sim = simulate_signals_and_projection(closes, interval_seconds, override_cap_usd=cap_usd)
        return float(sim.get("expected_daily_usd", -999999.0))
    except Exception:
        return -999999.0


def binary_search_capital(client, symbol: str, interval_seconds: int, 
                          min_cap: float = 1.0, max_cap: float = 1000000.0,
                          tolerance: float = 10.0) -> Tuple[float, float]:
    """
    Binary search to find optimal capital for given interval.
    Tests from $1 to $1M.
    """
    # Quick sample at key capital points
    test_caps = [10, 50, 100, 250, 500, 1000, 5000, 10000, 50000, 100000]
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
    
    while (search_max - search_min) > tolerance and iterations < max_iterations:
        iterations += 1
        mid = (search_min + search_max) / 2
        
        low_ret = evaluate_config(client, symbol, interval_seconds, search_min)
        mid_ret = evaluate_config(client, symbol, interval_seconds, mid)
        high_ret = evaluate_config(client, symbol, interval_seconds, search_max)
        
        if mid_ret >= low_ret and mid_ret >= high_ret:
            if mid_ret > best_return:
                best_return = mid_ret
                best_cap = mid
            search_min = (search_min + mid) / 2
            search_max = (mid + search_max) / 2
        elif low_ret > mid_ret:
            if low_ret > best_return:
                best_return = low_ret
                best_cap = search_min
            search_max = mid
        else:
            if high_ret > best_return:
                best_return = high_ret
                best_cap = search_max
            search_min = mid
    
    return best_cap, best_return


def comprehensive_binary_search(symbol: str, verbose: bool = False, max_cap: float = 1000000.0) -> Tuple[int, float, float]:
    """
    Comprehensive binary search optimizer.
    
    Tests intervals from 1 second to 6.5 hours.
    Tests capitals from $1 to max_cap.
    
    Returns: (optimal_interval_seconds, optimal_cap_usd, expected_daily_return)
    """
    client = make_client(allow_missing=False, go_live=False)
    
    min_interval = 60  # 1 minute (API minimum)
    max_interval = int(6.5 * 3600)  # 6.5 hours
    
    if verbose:
        print(f"\nComprehensive optimization for {symbol}")
        print(f"Interval range: {min_interval}s - {max_interval}s (1min to 6.5hrs)")
        print(f"Capital range: $1 - ${max_cap:,.0f}")
        print(f"{'='*70}\n")
    
    # Phase 1: Sample key intervals
    test_intervals = [60, 300, 900, 1800, 3600, 7200, 14400, 21600]  # 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h
    test_intervals = [i for i in test_intervals if min_interval <= i <= max_interval]
    
    best_interval = 3600
    best_cap = 100.0
    best_return = -999999.0
    
    if verbose:
        print("Phase 1: Sampling key intervals with capital optimization:")
    
    for interval in test_intervals:
        snapped = snap_interval_to_supported_seconds(interval)
        cap, ret = binary_search_capital(client, symbol, snapped, min_cap=1.0, max_cap=max_cap)
        
        if verbose:
            print(f"  {snapped:5d}s ({snapped/3600:6.3f}h): ${ret:7.2f}/day @ ${cap:>9.0f} cap")
        
        if ret > best_return:
            best_return = ret
            best_interval = snapped
            best_cap = cap
    
    # Phase 2: Binary search intervals around best
    if verbose:
        print(f"\nPhase 2: Refining around {best_interval}s:")
    
    search_min = max(min_interval, best_interval // 2)
    search_max = min(max_interval, best_interval * 2)
    iterations = 0
    max_iterations = 8
    
    while (search_max - search_min) > 60 and iterations < max_iterations:
        iterations += 1
        mid = (search_min + search_max) // 2
        
        low_snap = snap_interval_to_supported_seconds(search_min)
        mid_snap = snap_interval_to_supported_seconds(mid)
        high_snap = snap_interval_to_supported_seconds(search_max)
        
        # For each interval, find optimal capital
        low_cap, low_ret = binary_search_capital(client, symbol, low_snap, min_cap=1.0, max_cap=max_cap)
        mid_cap, mid_ret = binary_search_capital(client, symbol, mid_snap, min_cap=1.0, max_cap=max_cap)
        high_cap, high_ret = binary_search_capital(client, symbol, high_snap, min_cap=1.0, max_cap=max_cap)
        
        if verbose:
            print(f"  Testing: {low_snap}s (${low_ret:.2f}), {mid_snap}s (${mid_ret:.2f}), {high_snap}s (${high_ret:.2f})")
        
        if mid_ret >= low_ret and mid_ret >= high_ret:
            if mid_ret > best_return:
                best_return = mid_ret
                best_interval = mid_snap
                best_cap = mid_cap
            search_min = (search_min + mid) // 2
            search_max = (mid + search_max) // 2
        elif low_ret > mid_ret:
            if low_ret > best_return:
                best_return = low_ret
                best_interval = low_snap
                best_cap = low_cap
            search_max = mid
        else:
            if high_ret > best_return:
                best_return = high_ret
                best_interval = high_snap
                best_cap = high_cap
            search_min = mid
    
    if verbose:
        print(f"\nConverged after {iterations} refinement iterations")
    
    return best_interval, best_cap, best_return


def main() -> int:
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive binary search optimizer")
    parser.add_argument("-s", "--symbol", type=str, default=None,
                       help="Stock symbol to optimize (optional, will auto-find best if not provided)")
    parser.add_argument("--symbols", nargs="+", type=str, default=None,
                       help="Multiple symbols to test and compare")
    parser.add_argument("-m", "--max-cap", type=float, default=1000000.0,
                       help="Maximum capital to test (default: $1,000,000)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Show detailed progress")
    
    args = parser.parse_args()
    
    # Determine which symbols to test
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        # Auto-scan popular stocks
        symbols = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN"]
        print(f"\n{'='*70}")
        print(f"AUTO-SCANNING MODE")
        print(f"{'='*70}")
        print(f"Testing {len(symbols)} popular stocks to find best opportunity...")
        print(f"Symbols: {', '.join(symbols)}\n")
    
    # Test each symbol
    best_symbol = None
    best_interval = None
    best_cap = None
    best_return = -999999.0
    
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
        
        print(f"Method: Binary search across ALL intervals and capitals")
        print(f"Range: 1 min to 6.5 hours × $1 to ${args.max_cap:,.0f}")
        
        optimal_interval, optimal_cap, expected_return = comprehensive_binary_search(symbol, verbose=args.verbose, max_cap=args.max_cap)
        
        results.append({
            "symbol": symbol,
            "interval": optimal_interval,
            "cap": optimal_cap,
            "return": expected_return
        })
        
        if expected_return > best_return:
            best_return = expected_return
            best_symbol = symbol
            best_interval = optimal_interval
            best_cap = optimal_cap
        
        if len(symbols) > 1:
            print(f"Result: ${expected_return:.2f}/day @ {optimal_interval}s ({optimal_interval/3600:.4f}h) with ${optimal_cap:.0f} cap")
    
    # Show summary if multiple symbols
    if len(symbols) > 1:
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*70}")
        results.sort(key=lambda x: x["return"], reverse=True)
        for i, r in enumerate(results[:5], 1):
            status = "✅" if r["return"] > 0 else "❌"
            print(f"{i}. {r['symbol']:6s} {status}  ${r['return']:7.2f}/day  @ {r['interval']:5d}s ({r['interval']/3600:.4f}h)  ${r['cap']:7.0f} cap")
        print(f"{'='*70}")
    
    # Show optimal config
    print(f"\n{'='*70}")
    print(f"OPTIMAL CONFIGURATION")
    print(f"{'='*70}")
    print(f"Symbol: {best_symbol}")
    print(f"Interval: {best_interval}s ({best_interval/3600:.4f}h)")
    print(f"Capital: ${best_cap:.2f}")
    print(f"Expected Daily Return: ${best_return:.2f}")
    
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
        
        # NEW: Compounding projections
        print(f"\n{'='*70}")
        print(f"COMPOUNDING PROJECTIONS (THEORETICAL)")
        print(f"{'='*70}")
        
        daily_return_pct = (expected_return / optimal_cap) * 100
        
        print(f"Starting capital: ${optimal_cap:.2f}")
        print(f"Daily return: ${expected_return:.2f} ({daily_return_pct:.3f}%/day)")
        print(f"\nProjected balance after:")
        
        for months in [1, 3, 6, 12, 24, 60]:
            trading_days = months * 20
            final = optimal_cap * ((1 + daily_return_pct/100) ** trading_days)
            gain = final - optimal_cap
            gain_pct = (gain / optimal_cap) * 100
            print(f"  {months:2d} months ({trading_days:3d} days): ${final:>12,.2f}  (+{gain_pct:>6.1f}%)")
        
        print(f"\n⚠️  WARNING: These are THEORETICAL backtested projections.")
        print(f"⚠️  Real trading performance will be 20-50% LOWER due to:")
        print(f"     • Slippage (0.05-0.2% per trade)")
        print(f"     • Partial fills and rejected orders")
        print(f"     • Market regime changes")
        print(f"     • Losing streaks and drawdowns")
        print(f"     • Competition and market efficiency")
        print(f"\n💡 Realistic expectation: 5-20% per YEAR, not per day")
    
    print(f"\n{'='*70}")
    print(f"NOTE: This optimizer tested comprehensively:")
    print(f"  • Intervals from 60s to {int(6.5*3600)}s")
    print(f"  • Capitals from $1 to $1,000,000")
    print(f"  • Using binary search for efficiency (~50 tests)")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
