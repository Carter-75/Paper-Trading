#!/usr/bin/env python3
"""
Validation script to ensure the multi-stock bot is configured correctly.
Run this before starting the bot to catch configuration errors.
"""

import sys
from typing import List, Tuple


def validate_stock_args(forced_stocks: List[str], max_stocks: int) -> Tuple[bool, str]:
    """Validate stock selection arguments."""
    if len(forced_stocks) > max_stocks:
        return False, (
            f"ERROR: You specified {len(forced_stocks)} forced stocks but max-stocks is {max_stocks}.\n"
            f"  Forced stocks: {', '.join(forced_stocks)}\n"
            f"  Max stocks: {max_stocks}\n"
            f"  Fix: Either reduce forced stocks OR increase --max-stocks to at least {len(forced_stocks)}"
        )
    
    # Check for duplicates
    if len(forced_stocks) != len(set(forced_stocks)):
        dupes = [s for s in forced_stocks if forced_stocks.count(s) > 1]
        return False, f"ERROR: Duplicate stocks in forced list: {', '.join(set(dupes))}"
    
    # Check for valid symbols
    for symbol in forced_stocks:
        if not symbol.isalpha() or len(symbol) > 5:
            return False, f"ERROR: Invalid stock symbol: {symbol}"
    
    return True, "Stock configuration valid"


def validate_capital(total_cap: float, cap_per_stock: float, max_stocks: int) -> Tuple[bool, str]:
    """Validate capital allocation."""
    if total_cap <= 0:
        return False, f"ERROR: Total capital must be positive, got {total_cap}"
    
    if cap_per_stock <= 0:
        return False, f"ERROR: Capital per stock must be positive, got {cap_per_stock}"
    
    max_possible_usage = cap_per_stock * max_stocks
    
    if max_possible_usage > total_cap * 2:
        return False, (
            f"WARNING: Capital allocation may exceed total capital!\n"
            f"  Total capital: ${total_cap:.2f}\n"
            f"  Cap per stock: ${cap_per_stock:.2f}\n"
            f"  Max stocks: {max_stocks}\n"
            f"  Max possible usage: ${max_possible_usage:.2f}\n"
            f"  This could lead to over-allocation. Consider reducing cap-per-stock."
        )
    
    return True, f"Capital allocation valid (${cap_per_stock:.2f} per stock, max ${max_possible_usage:.2f} total)"


def validate_interval(interval_hours: float) -> Tuple[bool, str]:
    """Validate trading interval."""
    if interval_hours <= 0:
        return False, f"ERROR: Interval must be positive, got {interval_hours}"
    
    if interval_hours < 0.0167:  # Less than 1 minute
        return False, f"ERROR: Interval too short ({interval_hours:.4f}h = {interval_hours*60:.1f}min). Minimum: 1 minute (0.0167h)"
    
    if interval_hours > 24:
        return False, f"WARNING: Interval very long ({interval_hours:.1f}h). Bot will trade infrequently."
    
    # Check if it's a supported interval
    interval_seconds = int(interval_hours * 3600)
    supported = [60, 300, 900, 3600, 14400, 86400]
    if interval_seconds not in supported:
        closest = min(supported, key=lambda x: abs(x - interval_seconds))
        return False, (
            f"WARNING: Interval {interval_seconds}s may not be supported by data provider.\n"
            f"  Supported: 60s (1min), 300s (5min), 900s (15min), 3600s (1hr), 14400s (4hr)\n"
            f"  Closest supported: {closest}s ({closest/3600:.4f}h)\n"
            f"  Consider using: -t {closest/3600:.4f}"
        )
    
    return True, f"Interval valid: {interval_seconds}s ({interval_hours}h)"


def simulate_args(args_dict: dict) -> None:
    """Simulate argument parsing and show what would happen."""
    print(f"\n{'='*70}")
    print("CONFIGURATION VALIDATION")
    print(f"{'='*70}\n")
    
    # Extract args
    interval = args_dict.get('interval', 0.25)
    total_cap = args_dict.get('total_cap', 1500.0)
    max_stocks = args_dict.get('max_stocks', 15)
    forced_stocks = args_dict.get('forced_stocks', [])
    cap_per_stock = args_dict.get('cap_per_stock', total_cap / max_stocks)
    rebalance_every = args_dict.get('rebalance_every', 4)
    
    # Validate each component
    all_valid = True
    
    # 1. Stock configuration
    valid, msg = validate_stock_args(forced_stocks, max_stocks)
    print(f"1. Stock Selection: {'✅' if valid else '❌'}")
    print(f"   {msg}")
    if not valid:
        all_valid = False
    
    # 2. Capital allocation
    valid, msg = validate_capital(total_cap, cap_per_stock, max_stocks)
    print(f"\n2. Capital Allocation: {'✅' if valid else '⚠️'}")
    print(f"   {msg}")
    
    # 3. Interval
    valid, msg = validate_interval(interval)
    print(f"\n3. Trading Interval: {'✅' if valid else '⚠️'}")
    print(f"   {msg}")
    
    # 4. Show what will happen
    print(f"\n{'='*70}")
    print("WHAT THE BOT WILL DO:")
    print(f"{'='*70}")
    print(f"  Total Capital: ${total_cap:.2f}")
    print(f"  Max Positions: {max_stocks}")
    print(f"  Capital per Stock: ${cap_per_stock:.2f}")
    print(f"  Trading Interval: {interval}h ({int(interval*3600)}s)")
    print(f"  Rebalance Every: {rebalance_every} intervals ({rebalance_every * interval:.2f}h)")
    print(f"\n  Stock Selection Strategy:")
    
    if forced_stocks:
        auto_count = max_stocks - len(forced_stocks)
        print(f"    • FORCED stocks (always kept): {', '.join(forced_stocks)}")
        print(f"    • AUTO-SELECTED slots: {auto_count}")
        if auto_count > 0:
            print(f"      → Bot will scan universe and pick {auto_count} best stocks")
            print(f"      → Bot can replace auto-selected stocks during rebalancing")
            print(f"      → Bot will NEVER replace forced stocks")
        else:
            print(f"      → No auto-selection (all slots used by forced stocks)")
    else:
        print(f"    • FULL AUTO-SELECTION: Bot picks all {max_stocks} stocks")
        print(f"      → Scans universe each rebalance cycle")
        print(f"      → Replaces underperformers automatically")
    
    print(f"\n  Trading Logic:")
    print(f"    • Each interval: Evaluate all {max_stocks} stock slots")
    print(f"    • For each stock: Analyze → BUY/SELL/HOLD")
    print(f"    • Every {rebalance_every} intervals: Check for better opportunities")
    print(f"    • Profitability gate: Min $0.10/day expected return")
    
    print(f"\n{'='*70}")
    if all_valid:
        print("✅ CONFIGURATION VALID - Ready to run!")
    else:
        print("❌ CONFIGURATION ERRORS - Fix issues above before running!")
    print(f"{'='*70}\n")
    
    return all_valid


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate multi-stock bot configuration")
    parser.add_argument("-t", "--interval", type=float, default=0.25,
                       help="Trading interval in hours")
    parser.add_argument("-m", "--total-cap", type=float, default=1500.0,
                       help="Total portfolio capital")
    parser.add_argument("--max-stocks", type=int, default=15,
                       help="Maximum number of stocks")
    parser.add_argument("--stocks", nargs="+", default=[],
                       help="Forced stocks")
    parser.add_argument("--cap-per-stock", type=float, default=None,
                       help="Capital per stock (default: total/max)")
    parser.add_argument("--rebalance-every", type=int, default=4,
                       help="Rebalance frequency")
    
    args = parser.parse_args()
    
    # Calculate cap per stock if not provided
    cap_per_stock = args.cap_per_stock or (args.total_cap / args.max_stocks)
    
    # Build args dict
    args_dict = {
        'interval': args.interval,
        'total_cap': args.total_cap,
        'max_stocks': args.max_stocks,
        'forced_stocks': [s.upper() for s in args.stocks],
        'cap_per_stock': cap_per_stock,
        'rebalance_every': args.rebalance_every,
    }
    
    # Validate
    valid = simulate_args(args_dict)
    
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())

