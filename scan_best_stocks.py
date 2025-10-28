#!/usr/bin/env python3
"""
Stock Scanner CLI - Find the best stocks to trade right now.
Usage: python scan_best_stocks.py --interval 0.25 --cap 100 --top 5
"""

import argparse
import sys
from stock_scanner import scan_stocks, get_stock_universe, DEFAULT_TOP_100_STOCKS


def main():
    parser = argparse.ArgumentParser(description="Scan stocks and find best trading opportunities")
    parser.add_argument("-t", "--interval", type=float, default=0.25,
                       help="Trading interval in hours (default: 0.25 = 15min)")
    parser.add_argument("-c", "--cap", type=float, default=100.0,
                       help="Capital per stock in USD (default: 100)")
    parser.add_argument("-n", "--top", type=int, default=5,
                       help="Number of top stocks to show (default: 5)")
    parser.add_argument("-s", "--symbols", nargs="+",
                       help="Specific symbols to scan (default: scan top 100 stocks by market cap)")
    parser.add_argument("--no-dynamic", action="store_true",
                       help="Use predefined list instead of dynamic top 100")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Show detailed progress")
    
    args = parser.parse_args()
    
    interval_seconds = int(args.interval * 3600)
    
    # Get stock universe
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = get_stock_universe(use_top_100=not args.no_dynamic)
    
    print(f"\n{'='*80}")
    print(f"STOCK SCANNER")
    print(f"{'='*80}")
    print(f"Interval: {interval_seconds}s ({args.interval}h)")
    print(f"Cap per stock: ${args.cap}")
    print(f"Scanning: {len(symbols)} stocks")
    if not args.symbols and not args.no_dynamic:
        print(f"Mode: Top 100 by market cap (cached)")
    print(f"{'='*80}\n")
    
    results = scan_stocks(
        symbols=symbols,
        interval_seconds=interval_seconds,
        cap_per_stock=args.cap,
        max_results=args.top,
        verbose=args.verbose
    )
    
    if not results:
        print("\n❌ No profitable stocks found!")
        print("   Try different interval or increase capital.")
        return 1
    
    print(f"\n{'='*80}")
    print(f"TOP {len(results)} STOCKS")
    print(f"{'='*80}\n")
    
    profitable_count = sum(1 for r in results if r['expected_daily'] > 0)
    
    for i, stock in enumerate(results, 1):
        symbol = stock['symbol']
        exp_daily = stock['expected_daily']
        win_rate = stock['win_rate'] * 100
        trades_day = stock['trades_per_day']
        price = stock['current_price']
        
        if exp_daily > 0:
            status = "✅"
        elif exp_daily > -0.5:
            status = "⚠️ "
        else:
            status = "❌"
        
        print(f"{i}. {status} {symbol:6s} | ${exp_daily:7.2f}/day | "
              f"Win: {win_rate:5.1f}% | Trades: {trades_day:4.1f}/day | "
              f"Price: ${price:7.2f}")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Profitable stocks: {profitable_count}/{len(results)}")
    
    if profitable_count > 0:
        best = results[0]
        print(f"\n🏆 BEST STOCK: {best['symbol']}")
        print(f"   Expected return: ${best['expected_daily']:.2f}/day")
        print(f"   Win rate: {best['win_rate']*100:.1f}%")
        print(f"\n   Run with:")
        print(f"   python runner.py -t {args.interval} -s {best['symbol']} -m {args.cap}")
    else:
        print(f"\n⚠️  All stocks show negative returns.")
        print(f"   Suggestions:")
        print(f"   - Try different interval (shorter: -t 0.083, longer: -t 1.0)")
        print(f"   - Try different stocks (-s AAPL MSFT GOOGL)")
        print(f"   - Wait for better market conditions")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

