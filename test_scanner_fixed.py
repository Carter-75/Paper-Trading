#!/usr/bin/env python3
"""Test scanner with fixed requirements"""

from stock_scanner import scan_stocks, get_stock_universe

universe = get_stock_universe()
results = scan_stocks(universe[:10], 900, 100, max_results=10, verbose=False)

print(f'Found {len(results)} tradeable stocks:')
for r in results:
    print(f'  {r["symbol"]:6s}: ${r["expected_daily"]:6.2f}/day (score: {r["score"]:6.2f})')

