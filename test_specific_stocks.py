#!/usr/bin/env python3
"""Test specific stocks that showed positive earlier"""

from stock_scanner import scan_stocks

symbols = ["SPY", "TSLA", "QQQ", "DIA"]
results = scan_stocks(symbols, 900, 100, max_results=10, verbose=False)

print(f'Found {len(results)} tradeable stocks:')
for r in results:
    print(f'  {r["symbol"]:6s}: ${r["expected_daily"]:7.2f}/day (score: {r["score"]:7.2f})')
    
if not results:
    print("  None! All filtered out")

