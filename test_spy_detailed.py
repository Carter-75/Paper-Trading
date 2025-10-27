#!/usr/bin/env python3
"""Test SPY in detail"""

from stock_scanner import score_stock

result = score_stock('SPY', 900, 100, bars=200)

if result:
    print(f"SPY Scoring:")
    print(f"  Expected daily: ${result['expected_daily']:.4f}")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Win rate: {result['win_rate']*100:.1f}%")
    print(f"  Trades/day: {result['trades_per_day']:.2f}")
    print(f"  Confidence: {result['confidence']:.6f}")
    print(f"  Volatility: {result['volatility']*100:.2f}%")
else:
    print("SPY: Scoring FAILED (returned None)")

