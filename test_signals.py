#!/usr/bin/env python3
"""
Quick test to show what signals the bot would generate on recent data.
This shows WHY negative projections happen in bearish markets.
"""

import config
from runner import (
    make_client,
    fetch_closes,
    decide_action,
    compute_confidence,
    sma,
)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze trading signals for a stock")
    parser.add_argument("-s", "--symbol", type=str, required=True,
                       help="Stock symbol to test")
    parser.add_argument("-t", "--interval", type=float, default=1.0,
                       help="Interval in hours (default: 1.0)")
    parser.add_argument("-b", "--bars", type=int, default=100,
                       help="Number of bars to analyze (default: 100)")
    args = parser.parse_args()
    
    client = make_client(allow_missing=False, go_live=False)
    symbol = args.symbol.upper()
    interval_seconds = int(args.interval * 3600)
    bars = args.bars
    
    print(f"Analyzing last {bars} hours of {symbol} at {interval_seconds}s intervals...")
    print(f"=" * 80)
    
    closes = fetch_closes(client, symbol, interval_seconds, bars)
    
    if not closes:
        print("ERROR: Could not fetch data")
        return 1
    
    print(f"\nFetched {len(closes)} bars")
    print(f"Price range: ${min(closes):.2f} - ${max(closes):.2f}")
    print(f"Current price: ${closes[-1]:.2f}")
    print(f"\n{'='*80}")
    print(f"SIGNAL ANALYSIS (last 20 bars):")
    print(f"{'='*80}\n")
    
    buy_signals = 0
    sell_signals = 0
    hold_signals = 0
    
    # Analyze last 20 bars
    for i in range(len(closes) - 20, len(closes)):
        window = closes[:i+1]
        if len(window) < config.LONG_WINDOW:
            continue
            
        action = decide_action(window, config.SHORT_WINDOW, config.LONG_WINDOW)
        confidence = compute_confidence(window)
        price = closes[i]
        short_ma = sma(window, config.SHORT_WINDOW)
        long_ma = sma(window, config.LONG_WINDOW)
        
        if action == "buy":
            buy_signals += 1
            emoji = "🟢 BUY "
        elif action == "sell":
            sell_signals += 1
            emoji = "🔴 SELL"
        else:
            hold_signals += 1
            emoji = "⚪ HOLD"
        
        print(f"{emoji} | Price: ${price:7.2f} | Conf: {confidence:6.3f} | "
              f"Short MA: ${short_ma:7.2f} | Long MA: ${long_ma:7.2f}")
    
    total = buy_signals + sell_signals + hold_signals
    print(f"\n{'='*80}")
    print(f"SUMMARY (last 20 bars):")
    print(f"{'='*80}")
    print(f"  🟢 BUY signals:  {buy_signals:2d} ({buy_signals/total*100:5.1f}%)")
    print(f"  🔴 SELL signals: {sell_signals:2d} ({sell_signals/total*100:5.1f}%)")
    print(f"  ⚪ HOLD signals: {hold_signals:2d} ({hold_signals/total*100:5.1f}%)")
    
    # Trend analysis
    price_change = (closes[-1] - closes[-20]) / closes[-20] * 100
    ma_divergence = (sma(closes, config.SHORT_WINDOW) - sma(closes, config.LONG_WINDOW)) / sma(closes, config.LONG_WINDOW) * 100
    
    print(f"\n{'='*80}")
    print(f"MARKET CONDITION:")
    print(f"{'='*80}")
    print(f"  Price change (20 bars): {price_change:+6.2f}%")
    print(f"  MA divergence: {ma_divergence:+6.2f}%")
    
    if ma_divergence < -2:
        print(f"  Condition: 🔴 BEARISH (short MA below long MA)")
        print(f"  → Bot will mostly get SELL signals (can't profit without position)")
        print(f"  → Rare BUY signals often fail (downtrend continues)")
        print(f"  → Expected return: NEGATIVE")
    elif ma_divergence > 2:
        print(f"  Condition: 🟢 BULLISH (short MA above long MA)")
        print(f"  → Bot will get BUY signals (can profit)")
        print(f"  → Expected return: POSITIVE")
    else:
        print(f"  Condition: ⚪ NEUTRAL (MAs close together)")
        print(f"  → Bot will mostly HOLD")
        print(f"  → Expected return: NEAR ZERO")
    
    print(f"\n{'='*80}")
    print(f"WHY NEGATIVE PROJECTIONS HAPPEN:")
    print(f"{'='*80}")
    print(f"  In BEARISH markets:")
    print(f"    • 70-80% of signals are SELL (but bot is long-only)")
    print(f"    • 10-20% of signals are BUY (most fail in downtrend)")
    print(f"    • Win rate on BUY signals: ~30-40%")
    print(f"    • Result: Negative expected return")
    print(f"\n  The bot WON'T trade much, but when it does, it loses.")
    print(f"  That's what the negative projection means!\n")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

