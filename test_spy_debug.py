#!/usr/bin/env python3
"""Debug SPY scoring"""

from runner import make_client, fetch_closes, simulate_signals_and_projection, compute_confidence, pct_stddev
import config

try:
    client = make_client(allow_missing=False, go_live=False)
    closes = fetch_closes(client, 'SPY', 900, 200)
    
    print(f"SPY closes: {len(closes)} bars")
    print(f"Min required: {max(config.LONG_WINDOW + 2, 25)}")
    
    if closes and len(closes) >= 25:
        conf = compute_confidence(closes)
        vol = pct_stddev(closes[-config.VOLATILITY_WINDOW:])
        sim = simulate_signals_and_projection(closes, 900, override_cap_usd=100)
        
        print(f"  Confidence: {conf:.6f}")
        print(f"  Volatility: {vol*100:.2f}%")
        print(f"  Expected daily: ${sim['expected_daily_usd']:.4f}")
        print(f"  Win rate: {sim['win_rate']*100:.1f}%")
        print(f"  Trades/day: {sim['expected_trades_per_day']:.2f}")
    else:
        print(f"NOT ENOUGH BARS: Need 25, got {len(closes)}")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

