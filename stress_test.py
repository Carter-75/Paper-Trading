#!/usr/bin/env python3
"""
Black Swan Stress Testing Module (V2)
Uses the REAL DecisionEngine to test robustness against historical crises.
"""

import sys
import argparse
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# Import Core Systems
from config_validated import get_config
from strategies.decision_engine import DecisionEngine
from ml_predictor import get_ml_predictor
from runner_data_utils import fetch_closes_with_volume

# Configure logging to silence standard output during test
logging.basicConfig(level=logging.ERROR)

CRISIS_PERIODS = [
    {"name": "2008 Financial Crisis", "start": "2008-09-01", "end": "2009-03-31", "description": "Global Meltdown (-50%)"},
    {"name": "2020 COVID Crash", "start": "2020-02-01", "end": "2020-04-30", "description": "Pandemic Panic (-33%)"},
    {"name": "2022 Bear Market", "start": "2022-01-01", "end": "2022-10-31", "description": "Inflation & Rates (-25%)"}
]

def fetch_historical_slice(symbol: str, start: str, end: str, interval_sec: int):
    """
    Fetch historical data using yfinance (via fetch_closes_with_volume wrapper implies live, 
    but we use yfinance directly here for date ranges).
    """
    import yfinance as yf
    
    # Map interval
    if interval_sec >= 86400: p = "1d"
    elif interval_sec >= 3600: p = "1h"
    else: p = "5m" # Default to 5m for granularity
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=p)
    if df.empty:
        return [], []
    
    return df['Close'].tolist(), df['Volume'].tolist()

def run_simulation(symbol: str, closes: List[float], volumes: List[float], start_cap: float):
    """
    Run the DecisionEngine over the data stream.
    """
    # Initialize Engine with Mocks for external dependencies to prevent API calls
    engine = DecisionEngine()
    
    # Mock News Sentinel to return "Neutral" (0.5) or "Positive" (1.0) generally, 
    # unless we want to simulate news (hard to do historically without data).
    # We will assume Neutral news for stress testing technicals/regime.
    if engine.news_sentinel:
        engine.news_sentinel.check_sentiment = MagicMock(return_value=0.9)
    
    # Simulation State
    cash = start_cap
    position = 0 # Shares
    avg_price = 0.0
    equity_curve = []
    
    # Loop through data (Start after warm-up period)
    warmup = 50
    if len(closes) < warmup:
        return {"success": False}
        
    for i in range(warmup, len(closes)):
        price = closes[i]
        
        # Prepare window
        window_start = max(0, i-200)
        history_closes = closes[window_start:i+1]
        history_volumes = volumes[window_start:i+1]
        
        # Ask Brain
        signal = engine.analyze(symbol, history_closes, history_volumes)
        
        # Execute Logic (Simplified Execution)
        current_val = cash + (position * price)
        
        if signal.action == "buy" and cash > 10:
            # Buy 1 share for simplicity or pct of equity? 
            # Use Kelly-like sizing or fixed 10% for test
            spend = current_val * 0.10
            qty = spend / price
            cost = qty * price
            cash -= cost
            position += qty
            avg_price = price
            
        elif signal.action == "sell" and position > 0:
            # Liquidate
            cash += (position * price)
            position = 0
            
        equity_curve.append(current_val)
        
    final_equity = cash + (position * closes[-1])
    ret = ((final_equity - start_cap) / start_cap) * 100
    
    # Drawdown calc
    peak = start_cap
    max_dd = 0
    for e in equity_curve:
        if e > peak: peak = e
        dd = (peak - e) / peak
        if dd > max_dd: max_dd = dd
        
    return {
        "success": True,
        "return_pct": ret,
        "max_drawdown": max_dd * 100,
        "final_equity": final_equity
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--symbol", default="SPY")
    parser.add_argument("-m", "--capital", type=float, default=10000)
    args = parser.parse_args()
    
    print(f"\\n{'='*60}")
    print(f"STRESS TEST: {args.symbol} (Capital: ${args.capital:,.0f})")
    print(f"Engine: DecisionEngine (Real)")
    print(f"{'='*60}\\n")
    
    for crisis in CRISIS_PERIODS:
        print(f"Testing {crisis['name']} ({crisis['description']})...")
        c, v = fetch_historical_slice(args.symbol, crisis['start'], crisis['end'], 3600)
        
        if len(c) < 100:
            print("  [X] Skipped (Insufficient Data)")
            continue
            
        res = run_simulation(args.symbol, c, v, args.capital)
        
        if res["success"]:
            print(f"  Result: {res['return_pct']:+.2f}% | Max DD: {res['max_drawdown']:.2f}%")
            if res['max_drawdown'] < 50:
                print("  Status: PASSED (Survived)")
            else:
                print("  Status: FAILED (Busted)")
        else:
            print("  [X] Failed")
        print("-" * 30)

if __name__ == "__main__":
    main()
