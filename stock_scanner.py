#!/usr/bin/env python3
"""
Stock Scanner - Evaluates and ranks multiple stocks for trading opportunities.
"""

from typing import List, Tuple, Dict, Optional
import config
from runner import (
    make_client,
    fetch_closes,
    simulate_signals_and_projection,
    compute_confidence,
    pct_stddev,
)


# Popular stocks with good liquidity
DEFAULT_STOCK_UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA",
    "META", "AMD", "NFLX", "DIS", "BA", "JPM", "GS", "V",
    "MA", "COST", "WMT", "HD", "NKE", "MCD", "SBUX", "PEP"
]


def score_stock(symbol: str, interval_seconds: int, cap_per_stock: float, bars: int = 200) -> Optional[Dict]:
    """
    Score a stock based on profitability potential.
    Returns dict with score and metrics, or None if failed.
    """
    try:
        client = make_client(allow_missing=False, go_live=False)
        closes = fetch_closes(client, symbol, interval_seconds, bars)
        
        # Minimum bars needed
        min_bars = config.LONG_WINDOW + 2
        if not closes or len(closes) < min_bars:
            return None
        
        # Calculate metrics
        confidence = compute_confidence(closes)
        vol_pct = pct_stddev(closes[-config.VOLATILITY_WINDOW:])
        
        # Run simulation with the provided capital
        # cap_per_stock is actually max_cap when called from runner.py
        print(f"  DEBUG SCANNER: {symbol} using cap=${cap_per_stock}")
        sim = simulate_signals_and_projection(
            closes,
            interval_seconds,
            override_cap_usd=cap_per_stock  # This should match what allocation uses
        )
        
        expected_daily = float(sim.get("expected_daily_usd", 0.0))
        trades_per_day = float(sim.get("expected_trades_per_day", 0.0))
        win_rate = float(sim.get("win_rate", 0.0))
        
        # Calculate composite score
        # Prioritize: expected return, then win rate, penalize excessive trading
        score = expected_daily
        if win_rate > 0.5:
            score *= (1 + (win_rate - 0.5))  # Bonus for high win rate
        if trades_per_day > 10:
            score *= 0.8  # Penalty for overtrading
        
        return {
            "symbol": symbol,
            "score": score,
            "expected_daily": expected_daily,
            "confidence": confidence,
            "volatility": vol_pct,
            "trades_per_day": trades_per_day,
            "win_rate": win_rate,
            "current_price": closes[-1] if closes else 0.0
        }
    except Exception as e:
        if verbose:
            print(f"Warning: Could not score {symbol}: {e}")
            import traceback
            traceback.print_exc()
        return None


def scan_stocks(symbols: List[str], interval_seconds: int, 
                cap_per_stock: float, max_results: int = 15,
                verbose: bool = False) -> List[Dict]:
    """
    Scan multiple stocks and return top N ranked by profitability.
    
    Args:
        symbols: List of stock symbols to scan
        interval_seconds: Trading interval
        cap_per_stock: Capital allocation per stock
        max_results: Maximum number of stocks to return
        verbose: Print progress
    
    Returns:
        List of dicts with stock scores and metrics, sorted best to worst
    """
    if verbose:
        print(f"\nScanning {len(symbols)} stocks...")
    
    results = []
    for i, symbol in enumerate(symbols, 1):
        if verbose:
            print(f"  [{i:2d}/{len(symbols)}] Evaluating {symbol:6s}...", end="", flush=True)
        
        score_data = score_stock(symbol, interval_seconds, cap_per_stock)
        
        if score_data:
            results.append(score_data)
            if verbose:
                exp_daily = score_data["expected_daily"]
                print(f" ${exp_daily:7.2f}/day (score: {score_data['score']:7.2f})")
        else:
            if verbose:
                print(" FAILED")
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top N
    return results[:max_results]


def get_stock_universe(user_symbols: Optional[List[str]] = None) -> List[str]:
    """
    Get list of stocks to scan.
    
    Args:
        user_symbols: User-provided list of symbols, or None for default universe
    
    Returns:
        List of stock symbols to scan
    """
    if user_symbols:
        # Use user-provided symbols
        return [s.upper() for s in user_symbols]
    else:
        # Use default universe
        return DEFAULT_STOCK_UNIVERSE.copy()

