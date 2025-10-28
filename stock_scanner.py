#!/usr/bin/env python3
"""
Stock Scanner - Evaluates and ranks multiple stocks for trading opportunities.
"""

from typing import List, Tuple, Dict, Optional
import os
import json
import time
import config
from runner import (
    make_client,
    fetch_closes,
    simulate_signals_and_projection,
    compute_confidence,
    pct_stddev,
)


# Cache file for top stocks (refreshed daily at market open)
CACHE_FILE = "top_stocks_cache.json"
CACHE_DURATION_HOURS = 24  # Refresh every 24 hours (or once per trading day)

# Fallback list if dynamic fetch fails
FALLBACK_STOCK_UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA",
    "META", "AMD", "NFLX", "DIS", "BA", "JPM", "GS", "V",
    "MA", "COST", "WMT", "HD", "NKE", "MCD", "SBUX", "PEP"
]

# Top 100 US stocks by market cap (updated as of 2024)
# This list is used as the primary source and can be refreshed dynamically
DEFAULT_TOP_100_STOCKS = [
    # Mega-cap tech (Top 10)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "LLY", "V",
    # Large-cap tech & communication (11-30)
    "UNH", "XOM", "WMT", "JPM", "JNJ", "MA", "AVGO", "PG", "HD", "CVX",
    "COST", "ABBV", "MRK", "BAC", "ORCL", "CRM", "KO", "AMD", "PEP", "TMO",
    # Large-cap diversified (31-50)
    "CSCO", "ACN", "LIN", "MCD", "ADBE", "NFLX", "ABT", "WFC", "DHR", "NKE",
    "DIS", "VZ", "TXN", "CMCSA", "NEE", "PM", "INTC", "COP", "BMY", "UNP",
    # Mid-large cap (51-70)
    "RTX", "UPS", "HON", "QCOM", "INTU", "T", "LOW", "MS", "AMGN", "CAT",
    "BA", "SBUX", "DE", "ELV", "GE", "AMAT", "BLK", "MDT", "AXP", "PLD",
    # Growth & diversified (71-90)
    "ADI", "GILD", "BKNG", "SYK", "LRCX", "AMT", "ISRG", "CI", "MMC", "VRTX",
    "TJX", "REGN", "C", "CVS", "PGR", "MDLZ", "ZTS", "NOC", "SCHW", "MU",
    # Additional liquid stocks (91-100)
    "CB", "EOG", "SO", "DUK", "BSX", "BDX", "ITW", "MMM", "APD", "SLB",
    # ETFs for diversification
    "SPY", "QQQ", "IWM", "DIA"
]


def is_market_open_today() -> bool:
    """Check if market opened today (used to trigger daily refresh)."""
    try:
        from datetime import datetime
        import pytz
        now = datetime.now(pytz.timezone('US/Eastern'))
        # Market opens at 9:30 AM ET
        return now.hour >= 9 and now.weekday() < 5  # Weekday and after 9 AM
    except:
        return True  # Assume yes if check fails


def fetch_top_stocks_dynamic(limit: int = 100, force_refresh: bool = False) -> List[str]:
    """
    Fetch top stocks dynamically from market data.
    Uses caching to avoid excessive API calls.
    
    Args:
        limit: Number of top stocks to return (default: 100)
        force_refresh: Force refresh even if cache is valid (default: False)
    
    Returns:
        List of stock symbols
    """
    # Check cache first (unless force refresh)
    if not force_refresh and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
                cache_time = cache.get('timestamp', 0)
                cache_age_hours = (time.time() - cache_time) / 3600
                
                # Check if cache is from a previous trading day
                from datetime import datetime
                import pytz
                cache_datetime = datetime.fromtimestamp(cache_time, pytz.timezone('US/Eastern'))
                now_datetime = datetime.now(pytz.timezone('US/Eastern'))
                is_same_day = cache_datetime.date() == now_datetime.date()
                
                # Use cache if: less than 24h old AND same trading day
                if cache_age_hours < CACHE_DURATION_HOURS and is_same_day:
                    symbols = cache.get('symbols', [])
                    if len(symbols) >= limit:
                        print(f"✓ Using cached top {len(symbols[:limit])} stocks (age: {cache_age_hours:.1f}h)")
                        return symbols[:limit]
                else:
                    if not is_same_day:
                        print(f"New trading day detected - refreshing stock list...")
                    else:
                        print(f"Cache expired ({cache_age_hours:.1f}h old) - refreshing stock list...")
        except Exception:
            pass
    
    # Try to fetch dynamically using yfinance
    try:
        import yfinance as yf
        import pandas as pd
        
        # Get S&P 500 components
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(sp500_url)
        sp500_table = tables[0]
        symbols = sp500_table['Symbol'].tolist()
        
        # Clean symbols (remove dots, special chars that cause issues)
        cleaned_symbols = []
        for sym in symbols:
            # Replace dots with hyphens (BRK.B -> BRK-B)
            clean_sym = sym.replace('.', '-')
            cleaned_symbols.append(clean_sym)
        
        # Get market caps for top symbols
        market_caps = {}
        print(f"Fetching market caps for {len(cleaned_symbols)} stocks (this may take a minute)...")
        
        for i, symbol in enumerate(cleaned_symbols[:200], 1):  # Check top 200 from S&P 500
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if 'marketCap' in info and info['marketCap']:
                    market_caps[symbol] = info['marketCap']
                if i % 20 == 0:
                    print(f"  Progress: {i}/200 stocks checked...")
            except Exception:
                continue
        
        # Sort by market cap and take top N
        sorted_symbols = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [sym for sym, _ in sorted_symbols[:limit]]
        
        # Add major ETFs for diversification
        etfs = ["SPY", "QQQ", "IWM", "DIA"]
        for etf in etfs:
            if etf not in top_symbols:
                top_symbols.append(etf)
        
        # Cache the results
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'symbols': top_symbols
                }, f)
        except Exception:
            pass
        
        print(f"✓ Successfully fetched top {len(top_symbols)} stocks")
        return top_symbols[:limit]
        
    except Exception as e:
        print(f"Warning: Could not fetch dynamic stock list ({e})")
        print("Using predefined top 100 stocks...")
        return DEFAULT_TOP_100_STOCKS[:limit]


def score_stock(symbol: str, interval_seconds: int, cap_per_stock: float, bars: int = 200, verbose: bool = False) -> Optional[Dict]:
    """
    Score a stock based on profitability potential.
    Returns dict with score and metrics, or None if failed.
    """
    try:
        # Add volume filter to avoid illiquid stocks
        import yfinance as yf
        ticker_obj = yf.Ticker(symbol)
        try:
            info = ticker_obj.info
            avg_volume = info.get('averageVolume', 0)
            if avg_volume < config.MIN_AVG_VOLUME:
                if verbose:
                    print(f"  {symbol}: Low volume ({avg_volume:,} < {config.MIN_AVG_VOLUME:,})")
                return None
        except:
            pass  # If volume check fails, continue anyway
        
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
        
        score_data = score_stock(symbol, interval_seconds, cap_per_stock, verbose=verbose)
        
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


def get_stock_universe(user_symbols: Optional[List[str]] = None, use_top_100: bool = True, force_refresh: bool = False) -> List[str]:
    """
    Get list of stocks to scan.
    
    Args:
        user_symbols: User-provided list of symbols, or None for automatic selection
        use_top_100: If True, uses top 100 stocks by market cap (default: True)
        force_refresh: Force refresh of top 100 list (default: False)
    
    Returns:
        List of stock symbols to scan
    """
    if user_symbols:
        # Use user-provided symbols
        return [s.upper() for s in user_symbols]
    elif use_top_100:
        # Fetch top 100 stocks dynamically (with daily caching)
        return fetch_top_stocks_dynamic(limit=100, force_refresh=force_refresh)
    else:
        # Use fallback list (for testing or if dynamic fetch is disabled)
        return FALLBACK_STOCK_UNIVERSE.copy()

