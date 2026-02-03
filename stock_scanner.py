#!/usr/bin/env python3
"""
Stock Scanner - Evaluates and ranks multiple stocks for trading opportunities.
"""

from typing import List, Tuple, Dict, Optional
import os
import json
import time
import config
import traceback
from runner_data_utils import (
    make_client,
    fetch_closes_with_volume,
)
from simulation import run_backtest_simulation
import numpy as np

def pct_stddev(data: List[float]) -> float:
    if len(data) < 2: return 0.0
    return float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else 0.0

# Safe print for scheduled task mode (no console attached)
def safe_print(*args, **kwargs):
    """Print that doesn't crash when stdout isn't available (scheduled task mode)"""
    try:
        print(*args, **kwargs)
    except (OSError, AttributeError):
        pass  # Silently ignore if no console available


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
    except Exception as e:
        safe_print(f"is_market_open_today check failed: {e}")
        safe_print(traceback.format_exc())
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
                        safe_print(f"[OK] Using cached top {len(symbols[:limit])} stocks (age: {cache_age_hours:.1f}h)")
                        return symbols[:limit]
                else:
                    if not is_same_day:
                        safe_print(f"New trading day detected - refreshing stock list...")
                    else:
                        safe_print(f"Cache expired ({cache_age_hours:.1f}h old) - refreshing stock list...")
        except Exception as e:
            safe_print(f"Failed to read cache {CACHE_FILE}: {e}")
            safe_print(traceback.format_exc())
    
    # Try to fetch dynamically using yfinance
    try:
        import yfinance as yf
        import pandas as pd
        import requests
        from io import StringIO
        
        # Get S&P 500 components (with headers to avoid 403)

        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(sp500_url, headers=headers, timeout=10)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        sp500_table = tables[0]

        # Robustly find the symbol column (works for unnamed or integer columns)
        symbol_col = None
        # Try string columns first
        for col in sp500_table.columns:
            if isinstance(col, str) and col.lower() in ('symbol', 'ticker'):
                symbol_col = col
                break
        # If not found, try integer columns by checking sample values
        if symbol_col is None:
            for col in sp500_table.columns:
                # Check if the first few values look like stock symbols (all uppercase, short, no spaces)
                sample = sp500_table[col].astype(str).head(10).tolist()
                if all((s.isupper() or ('.' in s and s.replace('.', '').isupper())) and 1 <= len(s) <= 6 and ' ' not in s for s in sample):
                    symbol_col = col
                    break
        if symbol_col is None:
            # Fallback: just use the first column
            symbol_col = sp500_table.columns[0]

        symbols = sp500_table[symbol_col].astype(str).tolist()

        # Clean symbols (remove dots, special chars that cause issues)
        cleaned_symbols = []
        for sym in symbols:
            # Replace dots with hyphens (BRK.B -> BRK-B)
            clean_sym = sym.replace('.', '-')
            # Remove any whitespace or stray characters
            clean_sym = clean_sym.strip().upper()
            if clean_sym and clean_sym.isalnum() or ('-' in clean_sym and clean_sym.replace('-', '').isalnum()):
                cleaned_symbols.append(clean_sym)
        
        # Get market caps for top symbols
        market_caps = {}
        safe_print(f"Fetching market caps for {len(cleaned_symbols)} stocks (this may take a minute)...")
        
        for i, symbol in enumerate(cleaned_symbols[:200], 1):  # Check top 200 from S&P 500
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if 'marketCap' in info and info['marketCap']:
                    market_caps[symbol] = info['marketCap']
                if i % 20 == 0:
                    safe_print(f"  Progress: {i}/200 stocks checked...")
            except Exception as e:
                safe_print(f"Market cap fetch failed for {symbol}: {e}")
                # Do not print traceback for expected API errors
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
        except Exception as e:
            safe_print(f"Failed to write cache {CACHE_FILE}: {e}")
            safe_print(traceback.format_exc())
        
        safe_print(f"[OK] Successfully fetched top {len(top_symbols)} stocks")
        return top_symbols[:limit]
        
    except Exception as e:
        safe_print(f"Warning: Could not fetch dynamic stock list ({e})")
        safe_print("Using predefined top 100 stocks...")
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
                    safe_print(f" skipped (low volume: {avg_volume:,})")
                return None
        except Exception as e:
            safe_print(f"Volume check failed for {symbol}: {e}")
            # If volume check fails, continue anyway
        
        client = make_client(allow_missing=False, go_live=False)
        closes, volumes = fetch_closes_with_volume(client, symbol, interval_seconds, bars)
        # Accept as many bars as possible, as long as hard minimum is met
        min_bars = config.LONG_WINDOW + 2
        if not closes or len(closes) < min_bars:
            if verbose:
                safe_print(f" [skipped: insufficient data ({len(closes) if closes else 0}/{min_bars} bars, need at least {min_bars})]")
            return None
        # If less than requested but above minimum, proceed and log
        if verbose and len(closes) < bars:
            safe_print(f" [using {len(closes)}/{bars} bars: partial data accepted]")
        
        # Calculate metrics
        vol_pct = pct_stddev(closes[-config.VOLATILITY_WINDOW:])
        
        # Run simulation with the provided capital
        # cap_per_stock is actually max_cap when called from runner.py
        sim = run_backtest_simulation(
            closes,
            volumes, # Need volumes!
            interval_seconds,
            start_capital=cap_per_stock
        )
        # Use simple confidence placeholder or extract from sim if possible?
        # DecisionEngine is complex, sim runs it.
        # Let's say confidence is N/A or default 0.5 for scanner purposes right now
        confidence = 0.5
        
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
            safe_print(f" [skipped: error - {str(e)[:50]}]")
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
        safe_print(f"\nScanning {len(symbols)} stocks...")
    
    results = []
    for i, symbol in enumerate(symbols, 1):
        if verbose:
            safe_print(f"  [{i:2d}/{len(symbols)}] Evaluating {symbol:6s}...", end="", flush=True)
        
        score_data = score_stock(symbol, interval_seconds, cap_per_stock, verbose=verbose)
        
        if score_data:
            results.append(score_data)
            if verbose:
                exp_daily = score_data["expected_daily"]
                safe_print(f" ${exp_daily:7.2f}/day (score: {score_data['score']:7.2f})")
        else:
            if verbose:
                safe_print(" [skipped]")
    
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

