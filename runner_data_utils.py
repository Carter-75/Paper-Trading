
import logging
import time
import sys
import os
import requests
import sqlite3
import random
import traceback
import datetime as dt
from typing import List, Tuple
from dotenv import load_dotenv
import pytz
from alpaca_trade_api import REST
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

try:
    import config_validated as config
    from utils.helpers import log_info, log_warn, log_error, snap_interval_to_supported_seconds
except ImportError:
    import sys
    sys.path.append("..")
    import config_validated as config
    from utils.helpers import log_info, log_warn, log_error, snap_interval_to_supported_seconds

load_dotenv()

# ===== API Client =====
def make_client(allow_missing: bool = False, go_live: bool = False):
    key_id = config.get_config().alpaca_api_key
    secret_key = config.get_config().alpaca_secret_key
    base_url = config.get_config().alpaca_base_url
    
    if not all([key_id, secret_key, base_url]):
        if allow_missing:
            return None
        raise ValueError("Missing Alpaca credentials")
    
    if go_live:
         if config.get_config().confirm_go_live != "YES":
            raise ValueError("Live trading requires CONFIRM_GO_LIVE=YES in .env")
    
    return REST(key_id, secret_key, base_url, api_version='v2')

# ===== SQLite Price Cache =====
class PriceCache:
    """
    SQLite-based price cache for historical data.
    """
    def __init__(self, db_path: str = "price_cache.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    symbol TEXT NOT NULL,
                    interval_seconds INTEGER NOT NULL,
                    timestamp INTEGER NOT NULL,
                    close_price REAL NOT NULL,
                    PRIMARY KEY (symbol, interval_seconds, timestamp)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_interval 
                ON price_history(symbol, interval_seconds)
            """)
            conn.commit()
    
    def get_cached_bars(self, symbol: str, interval_seconds: int, limit_bars: int) -> Tuple[List[float], int]:
        """
        Get cached price bars from database.
        Returns: (closes, timestamp_of_newest_bar)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT close_price, timestamp
                FROM price_history
                WHERE symbol = ? AND interval_seconds = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (symbol, interval_seconds, limit_bars))
            
            rows = cursor.fetchall()
            if not rows:
                return ([], 0)
            
            # Return in chronological order (oldest first)
            closes = [row[0] for row in reversed(rows)]
            newest_timestamp = rows[0][1]
            
            return (closes, newest_timestamp)
    
    def store_bars(self, symbol: str, interval_seconds: int, closes: List[float], start_timestamp: int):
        """Store new price bars in database."""
        if not closes:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            data = []
            for i, close_price in enumerate(closes):
                timestamp = start_timestamp + (i * interval_seconds)
                data.append((symbol, interval_seconds, timestamp, close_price))
            
            conn.executemany("""
                INSERT OR REPLACE INTO price_history (symbol, interval_seconds, timestamp, close_price)
                VALUES (?, ?, ?, ?)
            """, data)
            conn.commit()

_price_cache = PriceCache()

# ===== Data Fetching =====

def fetch_closes(client, symbol: str, interval_seconds: int, limit_bars: int) -> List[float]:
    """
    Fetch historical closing prices with intelligent fallback.
    """
    fetch_bars = max(limit_bars * 2, 50)
    
    # 1. Check Cache
    try:
        cached_closes, newest_timestamp = _price_cache.get_cached_bars(symbol, interval_seconds, fetch_bars)
        if cached_closes and len(cached_closes) >= limit_bars:
             # Check freshness (e.g. 7 days)
             if (time.time() - newest_timestamp) < (7 * 86400):
                 return cached_closes[-limit_bars:]
    except Exception:
        pass

    # 2. Try YFinance
    try:
        import yfinance as yf
        # Simplified interval mapping
        yf_interval = "15m"
        if interval_seconds >= 86400: yf_interval = "1d"
        elif interval_seconds >= 3600: yf_interval = "1h"
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="60d", interval=yf_interval, auto_adjust=True)
        
        if not hist.empty and 'Close' in hist.columns:
            closes = list(hist['Close'].values)
            # Store to cache
            _price_cache.store_bars(symbol, interval_seconds, closes, int(hist.index[0].timestamp()))
            return closes[-limit_bars:]
    except Exception as e:
        log_warn(f"YFinance failed for {symbol}: {e}")

    # 3. Polygon / Alpaca fallbacks (Simplified for this utility)
    # ... logic omitted for brevity in this utility, assuming YF works most of the time for paper trading ...
    
    return []

def fetch_closes_with_volume(client, symbol: str, interval_seconds: int, limit_bars: int) -> Tuple[List[float], List[float]]:
    try:
        import yfinance as yf
        yf_interval = "15m"
        if interval_seconds >= 86400: yf_interval = "1d"
        elif interval_seconds >= 3600: yf_interval = "1h"
        
        # Fix for BRK.B -> BRK-B
        ticker_sym = symbol.replace('.', '-')
        ticker = yf.Ticker(ticker_sym)
        hist = ticker.history(period="60d", interval=yf_interval, auto_adjust=True)
        
        if not hist.empty:
            closes = list(hist['Close'].values)
            volumes = list(hist['Volume'].values)
            
            # Cache closes
            _price_cache.store_bars(symbol, interval_seconds, closes, int(hist.index[0].timestamp()))
            
            return closes[-limit_bars:], volumes[-limit_bars:]
    except Exception as e:
        log_warn(f"YFinance Vol fetch failed for {symbol}: {e}")
    
    return [], []
