#!/usr/bin/env python3
"""
Unified Trading Bot - Single or Multi-Stock Portfolio

Usage:
  # Single stock (auto-sets max-stocks to 1 when using -s)
  python runner.py -t 0.25 -s AAPL -m 100
  
  # Multi-stock auto (bot picks best 15 stocks - default)
  python runner.py -t 0.25 -m 1500
  
  # Multi-stock with forced picks
  python runner.py -t 0.25 -m 1500 --stocks TSLA AAPL --max-stocks 10
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import datetime as dt
import uuid
from typing import List, Optional, Tuple, Dict
from dotenv import load_dotenv

import pytz
import requests
import sqlite3
from datetime import datetime, timedelta
from alpaca_trade_api import REST
from alpaca_trade_api.rest import APIError
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

import config
from portfolio_manager import PortfolioManager

# Add ML imports
try:
    from ml_predictor import get_ml_predictor
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

load_dotenv()

# Persistent session for yfinance (best practice: reuse connections, faster + less blocks)
_yfinance_session = None
_last_yfinance_request = 0  # Track last request time for rate limiting

def get_yf_session():
    """Get or create a persistent session for yfinance with proper headers"""
    global _yfinance_session
    if _yfinance_session is None:
        import requests
        _yfinance_session = requests.Session()
        # User agent rotation to avoid blocks (best practice)
        _yfinance_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    return _yfinance_session

def yfinance_rate_limit():
    """Enforce rate limit for yfinance (1 request per 0.5 seconds = 120/minute)"""
    global _last_yfinance_request
    now = time.time()
    elapsed = now - _last_yfinance_request
    if elapsed < 0.5:  # Minimum 0.5s between requests
        time.sleep(0.5 - elapsed)
    _last_yfinance_request = time.time()

# Idempotency: Track orders submitted this cycle to prevent duplicates
_order_ids_submitted_this_cycle = set()

# Drawdown tracking
_portfolio_peak_value = 0.0
_drawdown_protection_triggered = False

# ===== Logging Setup =====
LOG = logging.getLogger("paper_trading_bot")
LOG.setLevel(logging.INFO)
LOG.propagate = False  # Prevent duplicate logging from parent loggers

# Only add handler if not already added (prevents duplicates on module re-import)
if not LOG.handlers:
    # Use UTF-8 encoding to support emoji characters on Windows
    import io
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    _console = logging.StreamHandler(utf8_stdout)
    _console.setLevel(logging.INFO)
    _console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s"))
    LOG.addHandler(_console)

FILE_LOG_PATH = config.LOG_PATH
DISABLE_FILE_LOG = os.getenv("BOT_TEE_LOG", "") in ("1", "true", "True")
SCHEDULED_TASK_MODE = os.getenv("SCHEDULED_TASK_MODE", "0") in ("1", "true", "True")

if not os.path.exists(FILE_LOG_PATH):
    open(FILE_LOG_PATH, "a").close()

def strip_emojis(text: str) -> str:
    """Remove emoji characters that cause encoding issues in log files."""
    # Common emojis used in the bot
    emoji_map = {
        '✅': '[ON]',
        '❌': '[OFF]',
        '⚠️': '[WARN]',
        '⏸️': '[PAUSED]',
        '🎯': '[TARGET]',
        '📊': '[STATS]',
    }
    result = text
    for emoji, replacement in emoji_map.items():
        result = result.replace(emoji, replacement)
    return result

def append_to_log_line(line: str):
    if DISABLE_FILE_LOG:
        return
    # Strip emojis to prevent encoding issues in log file
    clean_line = strip_emojis(line)
    attempts = 3
    delay = 0.2
    for i in range(attempts):
        try:
            enforce_log_max_lines(250)
            with open(FILE_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(clean_line + "\n")
            return
        except PermissionError:
            if i < attempts - 1:
                time.sleep(delay)
                delay *= 2

def enforce_log_max_lines(max_lines: int = 100):
    """Truncate log file to max_lines, preserving INIT line at top."""
    try:
        if not os.path.exists(FILE_LOG_PATH):
            return
        
        # Read with retry logic for file access
        lines = None
        for attempt in range(3):
            try:
                with open(FILE_LOG_PATH, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = fh.readlines()
                break
            except (PermissionError, OSError) as e:
                if attempt < 2:
                    time.sleep(0.2 * (attempt + 1))  # Increasing delay
                else:
                    # If we can't read after retries, log the issue but don't crash
                    print(f"Warning: Could not read log file for truncation: {e}", file=sys.stderr)
                    return
        
        if lines is None:
            return
        
        # Keep only non-empty lines
        lines = [ln for ln in lines if ln.strip()]
        
        # If under limit, no truncation needed
        # Note: We're called BEFORE the new line is written, so check if current + 1 would exceed
        if len(lines) < max_lines:
            return
        
        # Find INIT line (should be first line with "INIT " prefix)
        init_line = None
        for ln in lines:
            if ln.startswith("INIT "):
                init_line = ln
                break
        
        # Keep last N lines
        kept = lines[-max_lines:]
        
        # Ensure INIT line is at the top if not already included
        if init_line and init_line not in kept:
            kept = [init_line] + kept[1:]  # Replace first line with INIT
        
        # Write with retry - use temp file for atomic write
        import tempfile
        temp_path = None
        try:
            # Write to temp file first
            fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(FILE_LOG_PATH), text=True)
            with os.fdopen(fd, 'w', encoding='utf-8') as temp_fh:
                temp_fh.writelines(kept)
            
            # Atomic replace (on Windows, need to remove dest first)
            if os.path.exists(FILE_LOG_PATH):
                try:
                    os.remove(FILE_LOG_PATH)
                except (PermissionError, OSError):
                    # If file is locked, try waiting and retry
                    time.sleep(0.5)
                    try:
                        os.remove(FILE_LOG_PATH)
                    except (PermissionError, OSError) as e:
                        # Clean up temp file and give up
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)
                        print(f"Warning: Could not truncate log (file locked): {e}", file=sys.stderr)
                        return
            
            # Move temp file to actual log
            os.rename(temp_path, FILE_LOG_PATH)
            temp_path = None  # Successfully moved, no cleanup needed
            
        except Exception as e:
            # Clean up temp file if it exists
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            print(f"Warning: Log truncation failed: {e}", file=sys.stderr)
            
    except Exception as e:
        # Last resort - log to stderr but don't crash
        print(f"Warning: Unexpected error in log truncation: {e}", file=sys.stderr)

def log_info(msg: str):
    LOG.info(msg)
    append_to_log_line(msg)

def log_warn(msg: str):
    LOG.warning(msg)
    append_to_log_line(f"WARN: {msg}")

def log_error(msg: str):
    LOG.error(msg)
    append_to_log_line(f"ERROR: {msg}")

# ===== API Client =====
def make_client(allow_missing: bool = False, go_live: bool = False):
    key_id = config.ALPACA_API_KEY
    secret_key = config.ALPACA_SECRET_KEY
    base_url = config.ALPACA_BASE_URL
    
    if not all([key_id, secret_key, base_url]):
        if allow_missing:
            return None
        raise ValueError("Missing Alpaca credentials")
    
    if go_live:
        confirm = os.getenv("CONFIRM_GO_LIVE", "NO")
        if confirm != "YES":
            raise ValueError("Live trading requires CONFIRM_GO_LIVE=YES in .env")
    
    return REST(key_id, secret_key, base_url, api_version='v2')

# ===== Polygon Rate Limiting =====
class PolygonRateLimiter:
    """Enforces Polygon free tier limit: 5 calls per minute with exponential backoff"""
    def __init__(self, calls_per_minute=5):
        self.calls = []
        self.limit = calls_per_minute
    
    def wait_if_needed(self):
        """Block if rate limit would be exceeded"""
        now = time.time()
        # Remove calls older than 60 seconds
        self.calls = [t for t in self.calls if now - t < 60]
        
        if len(self.calls) >= self.limit:
            # Calculate sleep time
            sleep_time = 60 - (now - self.calls[0]) + 0.1  # +0.1s buffer
            if sleep_time > 0:
                log_info(f"Polygon rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            # Re-clean after sleep
            now = time.time()
            self.calls = [t for t in self.calls if now - t < 60]
        
        self.calls.append(now)

# Global instance
_polygon_rate_limiter = PolygonRateLimiter(calls_per_minute=5)

# ===== SQLite Price Cache (Item 23: 80% fewer API calls!) =====
class PriceCache:
    """
    SQLite-based price cache for historical data.
    
    Benefits:
    - 80% fewer API calls (only fetch new bars)
    - Instant backtests after first run
    - Persists across bot restarts
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
        
        Returns:
            (closes, timestamp_of_newest_bar)
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
        """
        Store new price bars in database.
        
        Args:
            symbol: Stock symbol
            interval_seconds: Bar interval
            closes: List of closing prices
            start_timestamp: Unix timestamp of first bar
        """
        if not closes:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            # Prepare data
            data = []
            for i, close_price in enumerate(closes):
                timestamp = start_timestamp + (i * interval_seconds)
                data.append((symbol, interval_seconds, timestamp, close_price))
            
            # Insert or replace (handles duplicates)
            conn.executemany("""
                INSERT OR REPLACE INTO price_history (symbol, interval_seconds, timestamp, close_price)
                VALUES (?, ?, ?, ?)
            """, data)
            conn.commit()
    
    def clean_old_data(self, days_to_keep: int = 180):
        """Remove data older than N days to keep database small."""
        cutoff_timestamp = int(time.time()) - (days_to_keep * 86400)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM price_history WHERE timestamp < ?", (cutoff_timestamp,))
            conn.commit()

# Global price cache instance
_price_cache = PriceCache()

# ===== Market Data =====
def snap_interval_to_supported_seconds(seconds: int) -> int:
    if seconds < 60:
        return 60
    elif seconds < 300:
        return 60
    elif seconds < 900:
        return 300
    elif seconds < 3600:
        return 900
    elif seconds < 14400:
        return 3600
    else:
        return 14400

def fetch_closes(client, symbol: str, interval_seconds: int, limit_bars: int) -> List[float]:
    """
    Fetch historical closing prices with intelligent fallback.
    Best Practice: Requests 2-3x limit_bars for technical indicator stability.
    
    Fallback order: Cache → yfinance → Polygon → Alpaca
    
    Technical requirements:
    - 21-period MA needs 42-50+ bars for reliability
    - RSI needs 30+ bars for stable values
    - More data = better indicator accuracy
    """
    # Request extra bars for technical indicator warm-up (best practice: 2-3x requested)
    fetch_bars = max(limit_bars * 3, 100)  # Minimum 100 bars, preferably 3x requested
    
    # Check cache first (80% faster!)
    try:
        cached_closes, newest_timestamp = _price_cache.get_cached_bars(symbol, interval_seconds, fetch_bars)
        
        # If we have enough cached data and it's recent, use it
        if cached_closes and len(cached_closes) >= limit_bars:
            # Check if data is fresh enough (within last 24 hours)
            now_timestamp = int(time.time())
            age_hours = (now_timestamp - newest_timestamp) / 3600
            
            if age_hours < 24:
                # Cache hit! 80% faster than API call
                return cached_closes[-limit_bars:]
    except:
        pass  # If cache fails, fall through to API fetch
    
    # Try yfinance first (FREE, unlimited, ALWAYS use this for backtesting)
    # Best practice: Retry with exponential backoff for 100% reliability
    
    # Rate limit yfinance requests (0.5s between stocks = 120 stocks/minute max)
    yfinance_rate_limit()
    
    import yfinance as yf
    from datetime import datetime, timedelta
    import pytz
    
    # Smart mapping: Find CLOSEST yfinance interval that's <= trading interval
    # Available: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d
    MARKET_HOURS_SECONDS = 23400  # 6.5 hours
    
    # Map interval_seconds to closest yfinance interval
    if interval_seconds >= MARKET_HOURS_SECONDS:
        yf_interval = "1d"
        days = 365
    elif interval_seconds >= 5400:
        yf_interval = "90m"
        days = 59
    elif interval_seconds >= 3600:
        yf_interval = "1h"
        days = 59
    elif interval_seconds >= 1800:
        yf_interval = "30m"
        days = 59
    elif interval_seconds >= 900:
        yf_interval = "15m"
        days = 59
    elif interval_seconds >= 300:
        yf_interval = "5m"
        days = 59
    elif interval_seconds >= 120:
        yf_interval = "2m"
        days = 59
    else:
        yf_interval = "1m"
        days = 7
    
    end = datetime.now(pytz.UTC)
    start = end - timedelta(days=int(days))
    
    # RETRY LOGIC: Make yfinance bulletproof (3 attempts with backoff)
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Use persistent session for faster/more reliable connections
            session = get_yf_session()
            ticker = yf.Ticker(symbol, session=session)
            # Best practice: auto_adjust=True handles stock splits & dividends
            hist = ticker.history(start=start, end=end, interval=yf_interval, auto_adjust=True, timeout=10)
            
            if hist.empty or 'Close' not in hist.columns:
                raise Exception(f"No data returned (interval={yf_interval}, days={days})")
            
            # Data validation (best practice: check for quality issues)
            if hist['Close'].isnull().any():
                # Fill gaps with forward fill (common practice)
                hist['Close'] = hist['Close'].ffill()  # Newer pandas syntax
            
            closes = list(hist['Close'].values)
            
            # Return most recent bars (we fetched extra for indicator stability)
            if len(closes) >= limit_bars:
                # Cache the full dataset (not just what we need)
                try:
                    start_ts = int((end - timedelta(days=days)).timestamp())
                    _price_cache.store_bars(symbol, interval_seconds, closes, start_ts)
                except:
                    pass  # Don't fail if caching fails
                
                # SUCCESS! Return data
                return closes[-limit_bars:]
            else:
                raise Exception(f"Insufficient bars: got {len(closes)}/{limit_bars} (interval={yf_interval}, {days} days)")
                
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                backoff = 2 ** attempt
                log_warn(f"yfinance attempt {attempt+1}/{max_retries} failed for {symbol}: {e}, retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                # All retries exhausted
                log_warn(f"yfinance FAILED after {max_retries} attempts for {symbol}: {last_error}")
    
    # Fallback to Polygon (YOU HAVE API KEY - better than Alpaca for historical data)
    try:
        polygon_key = config.POLYGON_API_KEY
        if not polygon_key:
            raise Exception("No Polygon API key")
        
        # Enforce rate limit BEFORE making call
        _polygon_rate_limiter.wait_if_needed()
        
        snap = snap_interval_to_supported_seconds(interval_seconds)
        multiplier = 1
        timespan = "minute"
        
        if snap == 60:
            multiplier, timespan = 1, "minute"
        elif snap == 300:
            multiplier, timespan = 5, "minute"
        elif snap == 900:
            multiplier, timespan = 15, "minute"
        elif snap == 3600:
            multiplier, timespan = 1, "hour"
        else:
            multiplier, timespan = 4, "hour"
        
        end_date = dt.datetime.now(pytz.UTC)
        start_date = end_date - dt.timedelta(days=365)  # Get 1 year of data
        
        url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
               f"{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/"
               f"{end_date.strftime('%Y-%m-%d')}")
        
        # Retry with exponential backoff for 429 errors (best practice)
        max_retries = 3
        for attempt in range(max_retries):
            resp = requests.get(url, params={"apiKey": polygon_key, "limit": fetch_bars}, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get("results"):
                    closes = [float(r["c"]) for r in data["results"]]
                    if len(closes) >= limit_bars:
                        return closes[-limit_bars:]
                    else:
                        raise Exception(f"Polygon returned only {len(closes)}/{limit_bars} bars")
                break
            elif resp.status_code == 429:  # Rate limited
                if attempt < max_retries - 1:
                    backoff = (2 ** attempt) + random.uniform(0, 1)  # Exponential + jitter
                    log_warn(f"Polygon 429 rate limit, retry {attempt+1}/{max_retries} in {backoff:.1f}s")
                    time.sleep(backoff)
                else:
                    raise Exception(f"Polygon rate limit: max retries exceeded")
            else:
                raise Exception(f"Polygon error {resp.status_code}")
                
    except Exception as e:
        log_warn(f"Polygon failed for {symbol}: {e}")
    
    # Final fallback to Alpaca (limited historical data, but better than nothing)
    try:
        from datetime import datetime, timedelta
        snap = snap_interval_to_supported_seconds(interval_seconds)
        
        if snap == 60:
            tf = TimeFrame(1, TimeFrameUnit.Minute)
            days_back = 7  # 1min bars only available for ~7 days
        elif snap == 300:
            tf = TimeFrame(5, TimeFrameUnit.Minute)
            days_back = 30  # 5min bars available for ~30 days
        elif snap == 900:
            tf = TimeFrame(15, TimeFrameUnit.Minute)
            days_back = 60  # 15min bars available for ~60 days
        elif snap == 3600:
            tf = TimeFrame(1, TimeFrameUnit.Hour)
            days_back = 180  # 1hr bars available for ~180 days
        else:
            tf = TimeFrame(4, TimeFrameUnit.Hour)
            days_back = 365  # 4hr bars available for ~1 year
        
        # Calculate start date to get enough historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for Alpaca API (YYYY-MM-DD format only, no time)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Try to get data from Alpaca with date range
        bars = client.get_bars(
            symbol, 
            tf, 
            start=start_str,
            end=end_str,
            limit=None  # Get all bars in range
        ).df
        
        if not bars.empty:
            closes = list(bars['close'].values)
            # Return most recent bars
            result = closes[-limit_bars:] if len(closes) > limit_bars else closes
            if len(result) > 0:
                return result
    except Exception as e:
        log_warn(f"Alpaca failed for {symbol}: {e}")
    
    # All data sources failed
    log_warn(f"ALL DATA SOURCES FAILED for {symbol} - no historical data available")
    return []

def fetch_closes_with_volume(client, symbol: str, interval_seconds: int, 
                             limit_bars: int) -> Tuple[List[float], List[float]]:
    """
    Fetch both closing prices and volume data.
    Returns (closes, volumes) or ([], []) if failed.
    """
    try:
        import yfinance as yf
        
        # Smart mapping: Same as fetch_closes (keep consistent!)
        MARKET_HOURS_SECONDS = 23400  # 6.5 hours
        
        if interval_seconds >= MARKET_HOURS_SECONDS:
            yf_interval = "1d"
            days = 365
        elif interval_seconds >= 5400:
            yf_interval = "90m"
            days = 59
        elif interval_seconds >= 3600:
            yf_interval = "1h"
            days = 59
        elif interval_seconds >= 1800:
            yf_interval = "30m"
            days = 59
        elif interval_seconds >= 900:
            yf_interval = "15m"
            days = 59
        elif interval_seconds >= 300:
            yf_interval = "5m"
            days = 59
        elif interval_seconds >= 120:
            yf_interval = "2m"
            days = 59
        else:
            yf_interval = "1m"
            days = 7
        
        from datetime import datetime, timedelta
        import pytz
        end = datetime.now(pytz.UTC)
        start = end - timedelta(days=int(days))
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start, end=end, interval=yf_interval)
        
        if not hist.empty and 'Close' in hist.columns and 'Volume' in hist.columns:
            closes = list(hist['Close'].values)
            volumes = list(hist['Volume'].values)
            
            # Return most recent bars
            closes = closes[-limit_bars:] if len(closes) > limit_bars else closes
            volumes = volumes[-limit_bars:] if len(volumes) > limit_bars else volumes
            
            if len(closes) > 0 and len(volumes) > 0:
                return (closes, volumes)
    except Exception as e:
        log_warn(f"Volume fetch failed for {symbol}: {e}")
    
    # Fallback: return closes only (no volume)
    closes = fetch_closes(client, symbol, interval_seconds, limit_bars)
    volumes = [1.0] * len(closes)  # Dummy volumes
    return (closes, volumes)

# ===== Trading Logic =====
def sma(closes: List[float], window: int) -> float:
    if len(closes) < window:
        return closes[-1] if closes else 0.0
    return sum(closes[-window:]) / window

def decide_action(closes: List[float], short_w: int, long_w: int) -> str:
    if len(closes) < max(short_w, long_w):
        return "hold"
    short_ma = sma(closes, short_w)
    long_ma = sma(closes, long_w)
    
    if short_ma > long_ma * 1.002:
        return "buy"
    elif short_ma < long_ma * 0.998:
        return "sell"
    return "hold"

def decide_action_multi_timeframe(client, symbol: str, base_interval: int, 
                                  short_w: int, long_w: int) -> Tuple[str, Dict]:
    """
    Check multiple timeframes for signal confirmation.
    Returns (action, details_dict)
    
    Strategy: Trade if MIN_AGREEMENT timeframes agree (default: 1 for flexibility)
    Timeframes: 1x (short), 3x (medium), 5x (long) base interval
    """
    timeframes = {
        'short': base_interval,
        'medium': base_interval * 3,
        'long': base_interval * 5
    }
    
    signals = {}
    
    for tf_name, tf_seconds in timeframes.items():
        closes = fetch_closes(client, symbol, tf_seconds, long_w + 10)
        if closes:
            action = decide_action(closes, short_w, long_w)
            signals[tf_name] = action
        else:
            signals[tf_name] = "hold"  # No data = no trade
    
    # Count votes
    buy_votes = sum(1 for s in signals.values() if s == "buy")
    sell_votes = sum(1 for s in signals.values() if s == "sell")
    
    # Use configured minimum agreement (default: 2, but can be set to 1 for less strict)
    min_agreement = config.MULTI_TIMEFRAME_MIN_AGREEMENT
    if buy_votes >= min_agreement:
        return ("buy", signals)
    elif sell_votes >= min_agreement:
        return ("sell", signals)
    else:
        return ("hold", signals)

def compute_confidence(closes: List[float]) -> float:
    if len(closes) < config.LONG_WINDOW:
        return 0.0
    short_ma = sma(closes, config.SHORT_WINDOW)
    long_ma = sma(closes, config.LONG_WINDOW)
    base_conf = abs((short_ma / long_ma) - 1.0)
    
    # Momentum boost
    if len(closes) >= 5:
        recent_change = (closes[-1] - closes[-5]) / closes[-5]
        signal = 1 if short_ma > long_ma else -1
        momentum_direction = 1 if recent_change > 0 else -1
        if signal == momentum_direction:
            base_conf *= 1.2
    
    return base_conf

def pct_stddev(closes: List[float]) -> float:
    if not closes:
        return 0.0
    mean = sum(closes) / len(closes)
    variance = sum((x - mean) ** 2 for x in closes) / len(closes)
    stddev = math.sqrt(variance)
    return stddev / mean if mean != 0 else 0.0

def calculate_atr(closes: List[float], period: int = 14) -> float:
    """
    Calculate Average True Range (ATR) - a volatility indicator.
    
    ATR measures market volatility by decomposing the entire range of price movement.
    Used for dynamic stop-loss and take-profit calculation.
    
    Args:
        closes: List of closing prices
        period: Lookback period (default: 14)
    
    Returns:
        ATR as a percentage of current price
    """
    if len(closes) < period + 1:
        # Not enough data, return 1% default
        return 0.01
    
    # Calculate True Range for each period
    true_ranges = []
    for i in range(1, len(closes)):
        high_low = abs(closes[i] - closes[i-1])  # Simplified: use close-to-close
        true_ranges.append(high_low)
    
    # Average True Range = average of last N true ranges
    if len(true_ranges) >= period:
        atr = sum(true_ranges[-period:]) / period
        # Convert to percentage
        current_price = closes[-1]
        atr_pct = atr / current_price if current_price > 0 else 0.01
        return atr_pct
    else:
        return 0.01


def volume_weighted_volatility(closes: List[float], volumes: List[float]) -> float:
    """
    Calculate volatility weighted by volume.
    High-volume price moves are more significant than low-volume moves.
    
    Returns:
        Volume-weighted standard deviation as a percentage (0.0-1.0 range)
    """
    if len(closes) < 2 or len(volumes) < 2 or len(closes) != len(volumes):
        return pct_stddev(closes)  # Fallback to regular volatility
    
    # Calculate returns (percent change)
    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes)) if closes[i-1] != 0]
    volumes_aligned = volumes[1:len(returns)+1]  # Align with returns
    
    if len(returns) == 0 or sum(volumes_aligned) == 0:
        return pct_stddev(closes)  # Fallback if no volume
    
    # Volume-weighted mean return
    total_volume = sum(volumes_aligned)
    vw_mean = sum(r * v for r, v in zip(returns, volumes_aligned)) / total_volume
    
    # Volume-weighted variance
    vw_variance = sum(((r - vw_mean) ** 2) * v for r, v in zip(returns, volumes_aligned)) / total_volume
    
    # Volume-weighted standard deviation (as ratio, not percentage)
    vw_stddev = vw_variance ** 0.5
    
    return vw_stddev

def calculate_bollinger_bands(closes: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
    """
    Calculate Bollinger Bands for overbought/oversold detection.
    
    BB = SMA ± (StdDev × multiplier)
    
    Returns:
        (upper_band, middle_band, lower_band)
        
    Interpretation:
        - Price > Upper Band = Overbought (don't buy, consider selling)
        - Price < Lower Band = Oversold (good buy opportunity)
        - Price near Middle Band = Neutral
        - Bands narrow = Low volatility (breakout coming)
        - Bands wide = High volatility (trending)
    """
    if len(closes) < period:
        return (0.0, 0.0, 0.0)  # Not enough data
    
    # Calculate middle band (SMA)
    recent_closes = closes[-period:]
    middle_band = sum(recent_closes) / len(recent_closes)
    
    # Calculate standard deviation
    variance = sum((x - middle_band) ** 2 for x in recent_closes) / len(recent_closes)
    std = variance ** 0.5
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    
    return (upper_band, middle_band, lower_band)


def calculate_macd(closes: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[float, float, float]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    MACD = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD, signal_period)
    Histogram = MACD - Signal Line
    
    Returns:
        (macd_value, signal_line, histogram)
        
    Interpretation:
        - MACD > Signal = Bullish (buy signal)
        - MACD < Signal = Bearish (sell signal)
        - Histogram > 0 = Increasing momentum
        - Histogram < 0 = Decreasing momentum
    """
    if len(closes) < slow_period + signal_period:
        return (0.0, 0.0, 0.0)  # Not enough data
    
    # Calculate EMAs (Exponential Moving Averages)
    def calculate_ema(data: List[float], period: int) -> float:
        if len(data) < period:
            return sum(data) / len(data)  # Fallback to SMA
        
        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period  # Start with SMA
        
        for price in data[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    # Calculate MACD components
    fast_ema = calculate_ema(closes, fast_period)
    slow_ema = calculate_ema(closes, slow_period)
    macd_value = fast_ema - slow_ema
    
    # Calculate signal line (EMA of MACD)
    # For simplicity, use last few MACD values
    macd_history = []
    for i in range(max(slow_period, len(closes) - signal_period), len(closes)):
        if i < slow_period:
            continue
        segment = closes[:i+1]
        f_ema = calculate_ema(segment, fast_period)
        s_ema = calculate_ema(segment, slow_period)
        macd_history.append(f_ema - s_ema)
    
    if len(macd_history) >= signal_period:
        signal_line = calculate_ema(macd_history, signal_period)
    else:
        signal_line = sum(macd_history) / len(macd_history) if macd_history else 0.0
    
    histogram = macd_value - signal_line
    
    return (macd_value, signal_line, histogram)


def compute_rsi(closes: List[float], period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI) to identify overbought/oversold conditions.
    Returns value between 0-100.
    - RSI > 70 = Overbought (don't buy)
    - RSI < 30 = Oversold (don't sell)
    - RSI 40-60 = Neutral
    """
    if len(closes) < period + 1:
        return 50.0  # Neutral if insufficient data
    
    gains = []
    losses = []
    
    # Calculate price changes
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    # Calculate average gains and losses
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    # Handle edge case
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def check_volume_confirmation(volumes: List[float], lookback: int = 20) -> Tuple[bool, float]:
    """
    Check if recent volume confirms the move.
    Returns (is_strong, volume_ratio)
    
    - volume_ratio > 1.5 = Strong move (good for buying)
    - volume_ratio < 0.8 = Weak move (caution)
    """
    if len(volumes) < lookback + 5:
        return (True, 1.0)  # Not enough data, assume OK
    
    # Recent volume (last 5 bars)
    recent_volume = sum(volumes[-5:]) / 5
    
    # Average volume (lookback period)
    avg_volume = sum(volumes[-lookback:-5]) / (lookback - 5)
    
    if avg_volume == 0:
        return (True, 1.0)
    
    volume_ratio = recent_volume / avg_volume
    
    # Strong if 50% above average
    is_strong = volume_ratio > 1.5
    
    return (is_strong, volume_ratio)

# ===== Position Management =====
def get_position(client, symbol: str):
    try:
        pos = client.get_position(symbol)
        return {
            "qty": float(pos.qty),
            "avg_entry_price": float(pos.avg_entry_price),
            "market_value": float(pos.market_value),
            "unrealized_pl": float(pos.unrealized_pl),
        }
    except:
        return None

def adjust_runtime_params(confidence: float, base_tp: float, base_sl: float, base_frac: float):
    c = max(0.0, min(1.0, confidence / 0.1))
    tp = base_tp * (1.0 + 0.5 * c)
    sl = base_sl * (1.0 + 0.2 * c)
    frac = base_frac * (1.0 + 0.3 * c)
    
    tp = max(config.MIN_TAKE_PROFIT_PERCENT, min(config.MAX_TAKE_PROFIT_PERCENT, tp))
    sl = max(config.MIN_STOP_LOSS_PERCENT, min(config.MAX_STOP_LOSS_PERCENT, sl))
    frac = max(config.MIN_TRADE_SIZE_FRAC, min(config.MAX_TRADE_SIZE_FRAC, frac))
    
    if config.RISKY_MODE_ENABLED:
        tp *= config.RISKY_TP_MULT
        sl *= config.RISKY_SL_MULT
        frac *= config.RISKY_SIZE_MULT  # Fixed: was RISKY_FRAC_MULT
        frac = min(frac, config.RISKY_MAX_FRAC_CAP)
    
    return tp, sl, frac

def compute_order_qty_from_remaining(current_price: float, remaining_cap: float, fraction: float) -> float:
    """Calculate quantity to buy - supports fractional shares"""
    usable = remaining_cap * fraction
    qty = usable / current_price
    # Round to 6 decimal places (standard for fractional shares)
    return round(qty, 6)

def kelly_position_size(win_rate: float, avg_win_pct: float, avg_loss_pct: float, 
                       available_capital: float, base_fraction: float) -> Tuple[float, float]:
    """
    Calculate optimal position size using Kelly Criterion.
    Returns (kelly_fraction, recommended_capital)
    
    Kelly Formula: f = (p*b - q) / b
    where p = win rate, b = win/loss ratio, q = 1-p
    
    We use Half-Kelly for safety (50% of Kelly recommendation)
    """
    
    # Safety checks
    if win_rate <= 0.5 or avg_loss_pct <= 0 or avg_win_pct <= 0:
        # Not profitable or insufficient data - use minimum
        return (base_fraction * 0.5, available_capital * base_fraction * 0.5)
    
    # Calculate win/loss ratio
    b = avg_win_pct / avg_loss_pct
    q = 1 - win_rate
    
    # Kelly formula
    kelly_fraction = (win_rate * b - q) / b
    
    # Apply safety constraints
    # 1. Use Half-Kelly (more conservative)
    safe_kelly = kelly_fraction * 0.5
    
    # 2. Clamp to reasonable range
    safe_kelly = max(config.MIN_TRADE_SIZE_FRAC, min(config.MAX_TRADE_SIZE_FRAC * 0.7, safe_kelly))
    
    # 3. Don't go below base fraction (stay conservative)
    final_fraction = max(base_fraction * 0.8, safe_kelly)
    
    recommended_capital = available_capital * final_fraction
    
    return (final_fraction, recommended_capital)

def calculate_correlation_matrix(symbols: List[str], days: int = 60) -> dict:
    """
    Calculate correlation matrix for a list of stocks.
    Shows which stocks move together (for portfolio diversification).
    
    Args:
        symbols: List of stock symbols
        days: Number of days to analyze
    
    Returns:
        dict with:
            - matrix: Dict of {(symbol1, symbol2): correlation}
            - avg_correlation: Average correlation across portfolio
            - high_correlation_pairs: List of pairs with >0.7 correlation
    """
    try:
        import yfinance as yf
        import datetime
        
        if len(symbols) < 2:
            return {
                "matrix": {},
                "avg_correlation": 0.0,
                "high_correlation_pairs": []
            }
        
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=days)
        
        # Fetch returns for all symbols
        returns_dict = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start, end=end, interval='1d')
                if not hist.empty and len(hist) >= 10:
                    returns = hist['Close'].pct_change().dropna()
                    returns_dict[symbol] = returns
            except:
                continue
        
        if len(returns_dict) < 2:
            return {
                "matrix": {},
                "avg_correlation": 0.0,
                "high_correlation_pairs": []
            }
        
        # Calculate pairwise correlations
        matrix = {}
        correlations = []
        high_correlation_pairs = []
        
        symbols_with_data = list(returns_dict.keys())
        
        for i, sym1 in enumerate(symbols_with_data):
            for j, sym2 in enumerate(symbols_with_data):
                if i >= j:  # Skip duplicate pairs and self-correlation
                    continue
                
                # Align dates
                common_dates = returns_dict[sym1].index.intersection(returns_dict[sym2].index)
                if len(common_dates) < 10:
                    continue
                
                r1 = returns_dict[sym1].loc[common_dates]
                r2 = returns_dict[sym2].loc[common_dates]
                
                # Calculate correlation
                corr = r1.corr(r2)
                
                if not (corr != corr):  # Check for NaN
                    matrix[(sym1, sym2)] = corr
                    matrix[(sym2, sym1)] = corr  # Symmetric
                    correlations.append(abs(corr))
                    
                    # Track high correlations (>0.7)
                    if abs(corr) > 0.7:
                        high_correlation_pairs.append((sym1, sym2, corr))
        
        # Self-correlations
        for sym in symbols_with_data:
            matrix[(sym, sym)] = 1.0
        
        avg_corr = sum(correlations) / len(correlations) if correlations else 0.0
        
        return {
            "matrix": matrix,
            "avg_correlation": avg_corr,
            "high_correlation_pairs": high_correlation_pairs
        }
    except:
        return {
            "matrix": {},
            "avg_correlation": 0.0,
            "high_correlation_pairs": []
        }


def calculate_correlation(closes1: List[float], closes2: List[float]) -> float:
    """
    Calculate Pearson correlation between two price series.
    Returns value between -1 and 1.
    - 1.0 = Perfect positive correlation (move together)
    - 0.0 = No correlation
    - -1.0 = Perfect negative correlation (move opposite)
    """
    if len(closes1) != len(closes2) or len(closes1) < 20:
        return 0.5  # Unknown, assume moderate
    
    # Calculate returns
    returns1 = [(closes1[i] - closes1[i-1]) / closes1[i-1] 
                for i in range(1, min(len(closes1), 50))]
    returns2 = [(closes2[i] - closes2[i-1]) / closes2[i-1] 
                for i in range(1, min(len(closes2), 50))]
    
    n = min(len(returns1), len(returns2))
    returns1 = returns1[:n]
    returns2 = returns2[:n]
    
    # Calculate means
    mean1 = sum(returns1) / n
    mean2 = sum(returns2) / n
    
    # Calculate correlation
    numerator = sum((returns1[i] - mean1) * (returns2[i] - mean2) for i in range(n))
    
    sum_sq1 = sum((r - mean1) ** 2 for r in returns1)
    sum_sq2 = sum((r - mean2) ** 2 for r in returns2)
    
    denominator = (sum_sq1 * sum_sq2) ** 0.5
    
    if denominator == 0:
        return 0.5
    
    correlation = numerator / denominator
    return max(-1.0, min(1.0, correlation))  # Clamp to [-1, 1]


def check_portfolio_correlation(client, new_symbol: str, held_symbols: List[str], 
                                interval_seconds: int) -> Tuple[bool, Dict[str, float]]:
    """
    Check if new symbol is too correlated with existing holdings.
    Returns (is_acceptable, correlation_dict)
    """
    if len(held_symbols) == 0:
        return (True, {})
    
    new_closes = fetch_closes(client, new_symbol, interval_seconds, 100)
    if len(new_closes) < 20:
        return (True, {})  # Not enough data, allow
    
    correlations = {}
    
    for held in held_symbols:
        held_closes = fetch_closes(client, held, interval_seconds, 100)
        if len(held_closes) >= 20:
            corr = calculate_correlation(new_closes, held_closes)
            correlations[held] = corr
            
            # Reject if too highly correlated
            if corr > config.MAX_CORRELATION_THRESHOLD:
                return (False, correlations)
    
    return (True, correlations)

def buy_flow(client, symbol: str, last_price: float, available_cap: float,
             confidence: float, base_frac: float, base_tp: float, base_sl: float,
             dynamic_enabled: bool = True, interval_seconds: int = None,
             total_invested: float = 0.0, max_capital: float = None):
    """
    Buy a stock using available capital.
    available_cap: How much capital is available for THIS specific trade
    total_invested: Total capital currently invested across all positions
    max_capital: Total capital available for trading
    """
    pos = get_position(client, symbol)
    if pos:
        return (False, "Position exists")
    
    if confidence < config.MIN_CONFIDENCE_TO_TRADE:
        return (False, f"Low confidence: {confidence:.4f}")
    
    # Exposure limit check - don't go all-in, keep cash for opportunities
    if max_capital and max_capital > 0:
        # Calculate the order value we're about to place
        estimated_order_value = available_cap * base_frac
        
        can_buy, adjusted_value, exp_msg = check_exposure_limit(
            total_invested, max_capital, estimated_order_value
        )
        
        if not can_buy:
            return (False, exp_msg)
        
        # Adjust available capital if needed to respect exposure limit
        if adjusted_value < estimated_order_value:
            available_cap = adjusted_value / base_frac
            log_info(f"  [EXPOSURE LIMIT] {exp_msg}")
        else:
            log_info(f"  [EXPOSURE] {exp_msg}")
    
    # Correlation check - avoid doubling up on correlated positions
    if config.ENABLE_CORRELATION_CHECK and interval_seconds:
        try:
            # Get all current positions
            positions = client.list_positions()
            if positions:
                # Fetch closes for this symbol
                my_closes = fetch_closes(client, symbol, interval_seconds, 60)
                
                if my_closes and len(my_closes) >= 30:
                    for existing_pos in positions:
                        existing_symbol = existing_pos.symbol
                        if existing_symbol == symbol:
                            continue  # Skip self
                        
                        # Fetch closes for existing position
                        other_closes = fetch_closes(client, existing_symbol, interval_seconds, 60)
                        
                        if other_closes and len(other_closes) >= 30:
                            # Calculate correlation
                            # Align to same length
                            min_len = min(len(my_closes), len(other_closes))
                            corr = calculate_correlation(my_closes[-min_len:], other_closes[-min_len:])
                            
                            # If high correlation (>0.7), reduce position or skip
                            if abs(corr) > 0.7:
                                # Reduce available capital by 50% for highly correlated stocks
                                available_cap = available_cap * 0.5
                                log_info(f"  [!] High correlation with {existing_symbol}: {corr:.2f}")
                                log_info(f"     Reducing position size by 50% (${available_cap*2:.2f} → ${available_cap:.2f})")
                            elif abs(corr) > 0.5:
                                # Moderate correlation: reduce by 25%
                                available_cap = available_cap * 0.75
                                log_info(f"  [!] Moderate correlation with {existing_symbol}: {corr:.2f}")
                                log_info(f"     Reducing position size by 25% (${available_cap/0.75:.2f} → ${available_cap:.2f})")
        except:
            pass  # Don't fail trade if correlation check fails
    
    # Profitability check
    # Use max_capital (total portfolio) for consistent profitability calculation with scanner
    # Don't use available_cap here - that's just this stock's allocation slice
    if interval_seconds:
        closes = fetch_closes(client, symbol, interval_seconds, config.LONG_WINDOW + 50)
        if closes:
            # Use max_capital if available, otherwise fall back to available_cap for backward compatibility
            profitability_cap = max_capital if max_capital else available_cap
            sim = simulate_signals_and_projection(closes, interval_seconds, override_cap_usd=profitability_cap)
            expected_daily = float(sim.get("expected_daily_usd", 0.0))
            if expected_daily < config.PROFITABILITY_MIN_EXPECTED_USD:
                return (False, f"Expected ${expected_daily:.2f}/day < ${config.PROFITABILITY_MIN_EXPECTED_USD}")
            
            # Volatility check (volume-weighted for better signal quality)
            try:
                closes_vol, volumes_vol = fetch_closes_with_volume(client, symbol, interval_seconds, config.VOLATILITY_WINDOW)
                if closes_vol and volumes_vol and len(closes_vol) == len(volumes_vol):
                    vol_pct = volume_weighted_volatility(closes_vol, volumes_vol)
                    log_info(f"  Volume-weighted volatility: {vol_pct*100:.1f}%")
                else:
                    vol_pct = pct_stddev(closes[-config.VOLATILITY_WINDOW:])
                    log_info(f"  Standard volatility: {vol_pct*100:.1f}%")
            except:
                vol_pct = pct_stddev(closes[-config.VOLATILITY_WINDOW:])
            
            if vol_pct > config.VOLATILITY_PCT_THRESHOLD:
                return (False, f"High volatility: {vol_pct*100:.1f}%")
    
    # RSI check - don't buy if overbought
    if config.RSI_ENABLED and interval_seconds:
        closes_rsi = fetch_closes(client, symbol, interval_seconds, config.RSI_PERIOD + 20)
        if closes_rsi and len(closes_rsi) >= config.RSI_PERIOD + 1:
            rsi = compute_rsi(closes_rsi, config.RSI_PERIOD)
            if rsi > config.RSI_OVERBOUGHT:
                return (False, f"Overbought: RSI={rsi:.1f} > {config.RSI_OVERBOUGHT}")
            log_info(f"  RSI: {rsi:.1f} (neutral/bullish)")
    
    # MACD check - momentum + trend confirmation
    if interval_seconds and closes:
        try:
            macd_value, signal_line, histogram = calculate_macd(closes)
            
            # MACD must be above signal line (bullish momentum)
            if macd_value < signal_line:
                return (False, f"MACD bearish: {macd_value:.4f} < {signal_line:.4f}")
            
            # Histogram should be positive (increasing momentum)
            if histogram < 0:
                return (False, f"MACD histogram negative: {histogram:.4f} (weakening momentum)")
            
            log_info(f"  MACD: {macd_value:.4f} > Signal: {signal_line:.4f} (bullish momentum)")
        except:
            pass  # Don't fail trade if MACD calculation fails
    
    # Bollinger Bands check - avoid overbought conditions
    if interval_seconds and closes:
        try:
            upper_band, middle_band, lower_band = calculate_bollinger_bands(closes, period=20, std_dev=2.0)
            current_price = closes[-1]
            
            # Don't buy if price is at or above upper band (overbought)
            if current_price >= upper_band:
                return (False, f"Bollinger Bands: Price ${current_price:.2f} >= Upper ${upper_band:.2f} (overbought)")
            
            # Calculate % position in band (0% = lower, 50% = middle, 100% = upper)
            band_width = upper_band - lower_band
            if band_width > 0:
                position_pct = ((current_price - lower_band) / band_width) * 100
                
                # Warn if price is near upper band (>80%)
                if position_pct > 80:
                    log_info(f"  [!]  Bollinger Bands: {position_pct:.0f}% (near upper band, risky)")
                else:
                    log_info(f"  Bollinger Bands: {position_pct:.0f}% (good position)")
        except:
            pass  # Don't fail trade if BB calculation fails
    
    # Volume confirmation - ensure strong buying interest
    if config.VOLUME_CONFIRMATION_ENABLED and interval_seconds:
        closes_vol, volumes_vol = fetch_closes_with_volume(client, symbol, interval_seconds, config.LONG_WINDOW + 50)
        if volumes_vol:
            is_strong, vol_ratio = check_volume_confirmation(volumes_vol)
            if vol_ratio < config.VOLUME_CONFIRMATION_THRESHOLD:
                return (False, f"Weak volume: {vol_ratio:.2f}x avg (need {config.VOLUME_CONFIRMATION_THRESHOLD}x)")
            log_info(f"  Volume: {vol_ratio:.2f}x avg ({'strong' if is_strong else 'moderate'})")
    
    # Calculate adaptive TP/SL based on ATR (volatility-adjusted)
    tp, sl, frac = base_tp, base_sl, base_frac
    
    if interval_seconds and closes:
        try:
            # Calculate ATR for dynamic stops
            atr_pct = calculate_atr(closes, period=14)
            current_vol = pct_stddev(closes[-config.VOLATILITY_WINDOW:])
            avg_vol = 0.02  # Assume 2% average volatility
            
            # Volatility ratio: current_vol / avg_vol
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            # ATR-based adaptive stops (Item 59)
            # SL = base_SL × (2 × ATR)
            # TP = base_TP × (2 × ATR)
            atr_multiplier = max(0.5, min(2.0, 2 * atr_pct / 0.02))  # Scale to ~2% baseline
            
            # Combine both methods: Use larger of ATR-based or volatility-ratio
            adaptive_multiplier = max(vol_ratio, atr_multiplier)
            
            # Apply adaptive scaling (Item 52)
            tp = base_tp * adaptive_multiplier
            sl = base_sl * adaptive_multiplier
            
            # Clamp to reasonable limits
            tp = max(config.MIN_TAKE_PROFIT_PERCENT, min(config.MAX_TAKE_PROFIT_PERCENT, tp))
            sl = max(config.MIN_STOP_LOSS_PERCENT, min(config.MAX_STOP_LOSS_PERCENT, sl))
            
            log_info(f"  Adaptive TP/SL: ATR={atr_pct*100:.2f}%, Vol Ratio={vol_ratio:.2f}x")
            log_info(f"    → TP: {base_tp:.1f}% → {tp:.1f}%, SL: {base_sl:.1f}% → {sl:.1f}%")
        except:
            pass  # Fall back to base values on error
    
    # Apply dynamic adjustment based on confidence (if enabled)
    if dynamic_enabled:
        tp, sl, frac = adjust_runtime_params(confidence, tp, sl, frac)
    
    # Apply max loss per trade constraint FIRST (hard safety limit)
    if max_capital and max_capital > 0:
        risk_limited_cap, risk_msg = calculate_max_position_size_for_risk(
            max_capital, sl, available_cap
        )
        if risk_limited_cap < available_cap:
            log_info(f"  [MAX RISK] {risk_msg}")
            available_cap = risk_limited_cap
        elif config.MAX_LOSS_PER_TRADE_ENABLED:
            log_info(f"  [RISK CHECK] {risk_msg}")
    
    # Apply dynamic position sizing multiplier based on recent performance
    dynamic_multiplier = get_dynamic_position_multiplier()
    if dynamic_multiplier != 1.0:
        original_cap = available_cap
        available_cap = available_cap * dynamic_multiplier
        log_info(f"  Dynamic Position Sizing: {dynamic_multiplier:.2f}x (${original_cap:.2f} → ${available_cap:.2f})")
        
        # Ensure we don't exceed max capital limits
        if available_cap > config.MAX_CAP_USD:
            available_cap = config.MAX_CAP_USD
            log_info(f"    Capped at max: ${config.MAX_CAP_USD:.2f}")
        
        # Re-check risk limit after dynamic adjustment
        if max_capital and max_capital > 0:
            risk_limited_cap_2, _ = calculate_max_position_size_for_risk(
                max_capital, sl, available_cap
            )
            if risk_limited_cap_2 < available_cap:
                log_info(f"  [MAX RISK] Reduced after dynamic adjustment: ${available_cap:.2f} → ${risk_limited_cap_2:.2f}")
                available_cap = risk_limited_cap_2
    
    # Calculate position size (Kelly or static)
    if config.ENABLE_KELLY_SIZING and interval_seconds:
        # Run quick simulation to get win rate
        closes_sim = fetch_closes(client, symbol, interval_seconds, config.LONG_WINDOW + 100)
        if len(closes_sim) > config.LONG_WINDOW + 10:
            sim_quick = simulate_signals_and_projection(
                closes_sim, interval_seconds, 
                override_cap_usd=available_cap,
                use_walk_forward=False  # Fast, no walk-forward
            )
            win_rate_kelly = sim_quick.get("win_rate", 0.5)
            
            # Use Kelly sizing
            kelly_frac, kelly_cap = kelly_position_size(
                win_rate_kelly,
                tp,  # avg win %
                sl,  # avg loss %
                available_cap,
                frac
            )
            
            log_info(f"  Kelly sizing: {kelly_frac:.2%} of ${available_cap:.2f} = ${kelly_cap:.2f} (win_rate={win_rate_kelly:.1%})")
            qty = compute_order_qty_from_remaining(last_price, kelly_cap, 1.0)
        else:
            # Not enough data for Kelly, use static
            qty = compute_order_qty_from_remaining(last_price, available_cap, frac)
    else:
        # Static position sizing
        qty = compute_order_qty_from_remaining(last_price, available_cap, frac)
    
    if qty < 0.001:  # Minimum fractional share
        return (False, "Insufficient capital")
    
    # Apply slippage simulation for paper trading to match live expectations
    effective_price = last_price
    if config.SIMULATE_SLIPPAGE_ENABLED and "paper" in config.ALPACA_BASE_URL.lower():
        effective_price = last_price * (1 + config.SLIPPAGE_PERCENT / 100)
        # Recalculate qty with slippage-adjusted price
        qty = compute_order_qty_from_remaining(effective_price, available_cap, frac)
        if qty < 0.001:
            return (False, "Insufficient capital after slippage")
    
    try:
        # Use effective_price for TP/SL calculations
        tp_price = effective_price * (1 + tp / 100.0)
        sl_price = effective_price * (1 - sl / 100.0)
        
        # Generate unique client order ID for idempotency
        client_order_id = f"{symbol}_BUY_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Check if already submitted this cycle (network retry protection)
        if client_order_id in _order_ids_submitted_this_cycle:
            return (False, "Order already submitted this cycle")
        
        _order_ids_submitted_this_cycle.add(client_order_id)
        
        # Check if fractional
        is_fractional = (qty % 1 != 0)
        
        if is_fractional:
            # Fractional shares: Must use 'day' order, cannot use bracket orders
            # Determine order type
            if config.USE_LIMIT_ORDERS:
                # Limit order - place slightly below market for better price
                limit_price = effective_price * (1 - config.LIMIT_ORDER_OFFSET_PERCENT / 100)
                
                # Verify order safety before submission
                is_safe, verify_msg = verify_order_safety(
                    client, symbol, 'buy', qty, limit_price, last_price
                )
                if not is_safe:
                    return (False, f"Order verification failed: {verify_msg}")
                
                order = client.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='limit',
                    time_in_force='day',
                    limit_price=round(limit_price, 2),
                    client_order_id=client_order_id
                )
                log_info(f"  Limit order @ ${limit_price:.2f} (market: ${effective_price:.2f})")
            else:
                # Market order (original behavior)
                # Verify order safety before submission
                is_safe, verify_msg = verify_order_safety(
                    client, symbol, 'buy', qty, effective_price, last_price
                )
                if not is_safe:
                    return (False, f"Order verification failed: {verify_msg}")
                
                order = client.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='day',  # Required for fractional
                    client_order_id=client_order_id
                )
            
            # Wait for fill confirmation
            if config.USE_LIMIT_ORDERS:
                max_wait_seconds = config.LIMIT_ORDER_TIMEOUT_SECONDS
                log_info(f"  Waiting up to {max_wait_seconds}s for limit order fill...")
            else:
                max_wait_seconds = 30  # Market orders fill quickly
            fill_confirmed = False
            actual_filled_qty = 0
            
            for wait_iter in range(max_wait_seconds):
                try:
                    order_status = client.get_order(order.id)
                    if order_status.status == 'filled':
                        actual_filled_qty = float(order_status.filled_qty)
                        fill_confirmed = True
                        break
                    elif order_status.status == 'partially_filled':
                        actual_filled_qty = float(order_status.filled_qty)
                        # For fractional, partial is common - accept it
                        fill_confirmed = True
                        break
                except:
                    pass
                time.sleep(1)
            
            if not fill_confirmed:
                if config.USE_LIMIT_ORDERS:
                    # Limit order didn't fill - cancel and try market order as fallback
                    try:
                        client.cancel_order(order.id)
                        log_warn(f"Limit order timeout - switching to market order")
                        
                        # Submit market order instead
                        # Verify before fallback to market
                        is_safe, verify_msg = verify_order_safety(
                            client, symbol, 'buy', qty, effective_price, last_price
                        )
                        if not is_safe:
                            log_warn(f"Fallback market order verification failed: {verify_msg}")
                            return (False, f"Fallback verification failed: {verify_msg}")
                        
                        order = client.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='buy',
                            type='market',
                            time_in_force='day',
                            client_order_id=client_order_id + "_MKT"
                        )
                        
                        # Wait for market fill (should be fast)
                        for wait_iter in range(30):
                            try:
                                order_status = client.get_order(order.id)
                                if order_status.status == 'filled':
                                    actual_filled_qty = float(order_status.filled_qty)
                                    fill_confirmed = True
                                    break
                            except:
                                pass
                            time.sleep(1)
                    except Exception as e:
                        log_warn(f"Fallback market order failed: {e}")
                
                if not fill_confirmed:
                    return (False, f"Order not filled after {max_wait_seconds}s")
            
            # Place separate TP and SL orders using ACTUAL filled quantity
            try:
                client.submit_order(
                    symbol=symbol,
                    qty=actual_filled_qty,
                    side='sell',
                    type='limit',
                    time_in_force='day',
                    limit_price=round(tp_price, 2)
                )
                client.submit_order(
                    symbol=symbol,
                    qty=actual_filled_qty,
                    side='sell',
                    type='stop',
                    time_in_force='day',
                    stop_price=round(sl_price, 2)
                )
            except:
                pass  # TP/SL orders are optional for fractional shares
            
            shares_text = f"{actual_filled_qty:.6f}".rstrip('0').rstrip('.') if actual_filled_qty < 1 else f"{actual_filled_qty:.2f}"
            return (True, f"Bought {shares_text} shares @ ${last_price:.2f} (TP:{tp:.2f}% SL:{sl:.2f}%)")
        else:
            # Whole shares: Use bracket orders with GTC
            # Verify order safety before submission
            is_safe, verify_msg = verify_order_safety(
                client, symbol, 'buy', int(qty), effective_price, last_price
            )
            if not is_safe:
                return (False, f"Order verification failed: {verify_msg}")
            
            # Use trailing stop if enabled (locks in profits as price rises)
            if config.TRAILING_STOP_PERCENT > 0:
                stop_loss_config = {'trail_percent': config.TRAILING_STOP_PERCENT}
                log_info(f"  Using trailing stop: {config.TRAILING_STOP_PERCENT}% (locks in profits)")
            else:
                stop_loss_config = {'stop_price': round(sl_price, 2)}
            
            order = client.submit_order(
                symbol=symbol,
                qty=int(qty),
                side='buy',
                type='market',
                time_in_force='gtc',
                order_class='bracket',
                take_profit={'limit_price': round(tp_price, 2)},
                stop_loss=stop_loss_config,
                client_order_id=client_order_id
            )
            
            # Wait for fill confirmation
            max_wait_seconds = 30
            fill_confirmed = False
            actual_filled_qty = 0
            
            for wait_iter in range(max_wait_seconds):
                try:
                    order_status = client.get_order(order.id)
                    if order_status.status in ['filled', 'partially_filled']:
                        actual_filled_qty = float(order_status.filled_qty)
                        fill_confirmed = True
                        if order_status.status == 'filled':
                            break
                except:
                    pass
                time.sleep(1)
            
            if not fill_confirmed:
                return (False, f"Order not filled after {max_wait_seconds}s")
            elif actual_filled_qty < qty * 0.9:  # Less than 90% filled
                log_warn(f"Partial fill: {actual_filled_qty}/{qty} shares")
                # Continue anyway - we got something
            
            shares_text = f"{actual_filled_qty:.6f}".rstrip('0').rstrip('.') if actual_filled_qty < 1 else f"{actual_filled_qty:.2f}"
            return (True, f"Bought {shares_text} shares @ ${last_price:.2f} (TP:{tp:.2f}% SL:{sl:.2f}%)")
    except Exception as e:
        return (False, f"Order failed: {e}")

def sell_flow(client, symbol: str, confidence: float = 0.0):
    pos = get_position(client, symbol)
    if not pos:
        return (False, "No position")
    
    qty = float(pos["qty"])  # FIXED: Keep fractional shares
    realized_pl = pos["unrealized_pl"]
    
    # Log expected slippage cost (can't change paper fills, but document it)
    if config.SIMULATE_SLIPPAGE_ENABLED and "paper" in config.ALPACA_BASE_URL.lower():
        slippage_loss = float(pos["market_value"]) * (config.SLIPPAGE_PERCENT / 100)
        log_info(f"Expected slippage: -${slippage_loss:.2f}")
    
    try:
        # Cancel bracket orders
        orders = client.list_orders(status='open', symbols=[symbol])
        for order in orders:
            try:
                client.cancel_order(order.id)
            except:
                pass
        
        # Generate unique client order ID for idempotency
        client_order_id = f"{symbol}_SELL_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Check if already submitted this cycle (network retry protection)
        if client_order_id in _order_ids_submitted_this_cycle:
            return (False, "Order already submitted this cycle")
        
        _order_ids_submitted_this_cycle.add(client_order_id)
        
        # Market sell (use 'day' for fractional shares, 'gtc' for whole shares)
        is_fractional = (qty % 1 != 0)
        time_in_force = 'day' if is_fractional else 'gtc'
        
        # Verify order safety before submission
        # Get current price for verification
        try:
            current_price = fetch_closes(client, symbol, 900, 1)[-1] if fetch_closes(client, symbol, 900, 1) else None
        except:
            current_price = None
        
        is_safe, verify_msg = verify_order_safety(
            client, symbol, 'sell', qty, current_price if current_price else qty * 100, current_price
        )
        if not is_safe:
            log_warn(f"Sell order verification failed: {verify_msg}")
            # For sells, we might still want to sell even if verification fails (to exit bad position)
            # Just log warning but continue
        
        order = client.submit_order(
            symbol=symbol, 
            qty=qty, 
            side='sell', 
            type='market', 
            time_in_force=time_in_force,
            client_order_id=client_order_id
        )
        
        # Wait for fill confirmation
        max_wait_seconds = 30
        fill_confirmed = False
        actual_filled_qty = 0
        
        for wait_iter in range(max_wait_seconds):
            try:
                order_status = client.get_order(order.id)
                if order_status.status in ['filled', 'partially_filled']:
                    actual_filled_qty = float(order_status.filled_qty)
                    fill_confirmed = True
                    if order_status.status == 'filled':
                        break
            except:
                pass
            time.sleep(1)
        
        if not fill_confirmed:
            return (False, f"Sell order not filled after {max_wait_seconds}s")
        elif actual_filled_qty < qty * 0.9:
            log_warn(f"Partial sell: {actual_filled_qty}/{qty} shares")
        
        # Calculate profit percentage for dynamic position sizing
        try:
            cost_basis = float(pos.get("cost_basis", 0))
            current_value = float(pos.get("market_value", 0))
            if cost_basis > 0:
                profit_pct = ((current_value - cost_basis) / cost_basis) * 100
                update_trade_history(symbol, profit_pct)
                log_info(f"Dynamic Position Sizing: Recorded {profit_pct:+.2f}% trade")
        except:
            pass
        
        # Log to PnL ledger
        try:
            ledger_path = "pnl_ledger.json"
            ledger = []
            if os.path.exists(ledger_path):
                with open(ledger_path, "r") as f:
                    ledger = json.load(f)
            
            ledger.append({
                "timestamp": dt.datetime.now(pytz.UTC).isoformat(),
                "symbol": symbol,
                "realized_pl": realized_pl
            })
            
            with open(ledger_path, "w") as f:
                json.dump(ledger, f, indent=2)
        except:
            pass
        
        return (True, f"Sold {qty} shares (P&L: ${realized_pl:+.2f})")
    except Exception as e:
        return (False, f"Sell failed: {e}")

def enforce_all_safety_checks(client, symbol: str = None, portfolio = None) -> Tuple[bool, str]:
    """
    Check all safety limits. Returns (should_shutdown, reason).
    If should_shutdown=True, bot should immediately liquidate and exit.
    """
    try:
        account = client.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        
        if last_equity <= 0:
            return (False, "")  # Can't calculate on first run
        
        # Check 1: Daily loss percentage
        daily_pnl_pct = ((equity - last_equity) / last_equity) * 100
        if daily_pnl_pct < -config.MAX_DAILY_LOSS_PERCENT:
            return (True, f"Daily loss % limit: {daily_pnl_pct:.2f}% < -{config.MAX_DAILY_LOSS_PERCENT}%")
        
        # Check 2: Daily loss absolute USD
        daily_pnl_usd = equity - last_equity
        if daily_pnl_usd < -config.DAILY_LOSS_LIMIT_USD:
            return (True, f"Daily loss $ limit: ${daily_pnl_usd:.2f} < -${config.DAILY_LOSS_LIMIT_USD}")
        
        return (False, "")  # All checks passed
        
    except Exception as e:
        log_warn(f"Safety check error: {e}")
        return (False, "")  # Don't shutdown on check failure

# Keep old name for compatibility (just call new function)
def enforce_safety(client, symbol: str):
    should_stop, reason = enforce_all_safety_checks(client, symbol)
    if should_stop:
        log_warn(reason)
        sell_flow(client, symbol)
        sys.exit(1)

def check_drawdown_protection(current_value: float) -> Tuple[bool, str]:
    """
    Check if portfolio has dropped too much from peak.
    Returns (should_continue_trading, message)
    """
    global _portfolio_peak_value, _drawdown_protection_triggered
    
    if not config.ENABLE_DRAWDOWN_PROTECTION:
        return (True, "")
    
    # Update peak
    if current_value > _portfolio_peak_value:
        _portfolio_peak_value = current_value
        _drawdown_protection_triggered = False  # Reset if recovered
    
    # Calculate drawdown
    if _portfolio_peak_value > 0:
        drawdown_pct = ((current_value - _portfolio_peak_value) / _portfolio_peak_value) * 100
        
        if drawdown_pct < -config.MAX_PORTFOLIO_DRAWDOWN_PERCENT:
            if not _drawdown_protection_triggered:
                _drawdown_protection_triggered = True
                msg = (f"[STOP] DRAWDOWN PROTECTION TRIGGERED\n"
                      f"   Portfolio down {abs(drawdown_pct):.1f}% from peak (${_portfolio_peak_value:.2f})\n"
                      f"   Current value: ${current_value:.2f}\n"
                      f"   Max allowed: {config.MAX_PORTFOLIO_DRAWDOWN_PERCENT}%\n"
                      f"   Trading STOPPED until recovery or manual override")
                log_warn(msg)
            return (False, f"Drawdown protection active: {abs(drawdown_pct):.1f}% > {config.MAX_PORTFOLIO_DRAWDOWN_PERCENT}%")
    
    return (True, "")


def check_exposure_limit(total_invested: float, max_capital: float, new_order_value: float) -> Tuple[bool, float, str]:
    """
    Check if adding a new position would exceed maximum exposure limit.
    Returns (can_buy, adjusted_order_value, message)
    
    Purpose: Don't go all-in on one idea - keep some cash for opportunities.
    Default: 75% max exposure (keep 25% in cash)
    """
    max_allowed = max_capital * (config.MAX_EXPOSURE_PCT / 100.0)
    new_total = total_invested + new_order_value
    
    if new_total > max_allowed:
        # Calculate how much we can actually buy without exceeding limit
        available_room = max_allowed - total_invested
        
        if available_room <= 0:
            msg = (f"Exposure limit reached: {total_invested:.2f}/${max_allowed:.2f} "
                  f"({config.MAX_EXPOSURE_PCT:.0f}% of ${max_capital:.2f})")
            return (False, 0.0, msg)
        else:
            # Can buy partial amount
            exposure_pct = (total_invested / max_capital) * 100
            msg = (f"Reducing order size: ${new_order_value:.2f} → ${available_room:.2f} "
                  f"(exposure: {exposure_pct:.0f}%/{config.MAX_EXPOSURE_PCT:.0f}%)")
            return (True, available_room, msg)
    
    # Within limits
    exposure_pct = (new_total / max_capital) * 100
    return (True, new_order_value, f"Exposure OK: {exposure_pct:.0f}%/{config.MAX_EXPOSURE_PCT:.0f}%")


def check_kill_switch() -> Tuple[bool, str]:
    """
    Check if kill switch file exists - emergency stop mechanism.
    Returns (should_continue, message)
    
    Purpose: Quick way to halt bot during anomalies or manual override.
    
    To activate: Create file KILL_SWITCH.flag in project directory
    To deactivate: Delete the file
    """
    if not config.KILL_SWITCH_ENABLED:
        return (True, "")
    
    if os.path.exists(config.KILL_SWITCH_FILE):
        msg = (f"[KILL SWITCH ACTIVATED]\n"
              f"   File found: {config.KILL_SWITCH_FILE}\n"
              f"   Bot will stop trading until file is removed.\n"
              f"   To resume: Delete the file and restart bot")
        return (False, msg)
    
    return (True, "")


def verify_order_safety(client, symbol: str, side: str, qty: float, price: float, 
                        last_known_price: float = None) -> Tuple[bool, str]:
    """
    Verify order safety before submission - prevent fat-finger errors and API glitches.
    Returns (is_safe, message)
    
    Checks:
    1. Price is not >10% from last known price (prevents bad fills)
    2. Order size is not >10% of average daily volume (prevents market impact)
    3. Price exists and is valid (>0)
    """
    if not config.ORDER_VERIFICATION_ENABLED:
        return (True, "")
    
    # Check 1: Price validation
    if price <= 0:
        return (False, f"Invalid price: ${price:.2f} (must be > 0)")
    
    # Check 2: Price deviation from last known
    if last_known_price and last_known_price > 0:
        price_change_pct = abs(price - last_known_price) / last_known_price * 100
        
        if price_change_pct > config.MAX_PRICE_DEVIATION_PCT:
            return (False, 
                   f"Price deviation too large: ${last_known_price:.2f} → ${price:.2f} "
                   f"({price_change_pct:.1f}% > {config.MAX_PRICE_DEVIATION_PCT}% limit). "
                   f"Possible API glitch or flash crash.")
    
    # Check 3: Order size vs average daily volume
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        avg_volume = info.get('averageVolume', 0) or info.get('averageDailyVolume10Day', 0)
        
        if avg_volume > 0:
            # Calculate order size as % of average daily volume
            order_value = qty * price
            avg_daily_value = avg_volume * price
            order_pct_of_adv = (qty / avg_volume) * 100
            
            if order_pct_of_adv > config.MAX_ORDER_SIZE_ADV_PCT:
                return (False,
                       f"Order too large: {qty:.2f} shares = {order_pct_of_adv:.1f}% of avg daily volume "
                       f"({avg_volume:,.0f} shares). Limit: {config.MAX_ORDER_SIZE_ADV_PCT}%. "
                       f"This could cause significant market impact.")
            
            # Log if order is getting large (warning at 5%)
            if order_pct_of_adv > 5.0:
                log_info(f"  [ORDER SIZE] {order_pct_of_adv:.2f}% of daily volume (acceptable but large)")
    
    except Exception as e:
        # Don't fail order if volume check fails - just log warning
        log_warn(f"Could not verify order size vs volume for {symbol}: {e}")
    
    # All checks passed
    return (True, "Order verification passed")


def calculate_max_position_size_for_risk(total_capital: float, stop_loss_pct: float, 
                                          available_capital: float) -> Tuple[float, str]:
    """
    Calculate maximum position size based on max loss per trade rule.
    Returns (max_position_value, message)
    
    Formula: max_position_size = (total_capital × max_loss_pct) / stop_loss_pct
    
    Example: $10,000 capital, 2% max loss, 1% stop loss
    → max_loss = $200
    → max_position = $200 / 0.01 = $20,000
    → But capped at available_capital
    
    Purpose: Ensure that even if stop loss hits, we can't lose >2% of capital.
    """
    if not config.MAX_LOSS_PER_TRADE_ENABLED:
        return (available_capital, "Max loss per trade check disabled")
    
    if stop_loss_pct <= 0:
        # No stop loss or invalid - use very conservative limit
        max_position = total_capital * (config.MAX_LOSS_PER_TRADE_PCT / 100)
        msg = f"No stop loss - limiting position to {config.MAX_LOSS_PER_TRADE_PCT}% of capital"
        return (min(max_position, available_capital), msg)
    
    # Calculate max loss in dollars
    max_loss_usd = total_capital * (config.MAX_LOSS_PER_TRADE_PCT / 100)
    
    # Calculate max position size that would result in this max loss at stop loss price
    # If stop loss is 1% and we want max loss of $200:
    # max_position = $200 / 0.01 = $20,000
    max_position_from_risk = max_loss_usd / (stop_loss_pct / 100)
    
    # Cap at available capital (can't invest more than we have)
    max_position = min(max_position_from_risk, available_capital)
    
    # Calculate actual risk with this position size
    actual_risk_usd = max_position * (stop_loss_pct / 100)
    actual_risk_pct = (actual_risk_usd / total_capital) * 100
    
    if max_position < available_capital:
        msg = (f"Risk limit: Max ${max_position:.2f} position (risk ${actual_risk_usd:.2f} = "
              f"{actual_risk_pct:.2f}% at {stop_loss_pct:.1f}% SL)")
    else:
        msg = f"Risk OK: ${max_position:.2f} position (risk {actual_risk_pct:.2f}% of capital)"
    
    return (max_position, msg)


# VIX cache to avoid excessive API calls
_vix_cache = {"value": None, "timestamp": 0}

def get_vix_level() -> Tuple[Optional[float], str]:
    """
    Get current VIX (Volatility Index) level.
    Returns (vix_value, message)
    
    VIX interpretation:
    - <15: Low volatility (calm market)
    - 15-20: Normal volatility
    - 20-30: Elevated volatility (caution)
    - >30: Extreme fear (pause trading)
    - >40: Panic (major market event)
    
    Caches result for 15 minutes to avoid excessive API calls.
    """
    if not config.VIX_FILTER_ENABLED:
        return (None, "VIX filter disabled")
    
    # Check cache
    cache_age_seconds = time.time() - _vix_cache["timestamp"]
    cache_age_minutes = cache_age_seconds / 60
    
    if _vix_cache["value"] is not None and cache_age_minutes < config.VIX_CACHE_MINUTES:
        return (_vix_cache["value"], f"VIX: {_vix_cache['value']:.1f} (cached {cache_age_minutes:.1f}m ago)")
    
    # Fetch fresh VIX data
    try:
        import yfinance as yf
        
        # VIX symbol is ^VIX
        vix_ticker = yf.Ticker("^VIX")
        
        # Get most recent data (last close or current price if market is open)
        vix_data = vix_ticker.history(period="1d", interval="1m")
        
        if vix_data.empty:
            # Fallback to daily data if intraday fails
            vix_data = vix_ticker.history(period="5d")
        
        if not vix_data.empty:
            vix_value = float(vix_data['Close'].iloc[-1])
            
            # Update cache
            _vix_cache["value"] = vix_value
            _vix_cache["timestamp"] = time.time()
            
            # Interpret VIX level
            if vix_value > 40:
                level = "PANIC"
            elif vix_value > 30:
                level = "EXTREME FEAR"
            elif vix_value > 20:
                level = "ELEVATED"
            elif vix_value > 15:
                level = "NORMAL"
            else:
                level = "CALM"
            
            return (vix_value, f"VIX: {vix_value:.1f} ({level})")
        else:
            log_warn("Could not fetch VIX data - assuming safe to trade")
            return (None, "VIX data unavailable")
    
    except Exception as e:
        log_warn(f"Error fetching VIX: {e} - assuming safe to trade")
        return (None, f"VIX fetch error: {e}")


def check_vix_filter() -> Tuple[bool, str]:
    """
    Check if VIX is too high to trade safely.
    Returns (can_trade, message)
    
    Purpose: Pause trading during extreme market volatility/fear.
    """
    if not config.VIX_FILTER_ENABLED:
        return (True, "")
    
    vix_value, vix_msg = get_vix_level()
    
    if vix_value is None:
        # Can't get VIX - assume safe to trade (don't halt on API errors)
        return (True, vix_msg)
    
    if vix_value > config.VIX_THRESHOLD:
        msg = (f"[VIX FILTER TRIGGERED]\n"
              f"   Current VIX: {vix_value:.1f}\n"
              f"   Threshold: {config.VIX_THRESHOLD:.1f}\n"
              f"   Market is in extreme fear - pausing trading for safety.\n"
              f"   Will resume when VIX drops below {config.VIX_THRESHOLD:.1f}")
        return (False, msg)
    
    # Safe to trade
    return (True, vix_msg)

# ===== Simulation =====
def simulate_signals_and_projection(
    closes: List[float],
    interval_seconds: int,
    override_tp_pct: Optional[float] = None,
    override_sl_pct: Optional[float] = None,
    override_trade_frac: Optional[float] = None,
    override_cap_usd: Optional[float] = None,
    use_walk_forward: bool = True  # NEW PARAMETER
) -> dict:
    
    # Minimum bars needed for SMA calculation
    min_bars = config.LONG_WINDOW + 2  # Need at least LONG_WINDOW + a bit for comparison
    if len(closes) < min_bars:
        # Not enough data - return conservative estimates
        return {
            "win_rate": 0.5,  # Assume 50% win rate
            "expected_trades_per_day": 2.0,  # Estimate 2 trades/day
            "expected_daily_usd": 0.0  # Neutral expectation
        }
    
    # Walk-forward validation to avoid look-ahead bias
    if use_walk_forward and len(closes) >= min_bars * 2:
        # Split: 70% train, 30% test
        split_point = int(len(closes) * 0.7)
        train_closes = closes[:split_point]
        test_closes = closes[split_point:]
        
        # Recursively simulate on both sets (with walk_forward=False to avoid infinite recursion)
        train_sim = simulate_signals_and_projection(
            train_closes, interval_seconds, override_tp_pct, override_sl_pct,
            override_trade_frac, override_cap_usd, use_walk_forward=False
        )
        
        test_sim = simulate_signals_and_projection(
            test_closes, interval_seconds, override_tp_pct, override_sl_pct,
            override_trade_frac, override_cap_usd, use_walk_forward=False
        )
        
        # Return weighted average (70% test, 30% train - emphasize out-of-sample)
        # This is more conservative and realistic
        return {
            "win_rate": train_sim["win_rate"] * 0.3 + test_sim["win_rate"] * 0.7,
            "expected_trades_per_day": train_sim["expected_trades_per_day"] * 0.3 + test_sim["expected_trades_per_day"] * 0.7,
            "expected_daily_usd": train_sim["expected_daily_usd"] * 0.3 + test_sim["expected_daily_usd"] * 0.7,
            # Include individual results for debugging
            "_train_return": train_sim["expected_daily_usd"],
            "_test_return": test_sim["expected_daily_usd"]
        }
    
    # Rest of existing simulation logic (unchanged)
    tp_pct = override_tp_pct if override_tp_pct is not None else config.TAKE_PROFIT_PERCENT
    sl_pct = override_sl_pct if override_sl_pct is not None else config.STOP_LOSS_PERCENT
    frac = override_trade_frac if override_trade_frac is not None else config.TRADE_SIZE_FRAC_OF_CAP
    cap = override_cap_usd if override_cap_usd is not None else config.MAX_CAP_USD
    
    wins = 0
    losses = 0
    total_signals = 0
    total_win_pct = 0.0
    total_loss_pct = 0.0
    consecutive_losses = 0
    max_consecutive_losses = 0
    trade_durations = []
    trade_returns = []  # Track individual trade returns for Sharpe/Sortino
    max_adverse_excursions = []  # Track MAE for each trade
    
    # Track equity curve for max drawdown calculation
    equity = cap  # Starting capital
    equity_curve = [cap]
    peak_equity = cap
    
    for i in range(config.LONG_WINDOW, len(closes)):
        window = closes[:i+1]
        action = decide_action(window, config.SHORT_WINDOW, config.LONG_WINDOW)
        
        if action == "buy":
            total_signals += 1
            price = closes[i]
            
            tp_price = price * (1 + tp_pct / 100)
            sl_price = price * (1 - sl_pct / 100)
            
            # Track worst price against us (MAE)
            worst_price_against = price
            
            for j in range(i + 1, len(closes)):
                # Track MAE (how far price moved against us)
                if closes[j] < worst_price_against:
                    worst_price_against = closes[j]
                
                if closes[j] >= tp_price:
                    wins += 1
                    total_win_pct += tp_pct
                    consecutive_losses = 0
                    trade_durations.append(j - i)
                    trade_returns.append(tp_pct / 100)  # Return as decimal
                    mae = ((price - worst_price_against) / price) * 100  # MAE as percentage
                    max_adverse_excursions.append(mae)
                    # Update equity
                    trade_pnl = (cap * frac) * (tp_pct / 100)
                    equity += trade_pnl
                    equity_curve.append(equity)
                    if equity > peak_equity:
                        peak_equity = equity
                    break
                elif closes[j] <= sl_price:
                    losses += 1
                    total_loss_pct += sl_pct
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    trade_durations.append(j - i)
                    trade_returns.append(-sl_pct / 100)  # Negative return
                    mae = ((price - worst_price_against) / price) * 100
                    max_adverse_excursions.append(mae)
                    # Update equity
                    trade_pnl = -(cap * frac) * (sl_pct / 100)
                    equity += trade_pnl
                    equity_curve.append(equity)
                    if equity > peak_equity:
                        peak_equity = equity
                    break
        
        elif action == "sell":
            total_signals += 1
            price = closes[i]
            
            tp_price = price * (1 - tp_pct / 100)
            sl_price = price * (1 + sl_pct / 100)
            
            # Track worst price against us (MAE for shorts)
            worst_price_against = price
            
            for j in range(i + 1, len(closes)):
                # Track MAE (how far price moved against us - upward for shorts)
                if closes[j] > worst_price_against:
                    worst_price_against = closes[j]
                
                if closes[j] <= tp_price:
                    wins += 1
                    total_win_pct += tp_pct
                    consecutive_losses = 0
                    trade_durations.append(j - i)
                    trade_returns.append(tp_pct / 100)  # Return as decimal
                    mae = ((worst_price_against - price) / price) * 100  # MAE as percentage
                    max_adverse_excursions.append(mae)
                    # Update equity
                    trade_pnl = (cap * frac) * (tp_pct / 100)
                    equity += trade_pnl
                    equity_curve.append(equity)
                    if equity > peak_equity:
                        peak_equity = equity
                    break
                elif closes[j] >= sl_price:
                    losses += 1
                    total_loss_pct += sl_pct
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    trade_durations.append(j - i)
                    trade_returns.append(-sl_pct / 100)  # Negative return
                    mae = ((worst_price_against - price) / price) * 100
                    max_adverse_excursions.append(mae)
                    # Update equity
                    trade_pnl = -(cap * frac) * (sl_pct / 100)
                    equity += trade_pnl
                    equity_curve.append(equity)
                    if equity > peak_equity:
                        peak_equity = equity
                    break
    
    win_rate = wins / total_signals if total_signals > 0 else 0.55  # Assume 55% win rate if no history
    simulation_bars = max(1.0, len(closes) - config.LONG_WINDOW)
    bars_per_day = (86400 / interval_seconds)
    days_simulated = simulation_bars / bars_per_day
    trades_per_day = total_signals / days_simulated if days_simulated > 0 else 4.0  # Estimate 4 trades/day
    
    # Calculate advanced risk metrics
    avg_win_pct = (total_win_pct / wins) if wins > 0 else tp_pct
    avg_loss_pct = (total_loss_pct / losses) if losses > 0 else sl_pct
    profit_factor = (total_win_pct / total_loss_pct) if total_loss_pct > 0 else 2.0
    avg_trade_duration = (sum(trade_durations) / len(trade_durations)) if trade_durations else 5
    
    # Calculate Sharpe & Sortino Ratios
    if len(trade_returns) >= 2:
        import statistics
        mean_return = statistics.mean(trade_returns)
        std_dev = statistics.stdev(trade_returns)
        
        # Sharpe Ratio (risk-free rate assumed to be 0 for simplicity)
        sharpe_ratio = (mean_return / std_dev) if std_dev > 0 else 0.0
        
        # Sortino Ratio (only penalize downside volatility)
        downside_returns = [r for r in trade_returns if r < 0]
        if len(downside_returns) >= 2:
            downside_std = statistics.stdev(downside_returns)
            sortino_ratio = (mean_return / downside_std) if downside_std > 0 else 0.0
        elif len(downside_returns) == 0:
            sortino_ratio = float('inf') if mean_return > 0 else 0.0  # No downside = infinite Sortino
        else:
            sortino_ratio = sharpe_ratio  # Fallback to Sharpe if not enough downside data
    else:
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
    
    # Calculate Expectancy Score
    expectancy = (avg_win_pct * win_rate) - (avg_loss_pct * (1 - win_rate))
    
    # Calculate Maximum Adverse Excursion (MAE) - average of worst moves against us
    avg_mae = (sum(max_adverse_excursions) / len(max_adverse_excursions)) if max_adverse_excursions else 0.0
    max_mae = max(max_adverse_excursions) if max_adverse_excursions else 0.0
    
    # Calculate Max Drawdown from equity curve
    max_drawdown_pct = 0.0
    current_peak = equity_curve[0] if equity_curve else cap
    for eq in equity_curve:
        if eq > current_peak:
            current_peak = eq
        drawdown = ((current_peak - eq) / current_peak) * 100 if current_peak > 0 else 0.0
        if drawdown > max_drawdown_pct:
            max_drawdown_pct = drawdown
    
    # Calculate Recovery Factor: Total Return / Max Drawdown
    total_return_pct = ((equity - cap) / cap) * 100 if cap > 0 else 0.0
    if max_drawdown_pct > 0:
        recovery_factor = total_return_pct / max_drawdown_pct
    else:
        recovery_factor = float('inf') if total_return_pct > 0 else 0.0  # No drawdown = infinite recovery factor
    
    # If we have no historical signals, estimate based on typical SMA crossover frequency
    if total_signals == 0:
        win_rate = 0.55
        trades_per_day = 4.0  # Reasonable estimate for 15-min intervals
        max_consecutive_losses = 3  # Conservative estimate
        profit_factor = 1.5  # Conservative
        avg_trade_duration = 5  # bars
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        expectancy = 0.0
        avg_mae = 0.0
        max_mae = 0.0
        max_drawdown_pct = 0.0
        recovery_factor = 0.0
    
    expected_return_per_trade = ((tp_pct / 100) * win_rate) - ((sl_pct / 100) * (1 - win_rate))
    usable_cap_per_trade = cap * frac
    expected_usd_per_trade = usable_cap_per_trade * expected_return_per_trade
    expected_daily_usd = expected_usd_per_trade * trades_per_day
    
    return {
        "win_rate": win_rate,
        "expected_trades_per_day": trades_per_day,
        "expected_daily_usd": expected_daily_usd,
        # Advanced risk metrics
        "profit_factor": profit_factor,  # Total wins / total losses (>1 = profitable)
        "max_consecutive_losses": max_consecutive_losses,  # Streak risk
        "avg_win_pct": avg_win_pct,  # Average win size
        "avg_loss_pct": avg_loss_pct,  # Average loss size
        "avg_trade_duration_bars": avg_trade_duration,  # How long capital is tied up
        "total_trades": total_signals,  # Total number of trades
        # CRITICAL METRICS (Industry Standard)
        "sharpe_ratio": sharpe_ratio,  # Risk-adjusted return (higher = better)
        "sortino_ratio": sortino_ratio,  # Downside risk-adjusted return (higher = better)
        "expectancy": expectancy,  # Expected % return per trade
        "avg_mae": avg_mae,  # Average worst price move against us (%)
        "max_mae": max_mae,  # Worst MAE seen (%)
        "max_drawdown_pct": max_drawdown_pct,  # Maximum drawdown percentage
        "recovery_factor": recovery_factor,  # Total return / Max drawdown (higher = better recovery)
        # RAW DATA (for Monte Carlo)
        "trade_returns": trade_returns,  # List of individual trade returns (decimals)
    }

def calculate_market_beta(symbol: str, days: int = 60) -> float:
    """
    Calculate beta - how much stock moves relative to S&P 500.
    Beta > 1 = more volatile than market
    Beta < 1 = less volatile than market
    Beta ≈ 1 = moves with market
    
    Args:
        symbol: Stock symbol
        days: Number of days to analyze
    
    Returns:
        Beta value (typically 0.5 to 2.0)
    """
    try:
        import yfinance as yf
        import datetime
        
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=days)
        
        # Fetch stock and SPY (S&P 500 ETF) data
        stock = yf.Ticker(symbol)
        spy = yf.Ticker("SPY")
        
        stock_hist = stock.history(start=start, end=end, interval='1d')
        spy_hist = spy.history(start=start, end=end, interval='1d')
        
        if stock_hist.empty or spy_hist.empty or len(stock_hist) < 10 or len(spy_hist) < 10:
            return 1.0  # Default to market beta
        
        # Calculate daily returns
        stock_returns = stock_hist['Close'].pct_change().dropna()
        spy_returns = spy_hist['Close'].pct_change().dropna()
        
        # Align dates (in case of missing data)
        common_dates = stock_returns.index.intersection(spy_returns.index)
        if len(common_dates) < 10:
            return 1.0
        
        stock_returns = stock_returns.loc[common_dates]
        spy_returns = spy_returns.loc[common_dates]
        
        # Calculate beta: Covariance(stock, market) / Variance(market)
        covariance = stock_returns.cov(spy_returns)
        market_variance = spy_returns.var()
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        
        # Clamp to reasonable range (0.1 to 3.0)
        return max(0.1, min(3.0, beta))
    except:
        return 1.0  # Default to market beta on error


def calculate_overnight_gap_risk(symbol: str, client, days: int = 60) -> dict:
    """
    Calculate overnight gap risk by analyzing historical price gaps between
    close and next open.
    
    Args:
        symbol: Stock symbol
        client: Alpaca client
        days: Number of days to analyze (default: 60 days ~ 3 months)
    
    Returns:
        dict with:
            - gap_frequency: % of days with gaps >2%
            - avg_gap_size: Average gap size in %
            - max_gap: Largest gap seen
            - downward_gap_freq: % of gaps that were negative
    """
    try:
        import yfinance as yf
        import datetime
        
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=days)
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start, end=end, interval='1d')
        
        if hist.empty or len(hist) < 5:
            return {
                "gap_frequency": 0.0,
                "avg_gap_size": 0.0,
                "max_gap": 0.0,
                "downward_gap_freq": 0.0
            }
        
        # Calculate gaps: (Open - Previous Close) / Previous Close
        gaps = []
        downward_gaps = 0
        large_gaps = 0  # >2%
        
        for i in range(1, len(hist)):
            prev_close = hist['Close'].iloc[i-1]
            curr_open = hist['Open'].iloc[i]
            
            if prev_close > 0:
                gap_pct = ((curr_open - prev_close) / prev_close) * 100
                gaps.append(abs(gap_pct))
                
                if abs(gap_pct) > 2.0:
                    large_gaps += 1
                
                if gap_pct < 0:
                    downward_gaps += 1
        
        if len(gaps) == 0:
            return {
                "gap_frequency": 0.0,
                "avg_gap_size": 0.0,
                "max_gap": 0.0,
                "downward_gap_freq": 0.0
            }
        
        return {
            "gap_frequency": (large_gaps / len(gaps)) * 100,  # % with gaps >2%
            "avg_gap_size": sum(gaps) / len(gaps),  # Average gap size
            "max_gap": max(gaps),  # Largest gap
            "downward_gap_freq": (downward_gaps / len(gaps)) * 100  # % negative gaps
        }
    except:
        return {
            "gap_frequency": 0.0,
            "avg_gap_size": 0.0,
            "max_gap": 0.0,
            "downward_gap_freq": 0.0
        }


def monte_carlo_projection(
    trade_returns: list,
    starting_capital: float,
    trades_per_day: float,
    days: int = 30,
    num_simulations: int = 1000
) -> dict:
    """
    Run Monte Carlo simulations to estimate confidence intervals for returns.
    
    Args:
        trade_returns: List of individual trade returns (as decimals, e.g., 0.03 for 3%)
        starting_capital: Initial capital in USD
        trades_per_day: Expected number of trades per day
        days: Number of days to project forward
        num_simulations: Number of Monte Carlo runs (default: 1000)
    
    Returns:
        dict with:
            - p5: 5th percentile (pessimistic)
            - p50: 50th percentile (median)
            - p95: 95th percentile (optimistic)
            - mean: Average across all simulations
    """
    if not trade_returns or len(trade_returns) < 2:
        # Not enough data for Monte Carlo
        return {
            "p5": starting_capital,
            "p50": starting_capital,
            "p95": starting_capital,
            "mean": starting_capital
        }
    
    import random
    
    total_trades = int(trades_per_day * days)
    final_capitals = []
    
    for _ in range(num_simulations):
        capital = starting_capital
        
        # Randomly sample from historical returns with replacement
        for _ in range(total_trades):
            trade_return = random.choice(trade_returns)
            capital *= (1 + trade_return)
        
        final_capitals.append(capital)
    
    # Sort to find percentiles
    final_capitals.sort()
    
    p5_idx = int(num_simulations * 0.05)
    p50_idx = int(num_simulations * 0.50)
    p95_idx = int(num_simulations * 0.95)
    
    # Calculate VaR and CVaR
    # VaR (Value at Risk): 95% confidence that loss won't exceed this
    var_95 = starting_capital - final_capitals[p5_idx]
    
    # CVaR (Conditional VaR): Average loss in worst 5% of cases
    worst_5_percent = final_capitals[:p5_idx] if p5_idx > 0 else [final_capitals[0]]
    avg_worst_case = sum(worst_5_percent) / len(worst_5_percent)
    cvar_95 = starting_capital - avg_worst_case
    
    return {
        "p5": final_capitals[p5_idx],      # 5% chance worse than this (pessimistic)
        "p50": final_capitals[p50_idx],    # Median outcome
        "p95": final_capitals[p95_idx],    # 5% chance better than this (optimistic)
        "mean": sum(final_capitals) / len(final_capitals),
        "var_95": var_95,                  # VaR at 95% confidence
        "cvar_95": cvar_95,                # CVaR (Expected Shortfall) at 95%
    }


# Track if we've already done network wait (prevent duplicate waits)
_network_wait_done = False

# Dynamic Position Sizing: Track recent trade outcomes
# Format: list of (symbol, profit_pct, timestamp)
_recent_trades = []
_MAX_TRADE_HISTORY = 20  # Keep last 20 trades


def update_trade_history(symbol: str, profit_pct: float):
    """
    Track trade outcome for dynamic position sizing.
    
    Args:
        symbol: Stock symbol
        profit_pct: Profit/loss as percentage (positive = win, negative = loss)
    """
    global _recent_trades
    import time
    
    _recent_trades.append((symbol, profit_pct, time.time()))
    
    # Keep only recent trades
    if len(_recent_trades) > _MAX_TRADE_HISTORY:
        _recent_trades = _recent_trades[-_MAX_TRADE_HISTORY:]


def get_dynamic_position_multiplier() -> float:
    """
    Calculate position size multiplier based on recent performance.
    
    Strategy:
    - After wins: Slightly increase position size (compound gains)
    - After losses: Decrease position size (protect capital)
    - Look at last 5 trades
    
    Returns:
        Multiplier (0.5 to 1.5):
        - 0.5 = Half size (after losses)
        - 1.0 = Normal size (neutral)
        - 1.5 = 50% larger (after wins)
    """
    if not _recent_trades or len(_recent_trades) < 2:
        return 1.0  # Neutral if no history
    
    # Look at last 5 trades
    recent = _recent_trades[-5:]
    
    # Calculate win rate and average profit
    wins = sum(1 for _, pct, _ in recent if pct > 0)
    total = len(recent)
    win_rate = wins / total if total > 0 else 0.5
    
    # Calculate average profit/loss
    avg_pnl = sum(pct for _, pct, _ in recent) / total if total > 0 else 0.0
    
    # Dynamic multiplier based on recent performance
    if win_rate >= 0.6 and avg_pnl > 0.5:
        # Hot streak: 3+ wins in last 5, avg profit > 0.5%
        return 1.3  # Increase size by 30%
    elif win_rate >= 0.6:
        # Winning streak but small profits
        return 1.15  # Increase size by 15%
    elif win_rate <= 0.4 and avg_pnl < -0.5:
        # Cold streak: 3+ losses in last 5
        return 0.6  # Decrease size by 40%
    elif win_rate <= 0.4:
        # Losing streak
        return 0.75  # Decrease size by 25%
    else:
        # Neutral (40-60% win rate)
        return 1.0

# ===== Network Connectivity =====
def wait_for_network_on_boot(client, max_wait_seconds: int = 60):
    """
    Wait for network connectivity on system boot.
    Only applies when in SCHEDULED_TASK_MODE.
    If network doesn't come up within max_wait_seconds, exits with error code
    so the PowerShell wrapper restarts the bot (creating an infinite retry loop).
    Only runs once per process lifetime.
    """
    global _network_wait_done
    
    if _network_wait_done:
        return  # Already done, skip
    
    if not SCHEDULED_TASK_MODE:
        _network_wait_done = True
        return  # Not on boot, skip waiting
    
    # Check if we just booted (within last 5 minutes)
    try:
        import psutil
        boot_time = dt.datetime.fromtimestamp(psutil.boot_time())
        now = dt.datetime.now()
        time_since_boot = (now - boot_time).total_seconds()
        
        # If system booted more than 5 minutes ago, skip waiting
        if time_since_boot > 300:
            _network_wait_done = True
            return
    except:
        # If can't check boot time, be safe and wait
        pass
    
    log_info("System appears to have just booted. Waiting for network connectivity...")
    
    for attempt in range(max_wait_seconds):
        try:
            # Try a simple API call to check connectivity
            clock = client.get_clock()
            log_info(f"Network connection established (after {attempt + 1}s)")
            _network_wait_done = True
            return
        except Exception as e:
            if attempt == 0:
                log_info("Network not ready yet, waiting up to 60 seconds...")
            elif attempt % 10 == 0:
                log_info(f"Still waiting for network... ({attempt}s elapsed)")
            time.sleep(1)
    
    # Network never came up - exit with error so PowerShell wrapper restarts us
    log_error(f"Network connection failed after {max_wait_seconds}s. Exiting to retry...")
    log_info("Bot will automatically restart in 10 seconds (managed by PowerShell wrapper)")
    sys.exit(2)

# ===== Market Hours =====
def in_market_hours(client, is_first_check: bool = False) -> bool:
    """
    Check if market is currently open.
    If is_first_check=True, will wait for network on boot before checking.
    """
    if is_first_check:
        wait_for_network_on_boot(client)
    
    try:
        clock = client.get_clock()
        return clock.is_open
    except Exception as e:
        # Don't guess - if API fails, conservatively assume closed
        log_warn(f"Could not check market hours: {e}. Assuming CLOSED for safety.")
        return False

def is_safe_trading_time(client) -> Tuple[bool, str]:
    """
    Check if current time is safe for trading.
    Avoids market open/close volatility.
    Returns (is_safe, reason)
    """
    if not config.ENABLE_SAFE_HOURS:
        return (True, "")
    
    try:
        clock = client.get_clock()
        
        if not clock.is_open:
            return (False, "Market closed")
        
        now = clock.timestamp
        market_close = clock.next_close
        
        # Calculate today's market open time
        # Market is open for 6.5 hours (9:30 AM to 4:00 PM)
        import datetime as dt
        market_hours = dt.timedelta(hours=6, minutes=30)
        market_open = market_close - market_hours
        
        # Calculate minutes since open and until close
        minutes_since_open = (now - market_open).total_seconds() / 60
        minutes_until_close = (market_close - now).total_seconds() / 60
        
        # Check if too close to open
        if minutes_since_open < config.AVOID_FIRST_MINUTES:
            return (False, f"Too close to market open ({minutes_since_open:.0f}m < {config.AVOID_FIRST_MINUTES}m)")
        
        # Check if too close to close
        if minutes_until_close < config.AVOID_LAST_MINUTES:
            return (False, f"Too close to market close ({minutes_until_close:.0f}m < {config.AVOID_LAST_MINUTES}m)")
        
        return (True, "")
        
    except Exception as e:
        log_warn(f"Safe hours check failed: {e}")
        return (False, "Unable to verify trading hours")

def sleep_until_market_open(client):
    # Show countdown info once
    try:
        clock = client.get_clock()
        if clock.next_open:
            now = dt.datetime.now(pytz.UTC)
            next_open = clock.next_open
            if next_open.tzinfo is None:
                next_open = pytz.UTC.localize(next_open)
            
            time_until_open = (next_open - now).total_seconds()
            hours = int(time_until_open // 3600)
            minutes = int((time_until_open % 3600) // 60)
            
            log_info(f"Market closed. Opens in {hours}h {minutes}m ({next_open.astimezone(pytz.timezone('US/Eastern')).strftime('%I:%M %p ET')})")
        else:
            log_info("Market closed. Waiting until open...")
    except:
        log_info("Market closed. Waiting until open...")
    
    # Exit immediately if in scheduled task mode
    if SCHEDULED_TASK_MODE:
        log_info("Scheduled task mode - exiting until next run")
        sys.exit(0)
    
    # Otherwise sleep silently and check periodically (no log spam)
    log_info("Sleeping silently until market opens...")
    while not in_market_hours(client):
        time.sleep(300)  # Check every 5 minutes, but don't log
    
    # Market opened
    log_info("Market is now open!")

def prevent_system_sleep(enable: bool):
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        if enable:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        else:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except:
        pass

# ===== Smart Capital Allocation =====
def allocate_capital_smartly(
    client,
    symbols: List[str],
    forced_symbols: List[str],
    total_capital: float,
    interval_seconds: int,
    min_cap_per_stock: float = None  # Now optional, uses config default
) -> Dict[str, float]:
    """
    TRULY SMART allocation that:
    - Heavily weights best stocks (but maintains some diversification)
    - No fixed max positions (could be 1 or 50 stocks)
    - Considers both return AND risk
    - Skips stocks below minimum threshold
    """
    if min_cap_per_stock is None:
        min_cap_per_stock = config.MIN_ALLOCATION_USD
    
    allocations = {}
    stock_data = {}  # Store all metrics for each stock
    
    # Score each stock with risk adjustment
    log_info(f"Scoring {len(symbols)} stocks for capital allocation...")
    for symbol in symbols:
        try:
            closes = fetch_closes(client, symbol, interval_seconds, 200)
            if not closes or len(closes) < config.LONG_WINDOW + 10:
                log_info(f"  {symbol}: No data (skipped)")
                continue
            
            sim = simulate_signals_and_projection(closes, interval_seconds, override_cap_usd=total_capital)
            expected_daily = sim.get("expected_daily_usd", 0.0)
            confidence = compute_confidence(closes)
            
            # Skip negative expected returns (unless forced)
            if expected_daily <= 0 and symbol not in forced_symbols:
                log_info(f"  {symbol}: exp_daily=${expected_daily:.2f} (SKIPPED - negative)")
                continue
            
            # Calculate volatility (risk measure)
            volatility = pct_stddev(closes[-30:]) if len(closes) >= 30 else 0.05
            
            # Risk-adjusted score: return / volatility (Sharpe-like)
            # Higher return + lower volatility = better score
            risk_adjusted_return = expected_daily / max(volatility, 0.01)
            
            # Final score: risk-adjusted return + confidence boost
            # Best practice: Weight confidence significantly (300-500x) to favor high-probability trades
            score = risk_adjusted_return + (confidence * config.CONFIDENCE_WEIGHT_MULTIPLIER)
            
            # Forced stocks get minimum viable score
            if symbol in forced_symbols:
                score = max(score, 10.0)
            
            stock_data[symbol] = {
                'expected_daily': expected_daily,
                'confidence': confidence,
                'volatility': volatility,
                'risk_adjusted': risk_adjusted_return,
                'score': score
            }
            
            log_info(f"  {symbol}: exp_daily=${expected_daily:.2f}, vol={volatility*100:.1f}%, risk_adj={risk_adjusted_return:.1f}, score={score:.1f}")
        except Exception as e:
            if symbol in forced_symbols:
                stock_data[symbol] = {
                    'expected_daily': 0.01,
                    'confidence': 0.01,
                    'volatility': 0.05,
                    'risk_adjusted': 0.2,
                    'score': 10.0
                }
                log_info(f"  {symbol}: Error ({e}) - using default score (forced)")
    
    if not stock_data:
        log_warn(f"No viable stocks found. Holding cash.")
        return {}
    
    log_info(f"Viable stocks: {len(stock_data)}/{len(symbols)}")
    
    # Best practice: Keep reserve cash (20-30%) for opportunities and safety
    effective_capital = total_capital * (1.0 - config.RESERVE_CASH_PERCENT / 100.0)
    log_info(f"Capital allocation: ${effective_capital:.2f} active (${total_capital - effective_capital:.2f} reserve)")
    
    # Apply concentration factor (makes winners get even MORE)
    # concentration=1.0: proportional
    # concentration=2.0: winners get 2× more than proportional
    # concentration=3.0: winners get 3× more (very aggressive)
    scores = {s: data['score'] ** config.ALLOCATION_CONCENTRATION for s, data in stock_data.items()}
    
    # Calculate raw proportional allocation
    total_score = sum(scores.values())
    if total_score == 0:
        return {}
    
    raw_allocations = {s: (scores[s] / total_score) * effective_capital for s in scores}
    
    # Safety limits (best practice: max 15-25% per stock)
    max_per_stock = effective_capital * (config.MAX_SINGLE_STOCK_PERCENT / 100)
    
    # Apply limits and enforce minimums
    for symbol in list(raw_allocations.keys()):
        # Cap maximum per stock
        if raw_allocations[symbol] > max_per_stock:
            raw_allocations[symbol] = max_per_stock
        
        # Remove stocks below minimum (unless forced or we'd have < 3 stocks)
        if raw_allocations[symbol] < min_cap_per_stock:
            if symbol not in forced_symbols and len(raw_allocations) > config.MIN_DIVERSIFICATION_STOCKS:
                log_info(f"  {symbol}: Allocation ${raw_allocations[symbol]:.2f} < ${min_cap_per_stock:.2f} minimum (removed)")
                del raw_allocations[symbol]
            else:
                # Boost to minimum for diversification
                raw_allocations[symbol] = min_cap_per_stock
                log_info(f"  {symbol}: Boosted to ${min_cap_per_stock:.2f} minimum (diversification)")
    
    # Normalize to stay within effective_capital (with reserve)
    total_allocated = sum(raw_allocations.values())
    if total_allocated > effective_capital:
        scale_factor = effective_capital / total_allocated
        for symbol in raw_allocations:
            raw_allocations[symbol] *= scale_factor
    
    # Final allocations
    allocations = raw_allocations
    
    # Log concentration stats
    if allocations:
        alloc_values = list(allocations.values())
        top_stock_pct = (max(alloc_values) / total_capital) * 100
        avg_per_stock = sum(alloc_values) / len(alloc_values)
        total_used = sum(alloc_values)
        reserve_actual = total_capital - total_used
        log_info(f"Portfolio: {len(allocations)} stocks, top: {top_stock_pct:.1f}%, avg: ${avg_per_stock:.2f}, reserve: ${reserve_actual:.2f}")
    
    return allocations

# ===== Portfolio Evaluation =====
def evaluate_portfolio_and_opportunities(
    client,
    current_symbols: List[str],
    forced_symbols: List[str],
    scan_universe: List[str],
    interval_seconds: int,
    cap_per_stock: float,
    max_positions: int
) -> Dict:
    # Import here to avoid circular dependency
    from stock_scanner import scan_stocks
    
    results = {
        "current_scores": {},
        "opportunities": [],
        "rebalance_suggestions": []
    }
    
    # Score current positions
    for symbol in current_symbols:
        try:
            closes = fetch_closes(client, symbol, interval_seconds, config.LONG_WINDOW + 50)
            if not closes:
                continue
            confidence = compute_confidence(closes)
            action = decide_action(closes, config.SHORT_WINDOW, config.LONG_WINDOW)
            
            score = confidence
            if action == "sell":
                score -= 0.05
            
            if symbol in forced_symbols:
                score += 999
            
            results["current_scores"][symbol] = score
        except:
            if symbol in forced_symbols:
                results["current_scores"][symbol] = 998
            else:
                results["current_scores"][symbol] = -999
    
    # Scan for opportunities
    if len(current_symbols) < max_positions or (results["current_scores"] and min(results["current_scores"].values()) < 0):
        try:
            # Note: cap_per_stock here is actually max_cap from the caller
            scan_results = scan_stocks(
                symbols=[s for s in scan_universe if s not in current_symbols],
                interval_seconds=interval_seconds,
                cap_per_stock=cap_per_stock,  # This is max_cap, passed from main
                max_results=5,
                verbose=False
            )
            results["opportunities"] = scan_results
        except Exception as e:
            log_warn(f"Stock scan failed: {e}")
    
    # Generate rebalance suggestions
    if results["opportunities"] and results["current_scores"]:
        replaceable_stocks = {s: score for s, score in results["current_scores"].items() 
                             if s not in forced_symbols}
        
        if replaceable_stocks:
            worst_held = min(replaceable_stocks.items(), key=lambda x: x[1])
            best_opportunity = results["opportunities"][0] if results["opportunities"] else None
            
            if best_opportunity and worst_held[1] < 0 and best_opportunity["score"] > worst_held[1] + 0.1:
                results["rebalance_suggestions"].append((worst_held[0], best_opportunity["symbol"]))
    
    return results

# ===== Main =====
def main():
    # Import here to avoid circular dependency
    from stock_scanner import scan_stocks, get_stock_universe
    
    # Quick check: Don't even start if market is closed and won't open soon
    if SCHEDULED_TASK_MODE:
        try:
            now = dt.datetime.now(pytz.timezone('US/Eastern'))
            # Only run if it's a weekday and either market is open or within 1 hour of opening
            is_weekday = now.weekday() < 5
            hour = now.hour
            minute = now.minute
            time_in_minutes = hour * 60 + minute
            market_open_minutes = 9 * 60 + 30  # 9:30 AM
            market_close_minutes = 16 * 60      # 4:00 PM
            
            # Exit if it's weekend OR if market closed and more than 1 hour away
            if not is_weekday:
                print("Weekend - exiting")
                return 0
            if time_in_minutes < (market_open_minutes - 60):  # Before 8:30 AM
                print(f"Too early ({now.strftime('%I:%M %p')}) - exiting")
                return 0
            if time_in_minutes > market_close_minutes:  # After 4:00 PM
                print(f"Market closed ({now.strftime('%I:%M %p')}) - exiting")
                return 0
        except:
            pass  # If check fails, continue anyway
    
    parser = argparse.ArgumentParser(description="Unified trading bot (single or multi-stock)")
    parser.add_argument("-t", "--time", type=float, required=True,
                       help="Trading interval in hours (0.25=15min, 1.0=1hr)")
    parser.add_argument("-m", "--max-cap", type=float, required=True,
                       help="Total capital (for multi-stock) or max capital (for single-stock)")
    parser.add_argument("-s", "--symbol", type=str,
                       help="For single-stock mode: stock symbol. For multi-stock: leave blank or use --stocks")
    parser.add_argument("--stocks", nargs="+",
                       help="Force specific stocks in portfolio")
    parser.add_argument("--max-stocks", type=int, default=15,
                       help="Max positions (default: 15, bot will use fewer if not enough profitable stocks)")
    parser.add_argument("--cap-per-stock", type=float,
                       help="Capital per stock (default: total/max)")
    parser.add_argument("--tp", type=float,
                       help="Take profit percent (overrides config)")
    parser.add_argument("--sl", type=float,
                       help="Stop loss percent (overrides config)")
    parser.add_argument("--frac", type=float,
                       help="Position size fraction (overrides config)")
    parser.add_argument("--no-dynamic", action="store_true",
                       help="Disable dynamic adjustments")
    parser.add_argument("--rebalance-every", type=int, default=4,
                       help="Rebalance every N intervals (multi-stock only)")
    parser.add_argument("--go-live", action="store_true",
                       help="Enable live trading")
    parser.add_argument("--allow-missing-keys", action="store_true",
                       help="Debug mode")
    
    args = parser.parse_args()

    # Determine mode
    # If user provides -s (single symbol), they want single-stock mode
    if args.symbol and not args.stocks:
        # Single-stock mode implied
        if args.max_stocks != 15:  # If user explicitly changed it
            pass  # Use their value
        else:
            # Default case with -s: force single-stock
            args.max_stocks = 1
    
    is_multi_stock = args.max_stocks > 1 or args.stocks
    
    if is_multi_stock:
        # Multi-stock mode
        forced_stocks = [s.upper() for s in args.stocks] if args.stocks else []
        
        if len(forced_stocks) > args.max_stocks:
            log_warn(f"Specified {len(forced_stocks)} stocks but max is {args.max_stocks}")
            return 1
        
        # Use smart allocation by default, fallback to equal split if user specifies cap-per-stock
        cap_per_stock = args.cap_per_stock or (args.max_cap / args.max_stocks)
        use_smart_allocation = args.cap_per_stock is None  # Smart unless user forced specific cap
        
        # Only warn if user manually set cap-per-stock to a very small value
        if not use_smart_allocation and cap_per_stock < 10:
            log_warn(f"Manual cap per stock is ${cap_per_stock:.2f} (very small)")
            if not SCHEDULED_TASK_MODE:
                response = input("Continue anyway? (y/n): ").strip().lower()
                if response != 'y':
                    return 1
    else:
        # Single-stock mode
        if not args.symbol:
            log_warn("Single-stock mode requires -s/--symbol")
            return 1
        
        symbol = args.symbol.upper()
        cap_per_stock = args.max_cap
        forced_stocks = [symbol]
    
    # Setup
    interval_seconds = int(args.time * 3600)
    
    try:
        client = make_client(allow_missing=args.allow_missing_keys, go_live=args.go_live)
    except Exception as e:
        log_warn(f"Failed to create client: {e}")
        return 2
    
    # Initialize portfolio manager
    portfolio = PortfolioManager() if is_multi_stock else None
    
    # Log startup
    log_info(f"{'='*70}")
    log_info(f"UNIFIED TRADING BOT - Production Ready")
    log_info(f"{'='*70}")
    log_info(f"  Mode: {'Multi-Stock Portfolio' if is_multi_stock else 'Single-Stock'}")
    log_info(f"  Interval: {interval_seconds}s ({args.time}h)")
    log_info(f"  Total Capital: ${args.max_cap}")
    log_info(f"  Max Positions: {args.max_stocks}")
    
    if is_multi_stock:
        log_info(f"  Forced Stocks: {len(forced_stocks)}")
        log_info(f"  Auto-Fill Slots: {args.max_stocks - len(forced_stocks)}")
        log_info(f"  Rebalance Every: {args.rebalance_every} intervals")
        log_info(f"  Allocation: {'Smart (profit-based)' if use_smart_allocation else f'Equal (${cap_per_stock:.2f} each)'}")
    
    # Log feature status (all 9 improvements)
    log_info(f"\n  Strategy Improvements (9 features enabled by default):")
    log_info(f"    Phase 1 - Advanced Filters:")
    log_info(f"      - RSI Filter: {'[ON]' if config.RSI_ENABLED else '[OFF]'} (blocks overbought/oversold)")
    log_info(f"      - Multi-Timeframe: {'[ON]' if config.MULTI_TIMEFRAME_ENABLED else '[OFF]'} (3 timeframes must agree)")
    log_info(f"      - Volume Confirmation: {'[ON]' if config.VOLUME_CONFIRMATION_ENABLED else '[OFF]'} (requires 1.2x avg volume)")
    log_info(f"    Phase 2 - Risk Management:")
    log_info(f"      - Drawdown Protection: {'[ON]' if config.ENABLE_DRAWDOWN_PROTECTION else '[OFF]'} (stops at {config.MAX_PORTFOLIO_DRAWDOWN_PERCENT}% loss)")
    log_info(f"      - Kelly Criterion: {'[ON]' if config.ENABLE_KELLY_SIZING else '[OFF]'} (optimal position sizing)")
    log_info(f"      - Correlation Check: {'[ON]' if config.ENABLE_CORRELATION_CHECK else '[OFF]'} (diversification)")
    log_info(f"    Phase 3 - Better Execution:")
    log_info(f"      - Limit Orders: {'[ON]' if config.USE_LIMIT_ORDERS else '[OFF]'} ({config.LIMIT_ORDER_OFFSET_PERCENT}% better than market)")
    log_info(f"      - Safe Trading Hours: {'[ON]' if config.ENABLE_SAFE_HOURS else '[OFF]'} (avoids first/last {config.AVOID_FIRST_MINUTES}min)")
    log_info(f"    Phase 4 - Machine Learning:")
    ml = get_ml_predictor()
    ml_status = "[ON & TRAINED]" if (config.ENABLE_ML_PREDICTION and ml.is_trained) else ("[ON - NO MODEL]" if config.ENABLE_ML_PREDICTION else "[OFF]")
    log_info(f"      - Random Forest: {ml_status} (confirms/overrides signals)")
    
    # Wait for network on boot before attempting initial sync
    wait_for_network_on_boot(client)
    
    if is_multi_stock:
        # Sync with broker
        try:
            account = client.get_account()
            positions = client.list_positions()
            log_info(f"Broker: ${float(account.portfolio_value):.2f} total value")
            log_info(f"Found {len(positions)} existing positions")
            for pos in positions:
                portfolio.update_position(
                    pos.symbol,
                    float(pos.qty),
                    float(pos.avg_entry_price),
                    float(pos.market_value),
                    float(pos.unrealized_pl)
                )
        except Exception as e:
            log_warn(f"Could not sync: {e}")
        
        scan_universe = get_stock_universe()
    else:
        log_info(f"  Symbol: {symbol}")
        log_info(f"Broker: ${float(client.get_account().portfolio_value):.2f} total value" if client else "")
    
    log_info(f"{'='*70}\n")
    
    iteration = 0
    consecutive_no_trade_cycles = 0  # Track cycles with no trading activity
    MAX_NO_TRADE_CYCLES = 20  # Alert after 20 idle cycles
    
    try:
        # Keep PC awake during market hours only
        prevent_system_sleep(False)  # Allow sleep initially
        
        while True:
            iteration += 1
            
            # Clear order ID tracker for new iteration (idempotency protection)
            _order_ids_submitted_this_cycle.clear()

            # On first iteration, wait for network if we just booted
            is_first_check = (iteration == 1)
            if not in_market_hours(client, is_first_check=is_first_check):
                prevent_system_sleep(False)  # Allow PC to sleep
                sleep_until_market_open(client)
                continue
            
            # Market is open - keep PC awake
            prevent_system_sleep(True)
            
            # Check if it's a safe time to trade
            is_safe, safe_reason = is_safe_trading_time(client)
            if not is_safe:
                log_info(f"Safe hours check: {safe_reason} - waiting...")
                time.sleep(60)  # Wait 1 minute and recheck
                continue

            log_info(f"=== Iteration {iteration} ===")
            
            # Kill switch check - emergency stop
            can_continue, kill_msg = check_kill_switch()
            if not can_continue:
                log_error(kill_msg)
                log_error("Bot halted by kill switch. Delete the file to resume.")
                # Wait and keep checking instead of exiting - allows resuming without restart
                while not check_kill_switch()[0]:
                    time.sleep(30)  # Check every 30 seconds
                log_info("Kill switch removed - resuming trading...")
                continue
            
            # VIX filter check - pause during extreme market fear
            can_trade_vix, vix_msg = check_vix_filter()
            if not can_trade_vix:
                log_warn(vix_msg)
                log_warn("Pausing trading due to high VIX. Will check again next iteration.")
                time.sleep(interval_seconds)
                continue
            elif vix_msg:
                log_info(f"  {vix_msg}")
            
            if is_multi_stock:
                # Multi-stock logic
                current_positions = portfolio.get_all_positions()
                held_symbols = list(current_positions.keys())
                total_invested = portfolio.get_total_market_value()
                
                log_info(f"Positions: {len(held_symbols)}/{args.max_stocks} | Invested: ${total_invested:.2f}/${args.max_cap:.2f}")
                
                # Safety check: If account value drops below max_cap, something is very wrong
                try:
                    account = client.get_account()
                    equity = float(account.equity)
                    account_value = float(account.portfolio_value)
                    
                    # Check drawdown protection
                    can_trade, dd_msg = check_drawdown_protection(equity)
                    if not can_trade:
                        log_warn(dd_msg)
                        log_warn("Skipping this iteration due to drawdown protection")
                        time.sleep(interval_seconds)
                        continue  # Skip trading this cycle
                    
                    if account_value < args.max_cap * 0.5:  # Lost more than 50%
                        log_error(f"Account value ${account_value:.2f} dropped below 50% of max cap!")
                        log_error("Emergency shutdown - liquidating all positions")
                        for sym in held_symbols:
                            sell_flow(client, sym)
                        sys.exit(1)
                except:
                    pass
                
                # Update positions
                for sym in held_symbols:
                    try:
                        pos = get_position(client, sym)
                        if pos:
                            portfolio.update_position(sym, pos["qty"], pos["avg_entry_price"], 
                                                    pos["market_value"], pos["unrealized_pl"])
                        else:
                            portfolio.remove_position(sym)
                    except:
                        pass
                
                # Rebalancing check
                should_rebalance = (iteration % args.rebalance_every == 0)
                
                if should_rebalance or len(held_symbols) < len(forced_stocks):
                    log_info("Evaluating portfolio...")
                    evaluation = evaluate_portfolio_and_opportunities(
                        client, held_symbols, forced_stocks, scan_universe,
                        interval_seconds, args.max_cap, args.max_stocks  # Use max_cap for consistent predictions
                    )
                    
                    # Execute rebalancing
                    for sell_sym, buy_sym in evaluation["rebalance_suggestions"]:
                        log_info(f"REBALANCE: Sell {sell_sym} -> Buy {buy_sym}")
                        ok, msg = sell_flow(client, sell_sym)
                        if ok:
                            portfolio.remove_position(sell_sym)
                            log_info(f"OK: {msg}")
                
                # Determine stocks to trade
                stocks_to_evaluate = list(set(forced_stocks + held_symbols))
                
                # Add opportunities if room
                if len(stocks_to_evaluate) < args.max_stocks:
                    try:
                        opportunities = scan_stocks(
                            symbols=[s for s in scan_universe if s not in stocks_to_evaluate],
                            interval_seconds=interval_seconds,
                            cap_per_stock=args.max_cap,  # Use total capital for consistent predictions with smart allocation
                            max_results=args.max_stocks - len(stocks_to_evaluate),
                            verbose=True  # Enable to see what's failing
                        )
                        log_info(f"Scanned {len(opportunities)} opportunities (threshold: ${config.PROFITABILITY_MIN_EXPECTED_USD:.2f}/day)")
                        for opp in opportunities:
                            log_info(f"  {opp['symbol']}: ${opp['expected_daily']:.2f}/day (score: {opp['score']:.2f})")
                            if opp["expected_daily"] >= config.PROFITABILITY_MIN_EXPECTED_USD:
                                stocks_to_evaluate.append(opp["symbol"])
                                log_info(f"    -> ADDED to portfolio")
                                if len(stocks_to_evaluate) >= args.max_stocks:
                                    break
                            else:
                                log_info(f"    -> SKIPPED (below ${config.PROFITABILITY_MIN_EXPECTED_USD:.2f} threshold)")
                    except Exception as e:
                        log_error(f"Error scanning stocks: {e}")
                        import traceback
                        log_error(traceback.format_exc())
                        # Continue anyway - don't let scanner errors stop trading
                        stocks_to_evaluate = []
                
                # Smart allocation: calculate capital per stock dynamically
                if use_smart_allocation and stocks_to_evaluate:
                    log_info("Calculating smart capital allocation...")
                    stock_allocations = allocate_capital_smartly(
                        client, stocks_to_evaluate, forced_stocks, 
                        args.max_cap, interval_seconds
                    )
                    
                    # Check if allocation is empty (no good opportunities)
                    if not stock_allocations:
                        log_info("No profitable opportunities found. Holding cash.")
                        continue  # Skip to next iteration
                    
                    # Identify potential sells: stocks we hold but aren't in optimal allocation
                    held_not_optimal = [s for s in held_symbols if s not in stock_allocations]
                    
                    # Consider selling underperformers if we need capital for better stocks
                    available_capital = args.max_cap - total_invested
                    needed_capital = sum(stock_allocations.values())
                    
                    if needed_capital > available_capital and held_not_optimal:
                        log_info(f"  Need ${needed_capital:.2f}, have ${available_capital:.2f} available")
                        log_info(f"  May sell: {', '.join(held_not_optimal)} (not in optimal portfolio)")
                        
                        # Sell weakest positions to free up capital
                        for sym in held_not_optimal:
                            pos = get_position(client, sym)
                            if pos:
                                log_info(f"  REBALANCE: Selling {sym} to reallocate capital")
                                ok, msg = sell_flow(client, sym)
                                if ok:
                                    portfolio.remove_position(sym)
                                    log_info(f"  OK: {msg}")
                                    available_capital += pos["market_value"]
                    
                    # Show allocation breakdown
                    log_info("Target Capital Allocation:")
                    for sym in sorted(stock_allocations.keys(), key=lambda x: stock_allocations[x], reverse=True):
                        has_position = "[HELD]" if sym in held_symbols else "[NEW] "
                        log_info(f"  {has_position} {sym}: ${stock_allocations[sym]:.2f}")
                else:
                    stock_allocations = {sym: cap_per_stock for sym in stocks_to_evaluate}
                
                # Trade each stock
                log_info(f"\nTrading {len(stocks_to_evaluate)} stocks:")
                trades_this_cycle = 0  # Count trades this iteration
                for i, sym in enumerate(stocks_to_evaluate, 1):
                    try:
                        stock_cap = stock_allocations.get(sym, cap_per_stock)
                        
                        # Check if we already have this position
                        existing_pos = get_position(client, sym)
                        existing_value = existing_pos["market_value"] if existing_pos else 0
                        
                        # Calculate how much MORE capital we can use for this stock
                        allocation_diff = stock_cap - existing_value  # Can be negative if over-allocated
                        additional_cap = max(0, allocation_diff)
                        
                        if existing_pos:
                            log_info(f"[{i}/{len(stocks_to_evaluate)}] {sym} (holding: ${existing_value:.2f}, target: ${stock_cap:.2f})...")
                        else:
                            log_info(f"[{i}/{len(stocks_to_evaluate)}] {sym} (target: ${stock_cap:.2f})...")
                        
                        # Check if we need to sell excess (over-allocated)
                        # BUT: Only sell if the stock is underperforming or there's a better opportunity
                        if existing_pos and allocation_diff < -10:  # Over target by $10+
                            # NEW: Check holding period before selling
                            pos_obj = portfolio.get_position(sym)
                            if pos_obj and "first_opened" in pos_obj:
                                try:
                                    opened_time = dt.datetime.fromisoformat(pos_obj["first_opened"])
                                    if opened_time.tzinfo is None:
                                        opened_time = pytz.UTC.localize(opened_time)
                                    held_hours = (dt.datetime.now(pytz.UTC) - opened_time).total_seconds() / 3600
                                    
                                    if held_hours < config.MIN_HOLDING_PERIOD_HOURS:
                                        log_info(f"  HOLD: Won't rebalance {sym} (held {held_hours:.1f}h < {config.MIN_HOLDING_PERIOD_HOURS}h min)")
                                        # Continue to signal check instead of selling
                                    else:
                                        # Held long enough, check if should rebalance
                                        if sym in stock_allocations and stock_allocations[sym] > 10:
                                            # Stock is still in optimal portfolio, keep the excess
                                            log_info(f"  HOLD: Over-allocated by ${-allocation_diff:.2f} but still profitable - keeping")
                                        else:
                                            # Stock is no longer optimal, sell it
                                            log_info(f"  REBALANCE: Over-allocated by ${-allocation_diff:.2f} and not optimal - selling")
                                            ok, msg = sell_flow(client, sym)
                                            if ok:
                                                portfolio.remove_position(sym)
                                                log_info(f"  OK {msg}")
                                            else:
                                                log_info(f"  -- {msg}")
                                            continue
                                except Exception as e:
                                    # If timestamp parsing fails, proceed with normal rebalance logic
                                    if sym in stock_allocations and stock_allocations[sym] > 10:
                                        log_info(f"  HOLD: Over-allocated by ${-allocation_diff:.2f} but still profitable - keeping")
                                    else:
                                        log_info(f"  REBALANCE: Over-allocated by ${-allocation_diff:.2f} and not optimal - selling")
                                        ok, msg = sell_flow(client, sym)
                                        if ok:
                                            portfolio.remove_position(sym)
                                            log_info(f"  OK {msg}")
                                        else:
                                            log_info(f"  -- {msg}")
                                        continue
                            else:
                                # No timestamp info, use old logic
                                if sym in stock_allocations and stock_allocations[sym] > 10:
                                    log_info(f"  HOLD: Over-allocated by ${-allocation_diff:.2f} but still profitable - keeping")
                                else:
                                    log_info(f"  REBALANCE: Over-allocated by ${-allocation_diff:.2f} and not optimal - selling")
                                    ok, msg = sell_flow(client, sym)
                                    if ok:
                                        portfolio.remove_position(sym)
                                        log_info(f"  OK {msg}")
                                    else:
                                        log_info(f"  -- {msg}")
                                    continue
                        
                        closes = fetch_closes(client, sym, interval_seconds, config.LONG_WINDOW + 10)
                        if not closes:
                            log_info(f"  No data")
                            continue
                        
                        # Use multi-timeframe if enabled, else single timeframe
                        if config.MULTI_TIMEFRAME_ENABLED:
                            action, tf_signals = decide_action_multi_timeframe(
                                client, sym, interval_seconds, 
                                config.SHORT_WINDOW, config.LONG_WINDOW
                            )
                            # Log timeframe breakdown
                            tf_str = " | ".join([f"{k}:{v}" for k, v in tf_signals.items()])
                            log_info(f"  Timeframes: {tf_str}")
                        else:
                            action = decide_action(closes, config.SHORT_WINDOW, config.LONG_WINDOW)
                        
                        confidence = compute_confidence(closes)
                        last_price = closes[-1]
                        
                        log_info(f"  ${last_price:.2f} | {action.upper()} | conf={confidence:.4f}")
                        
                        # ML enhancement (if enabled and available)
                        if config.ENABLE_ML_PREDICTION and ML_AVAILABLE:
                            try:
                                ml = get_ml_predictor()
                                if ml.is_trained:
                                    closes_ml, volumes_ml = fetch_closes_with_volume(
                                        client, sym, interval_seconds, 100
                                    )
                                    rsi_ml = compute_rsi(closes_ml) if config.RSI_ENABLED else None
                                    
                                    ml_pred, ml_conf = ml.predict(closes_ml, volumes_ml, rsi_ml)
                                    
                                    # ML agrees with signal
                                    if action == "buy" and ml_pred == 1 and ml_conf > config.ML_CONFIDENCE_THRESHOLD:
                                        log_info(f"  ML confirms BUY (conf={ml_conf:.2%})")
                                    elif action == "sell" and ml_pred == 0 and ml_conf > config.ML_CONFIDENCE_THRESHOLD:
                                        log_info(f"  ML confirms SELL (conf={ml_conf:.2%})")
                                    elif action == "buy" and ml_pred == 0 and ml_conf > config.ML_CONFIDENCE_THRESHOLD:
                                        log_info(f"  ML DISAGREES - predicts DOWN (conf={ml_conf:.2%})")
                                        action = "hold"  # Override signal
                                    elif action == "sell" and ml_pred == 1 and ml_conf > config.ML_CONFIDENCE_THRESHOLD:
                                        log_info(f"  ML DISAGREES - predicts UP (conf={ml_conf:.2%})")
                                        action = "hold"  # Override signal
                            except Exception as e:
                                log_warn(f"ML prediction failed: {e}")
                        
                        enforce_safety(client, sym)
                        
                        # Check correlation before buying new position
                        if action == "buy" and not existing_pos:
                            if config.ENABLE_CORRELATION_CHECK:
                                held_symbols_check = [s for s in held_symbols if s != sym]  # Exclude current
                                is_acceptable, correlations = check_portfolio_correlation(
                                    client, sym, held_symbols_check, interval_seconds
                                )
                                
                                if not is_acceptable:
                                    max_corr_sym = max(correlations, key=correlations.get)
                                    max_corr_val = correlations[max_corr_sym]
                                    log_info(f"  -- Skipped: Too correlated with {max_corr_sym} (corr={max_corr_val:.2f})")
                                    continue  # Skip this stock
                                
                                if correlations:
                                    avg_corr = sum(correlations.values()) / len(correlations)
                                    log_info(f"  Correlation check: avg={avg_corr:.2f} (diversification OK)")
                        
                        if action == "buy":
                            if existing_pos:
                                if additional_cap >= 10:  # Room to add more
                                    log_info(f"  -- Already holding, could add ${additional_cap:.2f} more")
                                else:
                                    log_info(f"  -- Already at target (within $10)")
                            else:
                                # New position
                                ok, msg = buy_flow(
                                    client, sym, last_price, stock_cap,
                                    max(0.0, confidence),
                                    args.frac or config.TRADE_SIZE_FRAC_OF_CAP,
                                    args.tp or config.TAKE_PROFIT_PERCENT,
                                    args.sl or config.STOP_LOSS_PERCENT,
                                    dynamic_enabled=not args.no_dynamic,
                                    interval_seconds=interval_seconds,
                                    total_invested=total_invested,
                                    max_capital=args.max_cap
                                )
                                if ok:
                                    trades_this_cycle += 1
                                log_info(f"  {'OK' if ok else '--'} {msg}")
                        elif action == "sell":
                            # RSI check - don't sell if oversold (might bounce)
                            if config.RSI_ENABLED:
                                closes_check = fetch_closes(client, sym, interval_seconds, config.RSI_PERIOD + 20)
                                if closes_check and len(closes_check) >= config.RSI_PERIOD + 1:
                                    rsi = compute_rsi(closes_check, config.RSI_PERIOD)
                                    if rsi < config.RSI_OVERSOLD:
                                        log_info(f"  -- Oversold: RSI={rsi:.1f} < {config.RSI_OVERSOLD} (holding)")
                                        continue  # Skip sell, might bounce
                            
                            ok, msg = sell_flow(client, sym)
                            if ok:
                                trades_this_cycle += 1
                            log_info(f"  {'OK' if ok else '--'} {msg}")
                        else:
                            if existing_pos:
                                log_info(f"  -- Holding (no signal)")
                            else:
                                log_info(f"  -- No signal")
                    except Exception as e:
                        log_warn(f"  Error: {e}")
                
                # Portfolio summary
                total_value = portfolio.get_total_market_value()
                total_pl = portfolio.get_total_unrealized_pl()
                log_info(f"\nPortfolio: ${total_value:.2f} | P&L: ${total_pl:+.2f}\n")
                
                # Check for freeze (no trades for extended period)
                if trades_this_cycle == 0:
                    consecutive_no_trade_cycles += 1
                    if consecutive_no_trade_cycles >= MAX_NO_TRADE_CYCLES:
                        hours_idle = (consecutive_no_trade_cycles * interval_seconds) / 3600
                        log_warn(f"[WARN] NO TRADES for {hours_idle:.1f}h ({consecutive_no_trade_cycles} cycles)")
                        log_warn(f"[WARN] Market may be unfavorable. Consider:")
                        log_warn(f"     - Checking if market is trending")
                        log_warn(f"     - Adjusting parameters")
                        log_warn(f"     - Pausing bot until conditions improve")
                        consecutive_no_trade_cycles = 0  # Reset
                else:
                    consecutive_no_trade_cycles = 0  # Reset on any trade
            
            else:
                # Single-stock logic
                log_info(f"Trading {symbol}...")
                closes = fetch_closes(client, symbol, interval_seconds, config.LONG_WINDOW + 10)
                if not closes:
                    log_info("No data available")
                    time.sleep(interval_seconds)
                    continue

                # Use multi-timeframe if enabled, else single timeframe
                if config.MULTI_TIMEFRAME_ENABLED:
                    action, tf_signals = decide_action_multi_timeframe(
                        client, symbol, interval_seconds, 
                        config.SHORT_WINDOW, config.LONG_WINDOW
                    )
                    # Log timeframe breakdown
                    tf_str = " | ".join([f"{k}:{v}" for k, v in tf_signals.items()])
                    log_info(f"Timeframes: {tf_str}")
                else:
                    action = decide_action(closes, config.SHORT_WINDOW, config.LONG_WINDOW)
                
                confidence = compute_confidence(closes)
                last_price = closes[-1]

                log_info(f"${last_price:.2f} | {action.upper()} | conf={confidence:.4f}")
                
                # ML enhancement (if enabled and available)
                if config.ENABLE_ML_PREDICTION and ML_AVAILABLE:
                    try:
                        ml = get_ml_predictor()
                        if ml.is_trained:
                            closes_ml, volumes_ml = fetch_closes_with_volume(
                                client, symbol, interval_seconds, 100
                            )
                            rsi_ml = compute_rsi(closes_ml) if config.RSI_ENABLED else None
                            
                            ml_pred, ml_conf = ml.predict(closes_ml, volumes_ml, rsi_ml)
                            
                            # ML agrees with signal
                            if action == "buy" and ml_pred == 1 and ml_conf > config.ML_CONFIDENCE_THRESHOLD:
                                log_info(f"ML confirms BUY (conf={ml_conf:.2%})")
                            elif action == "sell" and ml_pred == 0 and ml_conf > config.ML_CONFIDENCE_THRESHOLD:
                                log_info(f"ML confirms SELL (conf={ml_conf:.2%})")
                            elif action == "buy" and ml_pred == 0 and ml_conf > config.ML_CONFIDENCE_THRESHOLD:
                                log_info(f"ML DISAGREES - predicts DOWN (conf={ml_conf:.2%})")
                                action = "hold"  # Override signal
                            elif action == "sell" and ml_pred == 1 and ml_conf > config.ML_CONFIDENCE_THRESHOLD:
                                log_info(f"ML DISAGREES - predicts UP (conf={ml_conf:.2%})")
                                action = "hold"  # Override signal
                    except Exception as e:
                        log_warn(f"ML prediction failed: {e}")

                enforce_safety(client, symbol)

                if action == "buy":
                    # Get current position value for exposure check
                    current_pos = get_position(client, symbol)
                    current_invested = current_pos["market_value"] if current_pos else 0.0
                    
                    ok, msg = buy_flow(
                        client, symbol, last_price, cap_per_stock,
                        max(0.0, confidence),
                        args.frac or config.TRADE_SIZE_FRAC_OF_CAP,
                        args.tp or config.TAKE_PROFIT_PERCENT,
                        args.sl or config.STOP_LOSS_PERCENT,
                        dynamic_enabled=not args.no_dynamic,
                        interval_seconds=interval_seconds,
                        total_invested=current_invested,
                        max_capital=args.max_cap
                    )
                    log_info(f"{'OK' if ok else '--'} {msg}")
                elif action == "sell":
                    # RSI check - don't sell if oversold (might bounce)
                    if config.RSI_ENABLED:
                        closes_check = fetch_closes(client, symbol, interval_seconds, config.RSI_PERIOD + 20)
                        if closes_check and len(closes_check) >= config.RSI_PERIOD + 1:
                            rsi = compute_rsi(closes_check, config.RSI_PERIOD)
                            if rsi < config.RSI_OVERSOLD:
                                log_info(f"-- Oversold: RSI={rsi:.1f} < {config.RSI_OVERSOLD} (holding)")
                                time.sleep(interval_seconds)
                                continue  # Skip sell, might bounce
                    
                    ok, msg = sell_flow(client, symbol)
                    log_info(f"{'OK' if ok else '--'} {msg}\n")
            
            time.sleep(interval_seconds)
    
    except KeyboardInterrupt:
        log_info("Interrupted - shutting down")
        prevent_system_sleep(False)  # Allow PC to sleep
        return 0
    except Exception as e:
        log_error(f"Fatal error: {e}")
        prevent_system_sleep(False)  # Allow PC to sleep
        return 1
    finally:
        prevent_system_sleep(False)  # Always allow PC to sleep on exit


if __name__ == "__main__":
    sys.exit(main())
