"""
Configuration for Alpaca paper trading bot.

For local use only. You may hardcode keys below as this is not production.
"""

# --- API Keys ---
ALPACA_API_KEY: str = "PKDF9RC2I76C0MQWPEEQ"
ALPACA_API_SECRET: str = "bS1QYSqLXsvFAv2TbIQlJjf6GxSmAVh0zSCjQbPr"
POLYGON_API_KEY: str = "ivl4ABi6FGgDoFaIXFLQMXjdtl80NWq8"

# --- Endpoints ---
# Paper trading base URL by default; later you can switch to live trading
# Note: The client appends /v2 internally; providing the host is sufficient.
ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"

# --- Trading Settings ---
TICKER: str = "TSLA"

# On first run, allocate this USD notional into the ticker.
INITIAL_NOTIONAL_USD: float = 100.0

# Each buy dynamically sizes up to remaining cap; no fixed per-buy amount.

# Strategy uses EMA crossover on hourly closes
SHORT_EMA_HOURS: int = 10  # slower, fewer false buys
LONG_EMA_HOURS: int = 50   # stronger trend confirmation
HOURS_BACK_FOR_TREND: int = 120  # more history for stability

# Runner defaults
DEFAULT_MAX_RUNTIME_HOURS: int = 24

# --- Risk/exit settings ---
# Trailing stop percent for protective sells after buys (e.g., 3.0 for 3%)
TRAILING_STOP_PERCENT: float = 3.0  # protective trailing stop

# Optional bracket order settings for new buys. If both are > 0, buys will use
# a bracket order with take-profit and stop-loss attached instead of a plain
# market order. Prices are derived from the latest trade price.
TAKE_PROFIT_PERCENT: float = 8.0   # optional bracket TP
STOP_LOSS_PERCENT: float = 4.0     # optional bracket SL

# Max allowed drawdown on an open position (negative percent). Example: 5.0 => -5%
MAX_DRAWDOWN_PERCENT: float = 6.0  # force exit if drawdown exceeds this

# Max position age in hours before forcing an exit; 0 disables
MAX_POSITION_AGE_HOURS: float = 72.0  # exit stale positions after 3 days


# --- Logging ---
# Path to the main log file and the maximum age before it is rotated (cleared).
# The controller deletes the log once at startup; the runner also clears it if
# it grows older than this threshold during a long-running session.
LOG_PATH: str = "bot.log"
LOG_MAX_AGE_HOURS: float = 48.0

