# config.py
"""
Merged & improved config for Paper-Trading bot.

Keep secrets in environment variables or .env (not committed).
This file defines run bases and safe bounds for dynamic tuning.
"""

from typing import Optional
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------------------
# API / Environment
# -------------------
ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")

# Confirmation env var required to go live
CONFIRM_GO_LIVE: str = os.getenv("CONFIRM_GO_LIVE", "YES")

# -------------------
# Strategy / runtime bases (these remain as user-configured defaults)
# -------------------
DEFAULT_TICKER: str = os.getenv("DEFAULT_TICKER", "TSLA")
DEFAULT_INTERVAL_SECONDS: float = float(os.getenv("DEFAULT_INTERVAL_SECONDS", "900"))
SHORT_WINDOW: int = int(os.getenv("SHORT_WINDOW", "9"))
LONG_WINDOW: int = int(os.getenv("LONG_WINDOW", "21"))

# sizing
TRADE_SIZE_FRAC_OF_CAP: float = float(os.getenv("TRADE_SIZE_FRAC_OF_CAP", "0.50"))
FIXED_TRADE_USD: float = float(os.getenv("FIXED_TRADE_USD", "0.0"))

# per-symbol cap (USD)
MAX_CAP_USD: float = float(os.getenv("MAX_CAP_USD", "100.0"))

# TP/SL base percents
TAKE_PROFIT_PERCENT: float = float(os.getenv("TAKE_PROFIT_PERCENT", "3.0") or 0.0)
STOP_LOSS_PERCENT: float = float(os.getenv("STOP_LOSS_PERCENT", "1.0") or 0.0)
TRAILING_STOP_PERCENT: float = float(os.getenv("TRAILING_STOP_PERCENT", "0.0") or 0.0)

# Confidence sizing
CONFIDENCE_MULTIPLIER: float = float(os.getenv("CONFIDENCE_MULTIPLIER", "9.0"))
MIN_CONFIDENCE_TO_TRADE: float = float(os.getenv("MIN_CONFIDENCE_TO_TRADE", "0.005"))

# Volatility filter
VOLATILITY_WINDOW: int = int(os.getenv("VOLATILITY_WINDOW", "30"))
VOLATILITY_PCT_THRESHOLD: float = float(os.getenv("VOLATILITY_PCT_THRESHOLD", "0.10"))

# Sell partial
SELL_PARTIAL_ENABLED: bool = os.getenv("SELL_PARTIAL_ENABLED", "0") in ("1", "true", "True")

# Safety
MAX_DRAWDOWN_PERCENT: float = float(os.getenv("MAX_DRAWDOWN_PERCENT", "10.0") or 0.0)
MAX_POSITION_AGE_HOURS: float = float(os.getenv("MAX_POSITION_AGE_HOURS", "72.0") or 0.0)
DAILY_LOSS_LIMIT_USD: float = float(os.getenv("DAILY_LOSS_LIMIT_USD", "100.0") or 0.0)

# Runtime clamp bounds for auto-tuning
MIN_TAKE_PROFIT_PERCENT: float = float(os.getenv("MIN_TAKE_PROFIT_PERCENT", "0.25"))
MAX_TAKE_PROFIT_PERCENT: float = float(os.getenv("MAX_TAKE_PROFIT_PERCENT", "10.0"))
MIN_STOP_LOSS_PERCENT: float = float(os.getenv("MIN_STOP_LOSS_PERCENT", "0.25"))
MAX_STOP_LOSS_PERCENT: float = float(os.getenv("MAX_STOP_LOSS_PERCENT", "10.0"))
MIN_TRADE_SIZE_FRAC: float = float(os.getenv("MIN_TRADE_SIZE_FRAC", "0.01"))
MAX_TRADE_SIZE_FRAC: float = float(os.getenv("MAX_TRADE_SIZE_FRAC", "0.75"))

# Interval suggestion
INTERVAL_SUGGESTION_WINDOW_BARS: int = int(os.getenv("INTERVAL_SUGGESTION_WINDOW_BARS", "500"))
SUGGESTION_MAX_TRADES_PER_DAY: float = float(os.getenv("SUGGESTION_MAX_TRADES_PER_DAY", "20"))
SUGGESTION_MIN_TRIALS: int = int(os.getenv("SUGGESTION_MIN_TRIALS", "60"))

# Logging & ledger
LOG_PATH: str = os.getenv("LOG_PATH", "bot.log")
LOG_MAX_AGE_HOURS: float = float(os.getenv("LOG_MAX_AGE_HOURS", "48.0") or 0.0)
PNL_LEDGER_PATH: str = os.getenv("PNL_LEDGER_PATH", "pnl_ledger.json")

# Behavior
ALLOW_MISSING_KEYS_FOR_DEBUG: bool = os.getenv("ALLOW_MISSING_KEYS_FOR_DEBUG", "0") in ("1", "true", "True")
ENABLE_MARKET_HOURS_ONLY: bool = os.getenv("ENABLE_MARKET_HOURS_ONLY", "1") in ("1", "true", "True")

# Risk overlay (slightly riskier sizing/targets when expected return justifies it)
RISKY_MODE_ENABLED: bool = os.getenv("RISKY_MODE_ENABLED", "0") in ("1", "true", "True")
RISKY_EXPECTED_DAILY_USD_MIN: float = float(os.getenv("RISKY_EXPECTED_DAILY_USD_MIN", "0.10") or 0.0)
RISKY_VOL_MULTIPLIER: float = float(os.getenv("RISKY_VOL_MULTIPLIER", "1.5") or 1.0)
RISKY_TRADES_PER_DAY_MULTIPLIER: float = float(os.getenv("RISKY_TRADES_PER_DAY_MULTIPLIER", "1.5") or 1.0)
RISKY_TP_MULT: float = float(os.getenv("RISKY_TP_MULT", "1.10") or 1.0)
RISKY_SL_MULT: float = float(os.getenv("RISKY_SL_MULT", "1.05") or 1.0)
RISKY_SIZE_MULT: float = float(os.getenv("RISKY_SIZE_MULT", "1.15") or 1.0)
RISKY_MAX_FRAC_CAP: float = float(os.getenv("RISKY_MAX_FRAC_CAP", "0.85") or 1.0)

# Profitability/Confidence gates
PROFITABILITY_GATE_ENABLED: bool = os.getenv("PROFITABILITY_GATE_ENABLED", "1") in ("1", "true", "True")
PROFITABILITY_MIN_EXPECTED_USD: float = float(os.getenv("PROFITABILITY_MIN_EXPECTED_USD", "0.00") or 0.0)
STRONG_CONFIDENCE_THRESHOLD: float = float(os.getenv("STRONG_CONFIDENCE_THRESHOLD", "0.08"))
STRONG_CONFIDENCE_BYPASS_ENABLED: bool = os.getenv("STRONG_CONFIDENCE_BYPASS_ENABLED", "1") in ("1", "true", "True")

# -------------------
# Helpers
# -------------------
def validate_config(allow_missing_api_keys: bool = False) -> Optional[str]:
    missing = []
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        if not allow_missing_api_keys and not ALLOW_MISSING_KEYS_FOR_DEBUG:
            missing.append("ALPACA_API_KEY and/or ALPACA_SECRET_KEY")
    if missing:
        return f"Missing configuration: {', '.join(missing)}. Set env vars or enable debug override."
    return None

def wants_live_mode(cli_flag_go_live: bool = False) -> bool:
    return bool(cli_flag_go_live and CONFIRM_GO_LIVE == "YES")
