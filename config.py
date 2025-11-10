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
CONFIRM_GO_LIVE: str = os.getenv("CONFIRM_GO_LIVE", "NO")

# -------------------
# Strategy / runtime bases (these remain as user-configured defaults)
# -------------------
DEFAULT_INTERVAL_SECONDS: float = float(os.getenv("DEFAULT_INTERVAL_SECONDS", "900"))
SHORT_WINDOW: int = int(os.getenv("SHORT_WINDOW", "9"))
LONG_WINDOW: int = int(os.getenv("LONG_WINDOW", "21"))

# sizing (increased default from 0.50 to 0.65 for more aggressive profit targeting)
TRADE_SIZE_FRAC_OF_CAP: float = float(os.getenv("TRADE_SIZE_FRAC_OF_CAP", "0.75"))  # Increased from 0.65 for more aggressive trading
FIXED_TRADE_USD: float = float(os.getenv("FIXED_TRADE_USD", "0.0"))

# per-symbol cap (USD)
MAX_CAP_USD: float = float(os.getenv("MAX_CAP_USD", "100.0"))

# TP/SL base percents
TAKE_PROFIT_PERCENT: float = float(os.getenv("TAKE_PROFIT_PERCENT", "3.0") or 0.0)
STOP_LOSS_PERCENT: float = float(os.getenv("STOP_LOSS_PERCENT", "1.0") or 0.0)
TRAILING_STOP_PERCENT: float = float(os.getenv("TRAILING_STOP_PERCENT", "0.75") or 0.0)  # Enable by default: locks in profits

# Confidence sizing
CONFIDENCE_MULTIPLIER: float = float(os.getenv("CONFIDENCE_MULTIPLIER", "9.0"))
MIN_CONFIDENCE_TO_TRADE: float = float(os.getenv("MIN_CONFIDENCE_TO_TRADE", "0.005"))

# Volatility filter (increased threshold from 10% to 15% for more opportunities)
VOLATILITY_WINDOW: int = int(os.getenv("VOLATILITY_WINDOW", "30"))
VOLATILITY_PCT_THRESHOLD: float = float(os.getenv("VOLATILITY_PCT_THRESHOLD", "0.15"))

# Sell partial
SELL_PARTIAL_ENABLED: bool = os.getenv("SELL_PARTIAL_ENABLED", "0") in ("1", "true", "True")

# Safety
MAX_DRAWDOWN_PERCENT: float = float(os.getenv("MAX_DRAWDOWN_PERCENT", "10.0") or 0.0)
MAX_POSITION_AGE_HOURS: float = float(os.getenv("MAX_POSITION_AGE_HOURS", "72.0") or 0.0)
DAILY_LOSS_LIMIT_USD: float = float(os.getenv("DAILY_LOSS_LIMIT_USD", "100.0") or 0.0)
MAX_DAILY_LOSS_PERCENT: float = float(os.getenv("MAX_DAILY_LOSS_PERCENT", "5.0"))

# Exposure limits - prevent going all-in on one idea
MAX_EXPOSURE_PCT: float = float(os.getenv("MAX_EXPOSURE_PCT", "75.0"))  # Keep 25% in cash for opportunities

# Kill Switch - emergency stop mechanism
KILL_SWITCH_FILE: str = os.getenv("KILL_SWITCH_FILE", "KILL_SWITCH.flag")
KILL_SWITCH_ENABLED: bool = os.getenv("KILL_SWITCH_ENABLED", "1") in ("1", "true", "True")

# Order Verification - sanity checks before submission
ORDER_VERIFICATION_ENABLED: bool = os.getenv("ORDER_VERIFICATION_ENABLED", "1") in ("1", "true", "True")
MAX_PRICE_DEVIATION_PCT: float = float(os.getenv("MAX_PRICE_DEVIATION_PCT", "10.0"))  # Max price change from last
MAX_ORDER_SIZE_ADV_PCT: float = float(os.getenv("MAX_ORDER_SIZE_ADV_PCT", "10.0"))  # Max % of avg daily volume

# Max Loss Per Trade - never risk more than this % of capital on one trade
MAX_LOSS_PER_TRADE_PCT: float = float(os.getenv("MAX_LOSS_PER_TRADE_PCT", "2.0"))  # Max risk per trade
MAX_LOSS_PER_TRADE_ENABLED: bool = os.getenv("MAX_LOSS_PER_TRADE_ENABLED", "1") in ("1", "true", "True")

# VIX-Based Volatility Filter - pause trading during extreme market fear
VIX_FILTER_ENABLED: bool = os.getenv("VIX_FILTER_ENABLED", "1") in ("1", "true", "True")
VIX_THRESHOLD: float = float(os.getenv("VIX_THRESHOLD", "30.0"))  # Pause if VIX > 30 (extreme fear)
VIX_CACHE_MINUTES: int = int(os.getenv("VIX_CACHE_MINUTES", "15"))  # Cache VIX for 15 min (avoid excessive API calls)

# Drawdown Protection
MAX_PORTFOLIO_DRAWDOWN_PERCENT: float = float(os.getenv("MAX_PORTFOLIO_DRAWDOWN_PERCENT", "15.0"))
ENABLE_DRAWDOWN_PROTECTION: bool = os.getenv("ENABLE_DRAWDOWN_PROTECTION", "1") in ("1", "true", "True")

# Kelly Criterion Position Sizing
ENABLE_KELLY_SIZING: bool = os.getenv("ENABLE_KELLY_SIZING", "1") in ("1", "true", "True")
KELLY_USE_HALF: bool = os.getenv("KELLY_USE_HALF", "1") in ("1", "true", "True")  # More conservative

# Correlation-Based Diversification
ENABLE_CORRELATION_CHECK: bool = os.getenv("ENABLE_CORRELATION_CHECK", "1") in ("1", "true", "True")
MAX_CORRELATION_THRESHOLD: float = float(os.getenv("MAX_CORRELATION_THRESHOLD", "0.7"))  # 0.7 = high correlation

# Limit Order Settings
USE_LIMIT_ORDERS: bool = os.getenv("USE_LIMIT_ORDERS", "1") in ("1", "true", "True")
LIMIT_ORDER_OFFSET_PERCENT: float = float(os.getenv("LIMIT_ORDER_OFFSET_PERCENT", "0.1"))  # 0.1% better than market
LIMIT_ORDER_TIMEOUT_SECONDS: int = int(os.getenv("LIMIT_ORDER_TIMEOUT_SECONDS", "300"))  # 5 minutes

# Safe Trading Hours
ENABLE_SAFE_HOURS: bool = os.getenv("ENABLE_SAFE_HOURS", "1") in ("1", "true", "True")
AVOID_FIRST_MINUTES: int = int(os.getenv("AVOID_FIRST_MINUTES", "15"))  # Don't trade first 15 min
AVOID_LAST_MINUTES: int = int(os.getenv("AVOID_LAST_MINUTES", "15"))  # Don't trade last 15 min

# Machine Learning
ENABLE_ML_PREDICTION: bool = os.getenv("ENABLE_ML_PREDICTION", "1") in ("1", "true", "True")  # ON by default
ML_CONFIDENCE_THRESHOLD: float = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.6"))  # 60% confidence needed
ML_MODEL_PATH: str = os.getenv("ML_MODEL_PATH", "ml_model.pkl")

# Runtime clamp bounds for auto-tuning
MIN_TAKE_PROFIT_PERCENT: float = float(os.getenv("MIN_TAKE_PROFIT_PERCENT", "0.25"))
MAX_TAKE_PROFIT_PERCENT: float = float(os.getenv("MAX_TAKE_PROFIT_PERCENT", "10.0"))
MIN_STOP_LOSS_PERCENT: float = float(os.getenv("MIN_STOP_LOSS_PERCENT", "0.25"))
MAX_STOP_LOSS_PERCENT: float = float(os.getenv("MAX_STOP_LOSS_PERCENT", "10.0"))
MIN_TRADE_SIZE_FRAC: float = float(os.getenv("MIN_TRADE_SIZE_FRAC", "0.01"))
MAX_TRADE_SIZE_FRAC: float = float(os.getenv("MAX_TRADE_SIZE_FRAC", "0.95"))  # Increased to allow risky mode to reach 0.95

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

# Paper trading slippage simulation (to match live expectations)
SIMULATE_SLIPPAGE_ENABLED: bool = os.getenv("SIMULATE_SLIPPAGE_ENABLED", "1") in ("1", "true", "True")
SLIPPAGE_PERCENT: float = float(os.getenv("SLIPPAGE_PERCENT", "0.05"))  # 5 basis points

# Risk overlay (more aggressive sizing/targets when expected return justifies it)
RISKY_MODE_ENABLED: bool = os.getenv("RISKY_MODE_ENABLED", "1") in ("1", "true", "True")  # Enabled by default for max profit
RISKY_EXPECTED_DAILY_USD_MIN: float = float(os.getenv("RISKY_EXPECTED_DAILY_USD_MIN", "0.05") or 0.0)  # Lower threshold
RISKY_VOL_MULTIPLIER: float = float(os.getenv("RISKY_VOL_MULTIPLIER", "2.0") or 1.0)  # Accept more volatility
RISKY_TRADES_PER_DAY_MULTIPLIER: float = float(os.getenv("RISKY_TRADES_PER_DAY_MULTIPLIER", "2.0") or 1.0)  # More trades OK
RISKY_TP_MULT: float = float(os.getenv("RISKY_TP_MULT", "1.25") or 1.0)  # Higher profit targets
RISKY_SL_MULT: float = float(os.getenv("RISKY_SL_MULT", "1.15") or 1.0)  # Wider stops for more room
RISKY_SIZE_MULT: float = float(os.getenv("RISKY_SIZE_MULT", "1.30") or 1.0)  # Larger position sizes
RISKY_MAX_FRAC_CAP: float = float(os.getenv("RISKY_MAX_FRAC_CAP", "0.95") or 1.0)  # Use nearly all cap

# Profitability/Confidence gates
PROFITABILITY_GATE_ENABLED: bool = os.getenv("PROFITABILITY_GATE_ENABLED", "1") in ("1", "true", "True")  # Filter unprofitable stocks
PROFITABILITY_MIN_EXPECTED_USD: float = float(os.getenv("PROFITABILITY_MIN_EXPECTED_USD", "0.01") or 0.0)  # 1¢ minimum (works with small capital)
STRONG_CONFIDENCE_THRESHOLD: float = float(os.getenv("STRONG_CONFIDENCE_THRESHOLD", "0.08"))
STRONG_CONFIDENCE_BYPASS_ENABLED: bool = os.getenv("STRONG_CONFIDENCE_BYPASS_ENABLED", "1") in ("1", "true", "True")
# Startup safety: exit if expected return is negative
EXIT_ON_NEGATIVE_PROJECTION: bool = os.getenv("EXIT_ON_NEGATIVE_PROJECTION", "0") in ("1", "true", "True")  # Don't exit, just skip unprofitable stocks

# RSI Settings
RSI_PERIOD: int = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERBOUGHT: float = float(os.getenv("RSI_OVERBOUGHT", "70.0"))  # Don't buy above this
RSI_OVERSOLD: float = float(os.getenv("RSI_OVERSOLD", "30.0"))  # Don't sell below this
RSI_ENABLED: bool = os.getenv("RSI_ENABLED", "1") in ("1", "true", "True")

# Multi-Timeframe Settings
MULTI_TIMEFRAME_ENABLED: bool = os.getenv("MULTI_TIMEFRAME_ENABLED", "1") in ("1", "true", "True")
MULTI_TIMEFRAME_MIN_AGREEMENT: int = int(os.getenv("MULTI_TIMEFRAME_MIN_AGREEMENT", "1"))  # How many TFs must agree (1=any timeframe, 2=majority, 3=all)

# Volume Confirmation
VOLUME_CONFIRMATION_ENABLED: bool = os.getenv("VOLUME_CONFIRMATION_ENABLED", "1") in ("1", "true", "True")
VOLUME_CONFIRMATION_THRESHOLD: float = float(os.getenv("VOLUME_CONFIRMATION_THRESHOLD", "1.2"))  # 20% above average

# Rebalancing constraints
MIN_HOLDING_PERIOD_HOURS: float = float(os.getenv("MIN_HOLDING_PERIOD_HOURS", "2.0"))
REBALANCE_THRESHOLD_PERCENT: float = float(os.getenv("REBALANCE_THRESHOLD_PERCENT", "15.0"))

# Stock filtering
MIN_AVG_VOLUME: int = int(os.getenv("MIN_AVG_VOLUME", "1000000"))  # 1M shares/day minimum

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

def validate_risk_config() -> Optional[str]:
    """
    Validate that risk multipliers don't create impossible configurations.
    Returns error message if invalid, None if OK.
    """
    errors = []
    
    # Check if risky mode can exceed max position size
    if RISKY_MODE_ENABLED:
        max_possible_frac = TRADE_SIZE_FRAC_OF_CAP * RISKY_SIZE_MULT
        if max_possible_frac > MAX_TRADE_SIZE_FRAC:
            errors.append(
                f"RISKY_MODE: {TRADE_SIZE_FRAC_OF_CAP} * {RISKY_SIZE_MULT} = {max_possible_frac:.2f} "
                f"exceeds MAX_TRADE_SIZE_FRAC ({MAX_TRADE_SIZE_FRAC})"
            )
        
        if max_possible_frac > 1.0:
            errors.append(
                f"RISKY_MODE: Position size can exceed 100% of capital ({max_possible_frac:.1%})"
            )
    
    # Check TP/SL bounds
    if TAKE_PROFIT_PERCENT < MIN_TAKE_PROFIT_PERCENT:
        errors.append(f"TAKE_PROFIT_PERCENT ({TAKE_PROFIT_PERCENT}) < MIN ({MIN_TAKE_PROFIT_PERCENT})")
    
    if STOP_LOSS_PERCENT < MIN_STOP_LOSS_PERCENT:
        errors.append(f"STOP_LOSS_PERCENT ({STOP_LOSS_PERCENT}) < MIN ({MIN_STOP_LOSS_PERCENT})")
    
    if errors:
        return "Configuration validation failed:\n  - " + "\n  - ".join(errors)
    return None

# Auto-validate on import (warn only, don't crash)
_validation_error = validate_risk_config()
if _validation_error and not ALLOW_MISSING_KEYS_FOR_DEBUG:
    import sys
    import io
    # Use UTF-8 encoding to support emoji characters on Windows
    utf8_stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    # Use text format for log files to avoid encoding issues
    warning_msg = f"\n[WARN] WARNING: {_validation_error}\n"
    print(warning_msg, file=utf8_stderr)
    utf8_stderr.flush()

# Preset configurations
def apply_conservative_preset():
    """Apply safe defaults for new users"""
    global TRADE_SIZE_FRAC_OF_CAP, RISKY_MODE_ENABLED, TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT
    TRADE_SIZE_FRAC_OF_CAP = 0.30  # 30% per position
    RISKY_MODE_ENABLED = False
    TAKE_PROFIT_PERCENT = 2.0
    STOP_LOSS_PERCENT = 1.0
    print("[OK] Applied CONSERVATIVE preset (recommended for beginners)")

def apply_aggressive_preset():
    """Apply maximum profit settings (high risk!)"""
    global TRADE_SIZE_FRAC_OF_CAP, RISKY_MODE_ENABLED, TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT
    TRADE_SIZE_FRAC_OF_CAP = 0.75
    RISKY_MODE_ENABLED = True
    TAKE_PROFIT_PERCENT = 3.0
    STOP_LOSS_PERCENT = 1.5
    print("[!]  Applied AGGRESSIVE preset (high risk, experienced users only)")

# Auto-apply based on environment variable
PRESET_MODE = os.getenv("PRESET_MODE", "").upper()
if PRESET_MODE == "CONSERVATIVE":
    apply_conservative_preset()
elif PRESET_MODE == "AGGRESSIVE":
    apply_aggressive_preset()

