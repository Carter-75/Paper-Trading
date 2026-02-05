#!/usr/bin/env python3
"""
Pydantic-based configuration with automatic validation.
Replaces manual validation in config.py with type-safe, validated settings.
"""

from pydantic import Field, field_validator, model_validator, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class TradingBotConfig(BaseSettings):
    """Main configuration for the trading bot with automatic validation"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields in .env
    )
    
    # =====================
    # API / Environment
    # =====================
    alpaca_api_key: str = Field(default="", description="Alpaca API key")
    alpaca_secret_key: str = Field(default="", description="Alpaca secret key")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca API base URL"
    )
    polygon_api_key: str = Field(default="", description="Polygon API key (optional)")
    confirm_go_live: str = Field(default="NO", description="Confirmation to go live (YES/NO)")
    
    # =====================
    # Strategy / Runtime
    # =====================
    default_interval_seconds: float = Field(
        default=900.0,
        ge=60.0,
        le=23400.0,
        description="Trading interval in seconds (1 min to 6.5 hours)"
    )
    short_window: int = Field(default=9, ge=1, le=50, description="Short MA window")
    long_window: int = Field(default=21, ge=2, le=200, description="Long MA window")
    
    trade_size_frac_of_cap: float = Field(
        default=0.75,
        ge=0.01,
        le=0.95,
        description="Trade size as fraction of capital"
    )
    fixed_trade_usd: float = Field(default=0.0, ge=0.0, description="Fixed trade size in USD (0=use fraction)")
    max_cap_usd: float = Field(default=100.0, ge=0.0, validation_alias="KILL_SWITCH_EQUITY_FLOOR_USD", description="KILL SWITCH: if account equity drops below this USD, stop bot (0 disables)")
    virtual_account_size: Optional[float] = Field(default=None, description="Override actual account equity with this virtual amount")
    
    # =====================
    # Take Profit / Stop Loss
    # =====================
    take_profit_percent: float = Field(
        default=3.0,
        ge=0.1,
        le=50.0,
        description="Take profit percentage"
    )
    stop_loss_percent: float = Field(
        default=1.0,
        ge=0.1,
        le=50.0,
        description="Stop loss percentage"
    )
    trailing_stop_percent: float = Field(
        default=0.75,
        ge=0.0,
        le=10.0,
        description="Trailing stop percentage"
    )
    
    # =====================
    # Confidence & Volatility
    # =====================
    confidence_multiplier: float = Field(default=9.0, ge=0.1, description="Confidence scaling multiplier")
    min_confidence_to_trade: float = Field(
        default=0.005,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to trade"
    )
    volatility_window: int = Field(default=30, ge=5, le=200, description="Volatility calculation window")
    volatility_pct_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Max volatility threshold"
    )
    
    # =====================
    # Safety / Risk Controls
    # =====================
    max_drawdown_percent: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Maximum drawdown percentage"
    )
    max_position_age_hours: float = Field(
        default=72.0,
        ge=0.0,
        description="Maximum position age in hours"
    )
    daily_loss_limit_usd: float = Field(default=100.0, ge=0.0, description="Daily loss limit in USD")
    max_daily_loss_percent: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description="Max daily loss as % of capital"
    )
    
    max_exposure_pct: float = Field(
        default=75.0,
        ge=1.0,
        le=100.0,
        description="Maximum portfolio exposure percentage"
    )
    
    # =====================
    # Kill Switch
    # =====================
    kill_switch_file: str = Field(default="KILL_SWITCH.flag", description="Kill switch file path")
    kill_switch_enabled: bool = Field(default=True, description="Enable kill switch")
    
    # =====================
    # Order Verification
    # =====================
    order_verification_enabled: bool = Field(default=True, description="Enable order verification")
    max_price_deviation_pct: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Max price deviation percentage"
    )
    max_order_size_adv_pct: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Max order size as % of ADV"
    )
    
    # =====================
    # Max Loss Per Trade
    # =====================
    max_loss_per_trade_pct: float = Field(
        default=2.0,
        ge=0.1,
        le=50.0,
        description="Max loss per trade as % of capital"
    )
    max_loss_per_trade_enabled: bool = Field(default=True, description="Enable max loss per trade")
    
    # =====================
    # VIX Filter
    # =====================
    vix_filter_enabled: bool = Field(default=True, description="Enable VIX-based volatility filter")
    vix_threshold: float = Field(
        default=30.0,
        ge=10.0,
        le=100.0,
        description="VIX threshold for pausing trading"
    )
    vix_cache_minutes: int = Field(default=15, ge=1, le=1440, description="VIX cache duration in minutes")
    
    # =====================
    # Drawdown Protection
    # =====================
    max_portfolio_drawdown_percent: float = Field(
        default=15.0,
        ge=0.0,
        le=100.0,
        description="Max portfolio drawdown percentage"
    )
    enable_drawdown_protection: bool = Field(default=True, description="Enable drawdown protection")
    
    # =====================
    # Kelly Criterion
    # =====================
    enable_kelly_sizing: bool = Field(default=True, description="Enable Kelly criterion sizing")
    kelly_use_half: bool = Field(default=True, description="Use half-Kelly (more conservative)")
    
    # =====================
    # Correlation Check
    # =====================
    enable_correlation_check: bool = Field(default=True, description="Enable correlation-based diversification")
    max_correlation_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Max correlation threshold"
    )
    
    # =====================
    # Limit Orders
    # =====================
    use_limit_orders: bool = Field(default=True, description="Use limit orders instead of market orders")
    limit_order_offset_percent: float = Field(
        default=0.1,
        ge=0.0,
        le=10.0,
        description="Limit order offset percentage"
    )
    limit_order_timeout_seconds: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Limit order timeout in seconds"
    )
    
    # =====================
    # Safe Trading Hours
    # =====================
    enable_safe_hours: bool = Field(default=True, description="Enable safe trading hours")
    avoid_first_minutes: int = Field(default=15, ge=0, le=120, description="Avoid first N minutes of market")
    avoid_last_minutes: int = Field(default=15, ge=0, le=120, description="Avoid last N minutes of market")
    
    # =====================
    # Machine Learning
    # =====================
    enable_ml_prediction: bool = Field(default=True, description="Enable ML prediction")
    ml_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="ML confidence threshold"
    )
    ml_model_path: str = Field(default="ml_model.pkl", description="ML model file path")
    
    # =====================
    # Runtime Clamp Bounds
    # =====================
    min_take_profit_percent: float = Field(default=0.25, ge=0.1, description="Min take profit %")
    max_take_profit_percent: float = Field(default=10.0, ge=0.5, description="Max take profit %")
    min_stop_loss_percent: float = Field(default=0.25, ge=0.1, description="Min stop loss %")
    max_stop_loss_percent: float = Field(default=10.0, ge=0.5, description="Max stop loss %")
    min_trade_size_frac: float = Field(default=0.01, ge=0.001, le=0.5, description="Min trade size fraction")
    max_trade_size_frac: float = Field(default=0.95, ge=0.1, le=1.0, description="Max trade size fraction")
    
    # =====================
    # Logging & Ledger
    # =====================
    log_path: str = Field(default="bot.log", description="Log file path")
    log_max_age_hours: float = Field(default=48.0, ge=0.0, description="Max log age in hours")
    pnl_ledger_path: str = Field(default="pnl_ledger.json", description="P&L ledger file path")
    
    # =====================
    # Behavior
    # =====================
    allow_missing_keys_for_debug: bool = Field(default=False, description="Allow missing API keys for debug")
    enable_market_hours_only: bool = Field(default=True, description="Trade only during market hours")
    
    # =====================
    # Dynamic Scheduling
    # =====================
    dynamic_intervals_enabled: bool = Field(default=True, description="Enable per-stock adaptive polling")
    min_interval_seconds: int = Field(default=15, ge=5, description="Fastest polling rate (high volatility)")
    max_interval_seconds: int = Field(default=120, ge=30, description="Slowest polling rate (calm market)")
    max_api_calls_per_min: int = Field(default=120, le=200, description="Global API rate limit safety cap")

    # =====================
    # Paper Trading / Simulation
    # =====================
    simulate_slippage_enabled: bool = Field(default=True, description="Simulate slippage in paper trading")
    slippage_percent: float = Field(
        default=0.05,
        ge=0.0,
        le=5.0,
        description="Slippage percentage"
    )
    simulate_fees_enabled: bool = Field(default=True, description="Simulate trading fees")
    fee_per_trade_usd: float = Field(default=0.0, ge=0.0, description="Fee per trade in USD (e.g. 1.00)")

    
    # =====================
    # Risky Mode
    # =====================
    risky_mode_enabled: bool = Field(default=True, description="Enable risky mode for higher profits")
    risky_expected_daily_usd_min: float = Field(
        default=0.05,
        ge=0.0,
        description="Min expected daily USD for risky mode"
    )
    risky_vol_multiplier: float = Field(default=2.0, ge=1.0, le=10.0, description="Risky volatility multiplier")
    risky_trades_per_day_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Risky trades per day multiplier"
    )
    risky_tp_mult: float = Field(default=1.25, ge=1.0, le=5.0, description="Risky TP multiplier")
    risky_sl_mult: float = Field(default=1.15, ge=0.5, le=5.0, description="Risky SL multiplier")
    risky_size_mult: float = Field(default=1.30, ge=1.0, le=5.0, description="Risky size multiplier")
    risky_max_frac_cap: float = Field(default=0.95, ge=0.1, le=1.0, description="Risky max fraction cap")
    
    # =====================
    # Profitability Gates
    # =====================
    profitability_gate_enabled: bool = Field(default=True, description="Filter unprofitable stocks")
    profitability_min_expected_usd: float = Field(
        default=0.01,
        ge=0.0,
        description="Min expected USD for profitability"
    )
    strong_confidence_threshold: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Strong confidence threshold"
    )
    strong_confidence_bypass_enabled: bool = Field(
        default=True,
        description="Bypass filters with strong confidence"
    )
    exit_on_negative_projection: bool = Field(
        default=False,
        description="Exit bot if projection is negative"
    )
    
    # =====================
    # RSI Settings
    # =====================
    rsi_period: int = Field(default=14, ge=2, le=100, description="RSI calculation period")
    rsi_overbought: float = Field(default=70.0, ge=50.0, le=100.0, description="RSI overbought level")
    rsi_oversold: float = Field(default=30.0, ge=0.0, le=50.0, description="RSI oversold level")
    rsi_enabled: bool = Field(default=True, description="Enable RSI filter")
    
    # =====================
    # Multi-Timeframe
    # =====================
    multi_timeframe_enabled: bool = Field(default=True, description="Enable multi-timeframe analysis")
    multi_timeframe_min_agreement: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Min timeframes that must agree"
    )
    
    # =====================
    # Volume Confirmation
    # =====================
    volume_confirmation_enabled: bool = Field(default=True, description="Enable volume confirmation")
    volume_confirmation_threshold: float = Field(
        default=1.2,
        ge=1.0,
        le=10.0,
        description="Volume confirmation threshold"
    )
    
    # =====================
    # Rebalancing
    # =====================
    min_holding_period_hours: float = Field(
        default=2.0,
        ge=0.0,
        description="Min holding period in hours"
    )
    rebalance_threshold_percent: float = Field(
        default=15.0,
        ge=0.0,
        le=100.0,
        description="Rebalance threshold percentage"
    )
    
    # =====================
    # Stock Filtering
    # =====================
    min_avg_volume: int = Field(
        default=1000000,
        ge=0,
        description="Minimum average daily volume"
    )
    
    # =====================
    # Additional Settings
    # =====================
    sell_partial_enabled: bool = Field(default=False, description="Enable partial position selling")
    interval_suggestion_window_bars: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Bars for interval suggestion"
    )
    suggestion_max_trades_per_day: float = Field(
        default=20.0,
        ge=1.0,
        le=1000.0,
        description="Max trades per day for suggestion"
    )
    suggestion_min_trials: int = Field(default=60, ge=10, le=500, description="Min trials for suggestion")
    
    # Validators
    @field_validator("short_window")
    @classmethod
    def short_window_valid(cls, v, info):
        """Ensure short window is less than long window"""
        if "long_window" in info.data and v >= info.data["long_window"]:
            raise ValueError(f"short_window ({v}) must be < long_window ({info.data['long_window']})")
        return v
    
    @field_validator("rsi_overbought")
    @classmethod
    def rsi_overbought_valid(cls, v, info):
        """Ensure RSI overbought > oversold"""
        if "rsi_oversold" in info.data and v <= info.data["rsi_oversold"]:
            raise ValueError(f"rsi_overbought ({v}) must be > rsi_oversold ({info.data['rsi_oversold']})")
        return v
    
    @field_validator("take_profit_percent")
    @classmethod
    def take_profit_valid(cls, v, info):
        """Ensure take profit is within bounds"""
        if "min_take_profit_percent" in info.data and v < info.data["min_take_profit_percent"]:
            raise ValueError(f"take_profit_percent ({v}) < min ({info.data['min_take_profit_percent']})")
        if "max_take_profit_percent" in info.data and v > info.data["max_take_profit_percent"]:
            raise ValueError(f"take_profit_percent ({v}) > max ({info.data['max_take_profit_percent']})")
        return v
    
    @field_validator("stop_loss_percent")
    @classmethod
    def stop_loss_valid(cls, v, info):
        """Ensure stop loss is within bounds"""
        if "min_stop_loss_percent" in info.data and v < info.data["min_stop_loss_percent"]:
            raise ValueError(f"stop_loss_percent ({v}) < min ({info.data['min_stop_loss_percent']})")
        if "max_stop_loss_percent" in info.data and v > info.data["max_stop_loss_percent"]:
            raise ValueError(f"stop_loss_percent ({v}) > max ({info.data['max_stop_loss_percent']})")
        return v
    
    @model_validator(mode="after")
    def validate_api_keys(self):
        """Validate API keys if not in debug mode"""
        if not self.allow_missing_keys_for_debug:
            if not self.alpaca_api_key or not self.alpaca_secret_key:
                raise ValueError(
                    "ALPACA_API_KEY and ALPACA_SECRET_KEY are required. "
                    "Set them in .env file or enable ALLOW_MISSING_KEYS_FOR_DEBUG=1"
                )
        return self
    
    @model_validator(mode="after")
    def validate_risky_mode(self):
        """Validate risky mode multipliers don't exceed limits"""
        if self.risky_mode_enabled:
            max_possible_frac = self.trade_size_frac_of_cap * self.risky_size_mult
            if max_possible_frac > self.max_trade_size_frac:
                raise ValueError(
                    f"Risky mode: {self.trade_size_frac_of_cap} * {self.risky_size_mult} = "
                    f"{max_possible_frac:.2f} exceeds max_trade_size_frac ({self.max_trade_size_frac})"
                )
            if max_possible_frac > 1.0:
                raise ValueError(
                    f"Risky mode: Position size can exceed 100% of capital ({max_possible_frac:.1%})"
                )
        return self
    
    def wants_live_mode(self, cli_flag_go_live: bool = False) -> bool:
        """Check if live trading mode is enabled"""
        return bool(cli_flag_go_live and self.confirm_go_live == "YES")


# Global config instance
_config: Optional[TradingBotConfig] = None


def get_config(reload: bool = False) -> TradingBotConfig:
    """
    Get the global config instance.
    
    Args:
        reload: Force reload config from environment
    
    Returns:
        Validated TradingBotConfig instance
    """
    global _config
    if _config is None or reload:
        _config = TradingBotConfig()
    return _config


# Convenience function for backward compatibility
def validate_config(allow_missing_api_keys: bool = False) -> Optional[str]:
    """
    Validate configuration (backward compatible with old config.py).
    
    Args:
        allow_missing_api_keys: Allow missing API keys for debug
    
    Returns:
        Error message if validation fails, None if OK
    """
    try:
        config = TradingBotConfig(allow_missing_keys_for_debug=allow_missing_api_keys)
        return None
    except Exception as e:
        return f"Configuration validation failed: {str(e)}"


if __name__ == "__main__":
    # Test config validation
    print("Testing configuration validation...\n")
    
    try:
        config = get_config()
        print("[OK] Configuration is valid!")
        print(f"\nAPI Keys: {'[SET]' if config.alpaca_api_key else '[MISSING]'}")
        print(f"Interval: {config.default_interval_seconds}s ({config.default_interval_seconds/3600:.2f}h)")
        print(f"Max Capital: ${config.max_cap_usd}")
        print(f"Trade Size: {config.trade_size_frac_of_cap:.1%} of capital")
        print(f"Take Profit: {config.take_profit_percent}%")
        print(f"Stop Loss: {config.stop_loss_percent}%")
        print(f"Risk Controls: {len([k for k, v in config.model_dump().items() if 'enabled' in k and v])} enabled")
    except Exception as e:
        print(f"[X] Configuration validation failed:\n{e}")

