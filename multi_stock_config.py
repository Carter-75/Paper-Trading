"""
Configuration for multi-stock trading.
"""

import os
from typing import List, Optional

# Portfolio settings
MAX_POSITIONS: int = int(os.getenv("MAX_POSITIONS", "15"))
MIN_POSITIONS: int = int(os.getenv("MIN_POSITIONS", "3"))
CAP_PER_STOCK_USD: float = float(os.getenv("CAP_PER_STOCK_USD", "100.0"))
TOTAL_PORTFOLIO_CAP_USD: float = float(os.getenv("TOTAL_PORTFOLIO_CAP_USD", "1500.0"))

# Stock selection
AUTO_SELECT_STOCKS: bool = os.getenv("AUTO_SELECT_STOCKS", "1") in ("1", "true", "True")
FORCED_STOCKS: List[str] = []  # Will be set via CLI
SCAN_UNIVERSE_SIZE: int = int(os.getenv("SCAN_UNIVERSE_SIZE", "24"))  # How many to evaluate

# Rebalancing
REBALANCE_THRESHOLD_PCT: float = float(os.getenv("REBALANCE_THRESHOLD_PCT", "10.0"))  # Replace if 10% underperforming
CHECK_REBALANCE_EVERY_N_INTERVALS: int = int(os.getenv("CHECK_REBALANCE_EVERY_N_INTERVALS", "4"))

# Portfolio file
PORTFOLIO_FILE: str = os.getenv("PORTFOLIO_FILE", "portfolio.json")

