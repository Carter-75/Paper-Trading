
import logging
import sys

from logging.handlers import RotatingFileHandler

# Setup logging configuration if not already set
# Use RotatingFileHandler: Max 5MB per file, keep last 2 backups
file_handler = RotatingFileHandler("bot.log", maxBytes=5*1024*1024, backupCount=2)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        file_handler,
        logging.StreamHandler(sys.stdout)
    ]
)

def log_info(msg: str):
    logging.info(msg)

def log_warn(msg: str):
    logging.warning(msg)

def log_error(msg: str):
    logging.error(msg)
    
def snap_interval_to_supported_seconds(interval: int) -> int:
    """
    Snap arbitrary seconds to nearest supported Alpaca interval.
    Supported: 1m (60), 5m (300), 15m (900), 1h (3600), 1d (86400)
    """
    supported = [60, 300, 900, 3600, 86400]
    
    # Find closest
    closest = supported[0]
    min_diff = abs(interval - closest)
    
    for s in supported:
        diff = abs(interval - s)
        if diff < min_diff:
            min_diff = diff
            closest = s
            
    return closest

def adjust_runtime_params(interval: int, capital: float):
    """
    Placeholder for any dynamic parameter adjustment logic.
    Found in legacy code imports.
    """
    pass
