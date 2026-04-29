import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# Logging setup
# In Scheduled Task mode, stdout is already redirected to bot.log by start_bot.ps1.
# Creating a FileHandler on the same file can cause PermissionError on Windows.
_scheduled_mode = os.environ.get("SCHEDULED_TASK_MODE") == "1" or os.environ.get("BOT_TEE_LOG") == "1"

_handlers = [logging.StreamHandler(sys.stdout)]

if not _scheduled_mode:
    file_handler = RotatingFileHandler("bot.log", maxBytes=5 * 1024 * 1024, backupCount=2, delay=True)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    _handlers.insert(0, file_handler)

logging.basicConfig(
    level=logging.INFO,
    handlers=_handlers,
)


def _prune_log_if_needed():
    """Keeps the log file under 10MB by trimming the top (FIFO)."""
    log_path = "bot.log"
    max_size = 10 * 1024 * 1024  # 10 MB
    target_size = 2 * 1024 * 1024 # Keep latest 2 MB
    
    try:
        if os.path.exists(log_path) and os.path.getsize(log_path) > max_size:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(os.path.getsize(log_path) - target_size)
                # Skip the first (likely partial) line to keep it clean
                f.readline()
                latest_content = f.read()
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("--- LOG PRUNED (FIFO) ---\n" + latest_content)
    except Exception:
        pass # Don't crash the bot because of logging


def log_info(msg: str):
    _prune_log_if_needed()
    logging.info(msg)


def log_warn(msg: str):
    _prune_log_if_needed()
    logging.warning(msg)


def log_error(msg: str):
    _prune_log_if_needed()
    logging.error(msg)


def snap_interval_to_supported_seconds(interval: int) -> int:
    """Snap arbitrary seconds to nearest supported Alpaca interval.
    Supported: 1m (60), 5m (300), 15m (900), 1h (3600), 1d (86400)
    """
    supported = [60, 300, 900, 3600, 86400]

    closest = supported[0]
    min_diff = abs(interval - closest)

    for s in supported:
        diff = abs(interval - s)
        if diff < min_diff:
            min_diff = diff
            closest = s

    return closest


def adjust_runtime_params(interval: int, capital: float):
    """Placeholder for any dynamic parameter adjustment logic."""
    pass
