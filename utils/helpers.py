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


def log_info(msg: str):
    logging.info(msg)


def log_warn(msg: str):
    logging.warning(msg)


def log_error(msg: str):
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
