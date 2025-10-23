# runner.py
"""
Merged runner: single-file entrypoint.
- CLI: -t/--time (hours), -s/--symbol, -m/--max-cap
- Live mode requires CONFIRM_GO_LIVE=YES and --go-live
- Logs to bot.log with simple truncation/rotation
- Confidence-based dynamic sizing, volatility filter, daily projection
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import datetime as dt
from typing import List, Optional, Tuple
from dotenv import load_dotenv

import pytz
import requests
from alpaca_trade_api import REST
from alpaca_trade_api.rest import APIError
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

import config

# Load environment variables from .env
load_dotenv()

# Keys are read via config; local env reads are unnecessary

# ------- Logging setup (writes to bot.log and console). Keep backward-compatible bot.log use -------
LOG = logging.getLogger("paper_trading_bot")
LOG.setLevel(logging.INFO)

# console handler with simple color output (ANSI). Colors don't affect file logs.
_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s"))
LOG.addHandler(_console)

# file handler: we'll create/rotate/truncate file ourselves to preserve original truncation logic.
FILE_LOG_PATH = config.LOG_PATH
DISABLE_FILE_LOG = os.getenv("BOT_TEE_LOG", "") in ("1", "true", "True")
SCHEDULED_TASK_MODE = os.getenv("SCHEDULED_TASK_MODE", "0") in ("1", "true", "True")

# ensure file exists
if not os.path.exists(FILE_LOG_PATH):
    open(FILE_LOG_PATH, "a").close()

# Simple helper to append line to bot.log (keeps old truncation logic clarified)
def append_to_log_line(line: str):
    if DISABLE_FILE_LOG:
        return
    # Retry a few times to handle transient OneDrive sync file locks
    attempts = 3
    delay = 0.2
    for i in range(attempts):
        try:
            # Keep room for the new line
            try:
                enforce_log_max_lines(99)
            except Exception:
                pass
            with open(FILE_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            return
        except PermissionError:
            if i < attempts - 1:
                time.sleep(delay)
                delay *= 2
            else:
                # Silent skip on persistent lock
                return

def enforce_log_max_lines(max_lines: int = 100):
    try:
        if not os.path.exists(FILE_LOG_PATH):
            return
        with open(FILE_LOG_PATH, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        # Identify the most recent INIT line (keep only this one when truncating)
        init_line = None
        for ln in reversed(lines):
            if ln.startswith("INIT "):
                init_line = ln
                break
        # Identify the most recent session header line
        last_header_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if "Starting bot for" in lines[i]:
                last_header_idx = i
                break
        header_line = lines[last_header_idx] if last_header_idx is not None else None
        # Always enforce max_lines, with simple tailing and INIT handling
        if len(lines) <= max_lines:
            # Ensure only the latest INIT is kept and placed first (counted toward cap)
            try:
                reordered = []
                if init_line:
                    reordered.append(init_line)
                # Add header next if present and not duplicate
                if header_line and (not reordered or header_line != reordered[-1]):
                    reordered.append(header_line)
                # Append the remaining lines excluding duplicates and any other INITs
                seen = set(reordered)
                for l in lines:
                    if l.startswith("INIT "):
                        continue
                    if l in seen:
                        continue
                    reordered.append(l)
                # Clamp to max_lines while keeping INIT/header at the top
                if len(reordered) > max_lines:
                    keep_head = 2 if header_line else (1 if init_line else 0)
                    tail_keep = max_lines - keep_head
                    head = reordered[:keep_head]
                    tail = reordered[-tail_keep:] if tail_keep > 0 else []
                    lines = head + tail
                else:
                    lines = reordered
            except Exception:
                pass
            with open(FILE_LOG_PATH, "w", encoding="utf-8") as fh:
                fh.writelines(lines)
            return
        # File is over the limit: build a truncated view
        tail_count = max_lines - 1  # reserve 1 for ellipsis or INIT
        # Exclude all INIT lines from the tail selection
        non_init_lines = [l for l in lines if not l.startswith("INIT ")]
        tail = non_init_lines[-tail_count:]
        # Ensure the most recent header is present in the tail
        if header_line and header_line in non_init_lines and header_line not in tail:
            if tail:
                tail[0] = header_line
            else:
                tail = [header_line]
        # Start with ellipsis + tail
        new_lines = ["...\n"] + tail
        # Place the most recent INIT on top if present
        if init_line:
            # Ensure INIT is first and counted toward limit
            new_lines = [init_line] + new_lines[:-1]  # replace one tail slot to keep total at max_lines
        # Guarantee total lines does not exceed max_lines
        if len(new_lines) > max_lines:
            new_lines = new_lines[-max_lines:]
        lines = new_lines
        with open(FILE_LOG_PATH, "w", encoding="utf-8") as fh:
            fh.writelines(lines)
    except Exception:
        pass

# write wrapper to both console and file (keeps old log format in file)
def log_info(msg: str, *args):
    LOG.info(msg, *args)
    try:
        append_to_log_line(f"{dt.datetime.now(pytz.UTC).isoformat()} INFO - {msg % args if args else msg}")
    except Exception:
        append_to_log_line(f"{dt.datetime.now(pytz.UTC).isoformat()} INFO - {msg}")

def log_warn(msg: str, *args):
    LOG.warning(msg, *args)
    try:
        append_to_log_line(f"{dt.datetime.now(pytz.UTC).isoformat()} WARN - {msg % args if args else msg}")
    except Exception:
        append_to_log_line(f"{dt.datetime.now(pytz.UTC).isoformat()} WARN - {msg}")

def log_exc(msg: str, *args):
    LOG.exception(msg, *args)
    try:
        append_to_log_line(f"{dt.datetime.now(pytz.UTC).isoformat()} EXC - {msg % args if args else msg}")
    except Exception:
        append_to_log_line(f"{dt.datetime.now(pytz.UTC).isoformat()} EXC - {msg}")

# Utility
def now_utc() -> dt.datetime:
    return dt.datetime.now(pytz.UTC)

# Prevent system sleep (Windows): keep machine awake during market hours / pre-open
try:
    import ctypes  # type: ignore
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    # ES_DISPLAY_REQUIRED could be added if you want to prevent display sleep too
    def prevent_system_sleep(enable: bool):
        try:
            if os.name == "nt":
                flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
                ctypes.windll.kernel32.SetThreadExecutionState(flags)
        except Exception:
            pass
except Exception:
    def prevent_system_sleep(enable: bool):  # type: ignore
        return

def read_json(path):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        log_exc("read_json failed for %s", path)
    return {}

def write_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, default=str, indent=2)
    except Exception:
        log_exc("write_json failed for %s", path)

# Retry helper
def retry(fn, tries=3, delay=1.0, backoff=2.0, allowed_exceptions=(Exception,), name=None):
    n = 0
    last_exc = None
    while n < tries:
        try:
            return fn()
        except allowed_exceptions as e:
            last_exc = e
            n += 1
            log_warn("Transient error in %s: %s (attempt %d/%d)", name or getattr(fn, "__name__", str(fn)), e, n, tries)
            time.sleep(delay)
            delay *= backoff
    log_exc("All retries failed for %s", name or getattr(fn, "__name__", str(fn)))
    raise last_exc

# Alpaca client factory
def make_client(allow_missing=False, go_live=False) -> REST:
    # Use base URL depending on go_live (but still require explicit env + flag)
    if go_live:
        # require user set CONFIRM_GO_LIVE=YES
        if not config.wants_live_mode(cli_flag_go_live=True):
            raise RuntimeError("Attempt to go live without CONFIRM_GO_LIVE=YES")
    if (not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY) and not (allow_missing or config.ALLOW_MISSING_KEYS_FOR_DEBUG):
        raise RuntimeError("Missing Alpaca API keys")
    base = config.ALPACA_BASE_URL
    client = REST(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, base_url=base, api_version='v2')
    return client

# Market data helper
def map_interval_to_timeframe(interval_seconds: int) -> TimeFrame:
    if interval_seconds < 60:
        return TimeFrame.Minute
    minutes = interval_seconds // 60
    if minutes == 1:
        return TimeFrame.Minute
    if minutes == 5:
        return TimeFrame(5, TimeFrameUnit.Minute)
    if minutes == 15:
        return TimeFrame(15, TimeFrameUnit.Minute)
    if minutes >= 60:
        return TimeFrame.Hour
    return TimeFrame.Minute

def snap_interval_to_supported_seconds(interval_seconds: int) -> int:
    # Snap any requested interval to one of: 1, 5, 15, 60 minutes
    if interval_seconds < 60:
        return 60
    minutes = max(1, round(interval_seconds / 60.0))
    supported = [1, 5, 15, 60]
    closest = min(supported, key=lambda m: abs(m - minutes))
    return 3600 if closest >= 60 else closest * 60

def fetch_closes(client: REST, symbol: str, interval_seconds: int, bars: int) -> List[float]:
    requested_secs = max(1, int(interval_seconds))
    snapped_secs = snap_interval_to_supported_seconds(requested_secs)
    if snapped_secs != requested_secs:
        try:
            log_warn("Adjusted interval %ds -> %ds to match supported sizes (1/5/15/60m)", requested_secs, snapped_secs)
        except Exception:
            pass
    tf = map_interval_to_timeframe(snapped_secs)
    def _extract_closes(barset_obj) -> List[float]:
        closes_local: List[float] = []
        df = getattr(barset_obj, "df", None)
        if df is not None:
            try:
                closes_series = getattr(df, 'close', None)
                if closes_series is not None:
                    closes_list = list(closes_series)
                    closes_local = [float(x) for x in closes_list[-bars:]]
            except Exception:
                closes_local = []
        if not closes_local:
            try:
                closes_local = [float(getattr(b, "c", getattr(b, "close", 0.0))) for b in barset_obj][-bars:]
            except Exception:
                closes_local = []
        return closes_local

    def _get():
        # Helper to fetch from Alpaca (limit or start)
        def fetch_from_alpaca() -> List[float]:
            # First try a simple limit-based fetch (fast path)
            barset = client.get_bars(symbol, tf, limit=bars)
            closes_a = _extract_closes(barset)
            if len(closes_a) >= bars:
                return closes_a
            # Fallback: request by start time far enough back to guarantee bars
            minutes_local = max(1, int(snapped_secs) // 60)
            if minutes_local >= 60:
                seconds_per_bar_local = 3600
            elif minutes_local >= 15:
                seconds_per_bar_local = 900
            elif minutes_local >= 5:
                seconds_per_bar_local = 300
            else:
                seconds_per_bar_local = 60
            lookback_bars_local = max(bars * 5, bars + 10)
            start_dt_local = now_utc() - dt.timedelta(seconds=seconds_per_bar_local * lookback_bars_local)
            start_str_local = start_dt_local.astimezone(pytz.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            barset2 = client.get_bars(symbol, tf, start=start_str_local)
            closes_b = _extract_closes(barset2)
            return closes_b

        def map_to_polygon_timespan(interval_secs: int) -> str:
            if interval_secs < 60:
                return "minute"
            minutes_local = interval_secs // 60
            if minutes_local < 60:
                return "minute"
            if minutes_local < 24 * 60:
                return "hour"
            return "day"

        def fetch_from_polygon() -> List[float]:
            api_key = config.POLYGON_API_KEY
            if not api_key:
                return []
            # Decide multiplier and timespan based on our mapped timeframe
            minutes_local = max(1, int(snapped_secs) // 60)
            timespan = map_to_polygon_timespan(int(snapped_secs))
            multiplier = 1
            if timespan == "minute":
                if minutes_local in (1, 5, 15):
                    multiplier = minutes_local
                elif minutes_local >= 60:
                    timespan = "hour"
                    multiplier = max(1, minutes_local // 60)
                else:
                    multiplier = 1
            elif timespan == "hour":
                multiplier = max(1, minutes_local // 60)
            else:
                multiplier = 1
            lookback = max(bars * 5, bars + 10)
            end_dt = now_utc()
            # Estimate bars per trading day for chosen timespan/multiplier
            if timespan == "minute":
                bars_per_day = max(1.0, 390.0 / max(1, multiplier))
            elif timespan == "hour":
                bars_per_day = max(1.0, 6.5 / max(1, multiplier))
            else:  # day
                bars_per_day = 1.0 / max(1, multiplier)
            needed_days = int(math.ceil((lookback * 1.25) / max(1.0, bars_per_day)))
            needed_days = max(5, min(365, needed_days))
            start_dt_local = end_dt - dt.timedelta(days=needed_days)
            start_iso = start_dt_local.strftime("%Y-%m-%d")
            end_iso = end_dt.strftime("%Y-%m-%d")
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_iso}/{end_iso}"
            params = {"adjusted": "true", "sort": "asc", "limit": str(50000), "apiKey": api_key}
            try:
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code != 200:
                    return []
                data = resp.json()
                results = data.get("results", [])
                closes_p = [float(item.get("c", 0.0)) for item in results if item.get("c") is not None]
                if not closes_p:
                    return []
                return closes_p[-bars:]
            except Exception:
                return []

        # Optional test toggle to force Polygon-only path
        force_polygon = os.getenv("FORCE_POLYGON_FALLBACK", "").lower() in ("1", "true", "yes")
        if force_polygon:
            log_warn("FORCE_POLYGON_FALLBACK=1 set; using Polygon only for bars")
            closes_poly_only = fetch_from_polygon()
            if closes_poly_only:
                log_info("Using Polygon fallback (forced): %d bars", len(closes_poly_only))
                return closes_poly_only[-bars:]
            LOG.debug("Forced Polygon path returned no data; attempting Alpaca as backup")

        # Try Alpaca fetch; if it errors, catch and try Polygon
        try:
            closes = fetch_from_alpaca()
            if len(closes) >= min(1, bars):
                LOG.debug("Using Alpaca bars: %d", len(closes))
                return closes[-bars:]
        except Exception:
            LOG.debug("Alpaca bars fetch raised; attempting Polygon fallback")
        # Alpaca insufficient or failed; try Polygon fallback
        log_warn("Alpaca bars unavailable or insufficient; trying Polygon fallback")
        closes_poly = fetch_from_polygon()
        if closes_poly:
            log_info("Using Polygon fallback: %d bars", len(closes_poly))
            return closes_poly[-bars:]
        # If Polygon also fails, stop here and let the outer retry handle remaining attempts
        LOG.debug("Polygon fallback failed or empty; giving up this attempt")
        raise APIError("Market data unavailable from both Alpaca and Polygon in this attempt")
    # Cap to 3 total attempts (initial + 2 retries)
    return retry(_get, tries=3, delay=1.0, backoff=2.0, allowed_exceptions=(APIError, Exception), name=f"get_bars({symbol})")

# Indicators & logic
def sma(values: List[float], window: int) -> Optional[float]:
    if not values or window <= 0 or len(values) < window:
        return None
    return sum(values[-window:]) / window

def pct_stddev(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    return std / mean if mean != 0 else 0.0

def compute_confidence(closes: List[float]) -> float:
    short = sma(closes, config.SHORT_WINDOW)
    long = sma(closes, config.LONG_WINDOW)
    if short is None or long is None or long == 0:
        return 0.0
    raw = (short - long) / long
    scaled = raw * config.CONFIDENCE_MULTIPLIER
    if scaled > 1.0:
        scaled = 1.0
    if scaled < -1.0:
        scaled = -1.0
    return float(scaled)

def decide_action(closes: List[float], short_w: int, long_w: int) -> str:
    if not closes or len(closes) < max(short_w, long_w):
        return "hold"
    prev_short = sma(closes[:-1], short_w)
    prev_long = sma(closes[:-1], long_w)
    cur_short = sma(closes, short_w)
    cur_long = sma(closes, long_w)
    if any(v is None for v in (prev_short, prev_long, cur_short, cur_long)):
        return "hold"
    # Confidence gate: only hold if confidence is too low
    conf = compute_confidence(closes)
    if abs(conf) < config.MIN_CONFIDENCE_TO_TRADE:
        return "hold"
    # Prefer crossover signals
    if prev_short <= prev_long and cur_short > cur_long:
        return "buy"
    if prev_short >= prev_long and cur_short < cur_long:
        return "sell"
    # Otherwise follow current trend direction
    if cur_short > cur_long:
        return "buy"
    if cur_short < cur_long:
        return "sell"
    return "hold"

# Orders & positions
def has_open_order(client: REST, symbol: str, side: str) -> bool:
    try:
        # alpaca-trade-api>=3.x uses list_orders
        orders = client.list_orders(status='open', limit=50)
        return any((getattr(o, "symbol", "").upper() == symbol.upper()) and (getattr(o, "side", "").lower() == side.lower()) for o in orders)
    except Exception:
        log_exc("has_open_order failed for %s", symbol)
    return False

def get_position(client: REST, symbol: str) -> Optional[dict]:
    try:
        pos = client.get_position(symbol)
        return {"qty": float(pos.qty), "avg_entry_price": float(pos.avg_entry_price), "market_value": float(pos.market_value), "unrealized_pl": float(pos.unrealized_pl)}
    except Exception:
        return None

def wait_for_order_fill(client: REST, order_id: str, timeout: float = 30.0):
    start = time.time()
    while True:
        o = client.get_order(order_id)
        status = getattr(o, "status", "").lower()
        if status in ("filled", "partially_filled", "canceled", "cancelled"):
            return o
        if time.time() - start > timeout:
            raise TimeoutError(f"Order {order_id} not filled within {timeout}s (status={status})")
        time.sleep(1.0)

def place_market_order(client: REST, symbol: str, qty: float, side: str = "buy"):
    if qty <= 0:
        raise ValueError("qty must be > 0")
    log_info("Submitting market %s %s qty=%.6f", side.upper(), symbol, qty)
    order = client.submit_order(symbol=symbol, qty=str(qty), side=side, type="market", time_in_force="day")
    final = wait_for_order_fill(client, order.id, timeout=30.0)
    return final

def place_limit_order(client: REST, symbol: str, qty: float, price: float, side: str = "sell"):
    log_info("Submitting limit %s %s qty=%.6f limit=%s", side.upper(), symbol, qty, price)
    # Fractional equity orders must be DAY orders per Alpaca
    tif = "day"
    order = client.submit_order(symbol=symbol, qty=str(qty), side=side, type="limit", time_in_force=tif, limit_price=str(price))
    return order

def place_stop_order(client: REST, symbol: str, qty: float, stop_price: float, side: str = "sell"):
    log_info("Submitting stop %s %s qty=%.6f stop=%s", side.upper(), symbol, qty, stop_price)
    # Fractional equity orders must be DAY orders per Alpaca
    tif = "day"
    order = client.submit_order(symbol=symbol, qty=str(qty), side=side, type="stop", time_in_force=tif, stop_price=str(stop_price))
    return order

def submit_tp_sl(client: REST, symbol: str, qty: float, entry_price: float, tp_pct: float, sl_pct: float) -> Tuple[Optional[object], Optional[object]]:
    tp_order = None
    sl_order = None
    try:
        if tp_pct and tp_pct > 0:
            tp_price = round(entry_price * (1.0 + tp_pct / 100.0), 2)
            tp_order = place_limit_order(client, symbol, qty, tp_price, side="sell")
        if sl_pct and sl_pct > 0:
            stop_price = round(entry_price * (1.0 - sl_pct / 100.0), 2)
            sl_order = place_stop_order(client, symbol, qty, stop_price, side="sell")
    except Exception:
        log_exc("submit_tp_sl failed")
    return tp_order, sl_order

# Sizing & dynamic adjustments
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def adjust_runtime_params(base_tp, base_sl, base_frac, confidence, vol_pct):
    c = max(0.0, min(1.0, confidence))
    vol_factor = 1.0 / (1.0 + vol_pct * 10.0) if vol_pct >= 0 else 1.0
    tp = base_tp * (1.0 + 0.5 * c)
    sl = base_sl * (1.0 - 0.3 * c)
    frac = base_frac * (1.0 + 0.5 * c) * vol_factor
    tp = clamp(tp, config.MIN_TAKE_PROFIT_PERCENT, config.MAX_TAKE_PROFIT_PERCENT)
    sl = clamp(sl, config.MIN_STOP_LOSS_PERCENT, config.MAX_STOP_LOSS_PERCENT)
    frac = clamp(frac, config.MIN_TRADE_SIZE_FRAC, config.MAX_TRADE_SIZE_FRAC)
    return tp, sl, frac

def compute_order_qty_from_remaining(remaining_usd: float, price: float, confidence: float, frac: float, fixed_usd: float = 0.0, scale_with_confidence: bool = True):
    if price <= 0:
        return 0.0
    use_fixed = (fixed_usd and fixed_usd > 0) or (config.FIXED_TRADE_USD and config.FIXED_TRADE_USD > 0)
    if use_fixed:
        target = fixed_usd if (fixed_usd and fixed_usd > 0) else config.FIXED_TRADE_USD
        usd = min(remaining_usd, target)
    else:
        usd = remaining_usd * frac
    if scale_with_confidence:
        scale = max(0.0, min(1.0, confidence))
        usd = usd * scale
    if usd <= 0:
        return 0.0
    qty = math.floor((usd / price) * 1_000_000) / 1_000_000
    return qty

# Ledger for realized PnL
def ledger_load(path):
    return read_json(path) or {}

def ledger_add_realized(path, ts, realized_pl):
    data = ledger_load(path)
    recs = data.get("records", [])
    recs.append({"ts": ts, "realized_pl": float(realized_pl)})
    data["records"] = recs
    write_json(path, data)

def ledger_today_loss(path):
    data = ledger_load(path)
    recs = data.get("records", [])
    today = dt.datetime.now(pytz.UTC).date()
    total_loss = 0.0
    for r in recs:
        try:
            t = dt.datetime.fromisoformat(r["ts"])
        except Exception:
            try:
                t = dt.datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M:%S.%f%z")
            except Exception:
                continue
        if t.date() == today:
            pl = float(r.get("realized_pl", 0.0))
            if pl < 0:
                total_loss += -pl
    return total_loss

# Safety enforcement
def sell_all(client: REST, symbol: str):
    pos = get_position(client, symbol)
    if not pos:
        return False
    qty = float(pos.get("qty", 0.0))
    if qty <= 0:
        return False
    try:
        order = place_market_order(client, symbol, qty, side="sell")
        ledger_add_realized(config.PNL_LEDGER_PATH, now_utc().isoformat(), 0.0)
        log_info("Sold all position qty=%.6f", qty)
        return True
    except Exception:
        log_exc("sell_all failed")
        return False

def enforce_safety(client: REST, symbol: str):
    pos = get_position(client, symbol)
    if not pos:
        return
    unrealized = float(pos.get("unrealized_pl", 0.0))
    market_value = float(pos.get("market_value", 0.0))
    if market_value <= 0:
        return
    try:
        drawdown_pct = -100.0 * unrealized / (market_value - unrealized) if (market_value - unrealized) != 0 else 0.0
    except Exception:
        drawdown_pct = 0.0
    if drawdown_pct > 0 and drawdown_pct >= config.MAX_DRAWDOWN_PERCENT:
        log_warn("Drawdown %.2f%% >= max %.2f%% -> forcing exit", drawdown_pct, config.MAX_DRAWDOWN_PERCENT)
        sell_all(client, symbol)
        return
    if config.MAX_POSITION_AGE_HOURS and config.MAX_POSITION_AGE_HOURS > 0:
        try:
            orders = client.get_orders(status='closed', symbols=[symbol], limit=50, nested=False)
            last_fill = None
            for o in orders:
                if getattr(o, "side", "").lower() == "buy" and getattr(o, "filled_at", None):
                    t = getattr(o, "filled_at")
                    if isinstance(t, str):
                        t = dt.datetime.fromisoformat(t.replace("Z", "+00:00"))
                    if last_fill is None or t > last_fill:
                        last_fill = t
            if last_fill:
                age_hours = (now_utc() - last_fill).total_seconds() / 3600.0
                if age_hours >= config.MAX_POSITION_AGE_HOURS:
                    log_warn("Position age %.2fh >= %.2f -> forcing exit", age_hours, config.MAX_POSITION_AGE_HOURS)
                    sell_all(client, symbol)
                return
        except Exception:
            LOG.debug("Could not determine position age")
    if config.DAILY_LOSS_LIMIT_USD and config.DAILY_LOSS_LIMIT_USD > 0:
        today_loss = ledger_today_loss(config.PNL_LEDGER_PATH)
        if today_loss >= config.DAILY_LOSS_LIMIT_USD:
            log_warn("Daily loss limit reached ($%.2f >= $%.2f). Halting and exiting positions.", today_loss, config.DAILY_LOSS_LIMIT_USD)
            sell_all(client, symbol)
            return

# Buy & Sell flows
def buy_flow(client: REST, symbol: str, last_price: float, max_cap_usd: float, confidence: float, base_frac: float, base_tp: float, base_sl: float, dynamic_enabled: bool, fixed_usd_override: float = 0.0, interval_seconds: Optional[int] = None):
    try:
        if has_open_order(client, symbol, "buy"):
            return False, "duplicate buy order exists"
        pos = get_position(client, symbol)
        current_value = float(pos["market_value"]) if pos else 0.0
        if current_value >= max_cap_usd - 1e-9:
            return False, "at/above cap"
        remaining = max(0.0, max_cap_usd - current_value)

        vol_pct = 0.0
        try:
            closes_for_vol = fetch_closes(client, symbol, int(config.DEFAULT_INTERVAL_SECONDS), config.VOLATILITY_WINDOW)
            vol_pct = pct_stddev(closes_for_vol)
            if vol_pct >= config.VOLATILITY_PCT_THRESHOLD:
                return False, "volatility too high"
        except Exception:
            LOG.debug("volatility check failed")

        if confidence <= 0 or abs(confidence) < config.MIN_CONFIDENCE_TO_TRADE:
            return False, f"confidence {confidence:.4f} below min {config.MIN_CONFIDENCE_TO_TRADE}"

        if dynamic_enabled:
            tp, sl, frac = adjust_runtime_params(base_tp, base_sl, base_frac, confidence, vol_pct)
            # Optional risky overlay: nudge TP/SL and size when expected daily justifies it
            try:
                if config.RISKY_MODE_ENABLED:
                    use_secs = int(interval_seconds or config.DEFAULT_INTERVAL_SECONDS)
                    closes_for_proj = fetch_closes(client, symbol, use_secs, max(config.LONG_WINDOW + 50, 200))
                    proj = simulate_signals_and_projection(closes_for_proj, use_secs, override_tp_pct=tp, override_sl_pct=sl, override_trade_frac=frac, override_cap_usd=max_cap_usd) if closes_for_proj else None
                    if proj:
                        exp_daily = float(proj.get("expected_daily_usd", 0.0))
                        trades_per_day = float(proj.get("expected_trades_per_day", 0.0))
                        vol_gate = config.VOLATILITY_PCT_THRESHOLD * config.RISKY_VOL_MULTIPLIER
                        tpd_gate = config.SUGGESTION_MAX_TRADES_PER_DAY * config.RISKY_TRADES_PER_DAY_MULTIPLIER
                        if exp_daily >= config.RISKY_EXPECTED_DAILY_USD_MIN and vol_pct <= vol_gate and trades_per_day <= max(1.0, tpd_gate):
                            tp = min(config.MAX_TAKE_PROFIT_PERCENT, tp * config.RISKY_TP_MULT)
                            sl = min(config.MAX_STOP_LOSS_PERCENT, sl * config.RISKY_SL_MULT)
                            frac = min(config.RISKY_MAX_FRAC_CAP, max(config.MIN_TRADE_SIZE_FRAC, frac * config.RISKY_SIZE_MULT))
            except Exception:
                pass
            # Profitability gate: if expected daily is negative with current params, skip buy
            try:
                use_secs = int(interval_seconds or config.DEFAULT_INTERVAL_SECONDS)
                closes_for_gate = fetch_closes(client, symbol, use_secs, max(config.LONG_WINDOW + 50, 200))
                proj_gate = simulate_signals_and_projection(closes_for_gate, use_secs, override_tp_pct=tp, override_sl_pct=sl, override_trade_frac=frac, override_cap_usd=max_cap_usd) if closes_for_gate else None
                if proj_gate and float(proj_gate.get("expected_daily_usd", 0.0)) <= 0.0:
                    return False, "expected_daily <= 0 (profitability gate)"
            except Exception:
                pass
            qty = compute_order_qty_from_remaining(remaining, last_price, confidence, frac, fixed_usd=fixed_usd_override, scale_with_confidence=True)
        else:
            tp, sl, frac = base_tp, base_sl, base_frac
            qty = compute_order_qty_from_remaining(remaining, last_price, confidence, frac, fixed_usd=fixed_usd_override, scale_with_confidence=False)
        if qty <= 0:
            return False, "computed qty <= 0"

        order = place_market_order(client, symbol, qty, side="buy")
        filled_qty = float(getattr(order, "filled_qty", 0) or 0)
        avg_fill_price = float(getattr(order, "filled_avg_price", 0) or last_price)
        if filled_qty <= 0:
            return False, "buy did not fill"

        submit_tp_sl(client, symbol, filled_qty, avg_fill_price, tp, sl)
        log_info("Bought %.6f @ %.4f tp=%.2f sl=%.2f frac=%.3f vol=%.4f", filled_qty, avg_fill_price, tp, sl, frac, vol_pct)
        return True, f"bought {filled_qty} @ {avg_fill_price}"
    except TimeoutError as te:
        log_exc("buy timeout: %s", te)
        return False, "buy timeout"
    except Exception:
        log_exc("buy flow exception")
        return False, "buy exception"

def sell_flow(client: REST, symbol: str, confidence: Optional[float] = None):
    pos = get_position(client, symbol)
    if not pos:
        return False, "no position"
    qty = float(pos.get("qty", 0.0))
    if qty <= 0:
        return False, "qty <= 0"
    sell_qty = qty
    try:
        if config.SELL_PARTIAL_ENABLED and confidence is not None:
            frac = min(1.0, max(0.0, abs(confidence)))
            sell_qty = max(0.0, math.floor(qty * frac * 1_000_000) / 1_000_000)
            if sell_qty <= 0:
                return False, "sell fraction <= 0"
    except Exception:
        LOG.debug("sell partial calc failed")

    try:
        order = place_market_order(client, symbol, sell_qty, side="sell")
        ledger_add_realized(config.PNL_LEDGER_PATH, now_utc().isoformat(), 0.0)
        log_info("Sold qty=%.6f", sell_qty)
        return True, f"sold {sell_qty}"
    except Exception:
        log_exc("sell flow failed")
        return False, "sell exception"

# Interval suggestion & daily projection
def simulate_signals_and_projection(
    closes: List[float],
    interval_seconds: int,
    override_tp_pct: Optional[float] = None,
    override_sl_pct: Optional[float] = None,
    override_trade_frac: Optional[float] = None,
    override_cap_usd: Optional[float] = None,
):
    # simulate crossovers across 'closes' and attempt to estimate:
    # - expected signals per bar
    # - win ratio estimate by looking ahead up to lookahead bars for TP or SL
    signals = []
    for i in range(config.LONG_WINDOW + 1, len(closes) - 1):
        window = closes[: i + 1]
        act = decide_action(window, config.SHORT_WINDOW, config.LONG_WINDOW)
        if act in ("buy", "sell"):
            signals.append((i, act))
    bars_per_day = max(1.0, (390.0 * 60.0) / max(1.0, interval_seconds))
    expected_trades_per_day = (len(signals) / max(1.0, len(closes))) * bars_per_day

    # estimate win rate: for each signal, look forward up to lookahead bars and check if price crosses TP before SL
    lookahead = min(50, int(60 * 60 / max(1, interval_seconds)))  # up to 1 hour
    wins = 0
    trials = 0
    for idx, act in signals:
        entry = closes[idx]
        tp_pct = (override_tp_pct if override_tp_pct is not None else config.TAKE_PROFIT_PERCENT)
        sl_pct = (override_sl_pct if override_sl_pct is not None else config.STOP_LOSS_PERCENT)
        tp = entry * (1.0 + float(tp_pct) / 100.0)
        sl = entry * (1.0 - float(sl_pct) / 100.0)
        hit = None
        for j in range(idx + 1, min(len(closes), idx + lookahead + 1)):
            price = closes[j]
            if price >= tp:
                hit = "tp"
                break
            if price <= sl:
                hit = "sl"
                break
        if hit:
            trials += 1
            if hit == "tp":
                wins += 1
    win_rate = (wins / trials) if trials > 0 else 0.5  # fallback 50%
    trade_frac = float(override_trade_frac) if override_trade_frac is not None else float(config.TRADE_SIZE_FRAC_OF_CAP)
    cap_usd = float(override_cap_usd) if override_cap_usd is not None else float(config.MAX_CAP_USD)
    avg_trade_size = trade_frac * cap_usd
    tp_eval = float(override_tp_pct) if override_tp_pct is not None else float(config.TAKE_PROFIT_PERCENT)
    sl_eval = float(override_sl_pct) if override_sl_pct is not None else float(config.STOP_LOSS_PERCENT)
    expected_return_per_trade = ((tp_eval / 100.0) * win_rate) - ((sl_eval / 100.0) * (1 - win_rate))
    expected_daily = expected_return_per_trade * expected_trades_per_day * avg_trade_size
    min_trials = getattr(config, "SUGGESTION_MIN_TRIALS", 30)
    sufficient = trials >= max(1, int(min_trials))
    return {"expected_trades_per_day": expected_trades_per_day, "win_rate": win_rate, "expected_daily_usd": expected_daily, "signals": len(signals), "trials": trials, "wins": wins, "sufficient_samples": sufficient}

# Interval suggestion via public history across multiple candidate intervals
def suggest_interval_from_public_history(client: REST, symbol: str, candidates_seconds: List[int], bars: int):
    results = []  # (sec, trades_per_day, expected_daily_usd)
    for secs in candidates_seconds:
        try:
            closes = fetch_closes(client, symbol, int(secs), bars)
            if not closes or len(closes) < max(config.LONG_WINDOW + 10, 30):
                continue
            sim = simulate_signals_and_projection(closes, int(secs))
            results.append((int(secs), float(sim["expected_trades_per_day"]), float(sim["expected_daily_usd"])) )
        except Exception:
            continue
    if not results:
        return None, []
    # Pick the interval that maximizes expected_daily_usd (profit-driven)
    chosen = max(results, key=lambda r: r[2])
    return chosen[0], results

# Market hours helper
def in_market_hours(client: REST) -> bool:
    if not config.ENABLE_MARKET_HOURS_ONLY:
        return True
    try:
        clock = client.get_clock()
        return getattr(clock, "is_open", True)
    except Exception:
        LOG.debug("could not fetch market clock")
        return True

# Market open helpers
def seconds_until_next_open(client: REST) -> float:
    try:
        clock = client.get_clock()
        if getattr(clock, "is_open", False):
            return 0.0
        next_open = getattr(clock, "next_open", None)
        if next_open is None:
            return 300.0
        t = next_open
        if isinstance(t, str):
            t = dt.datetime.fromisoformat(t.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=pytz.UTC)
        now = now_utc()
        delta = (t - now).total_seconds()
        if delta < 0:
            return 0.0
        # clamp overly large waits (we'll re-check periodically anyway)
        return float(delta)
    except Exception:
        return 300.0

def sleep_until_market_open(client: REST, min_check_seconds: float = 60.0, max_check_seconds: float = 900.0):
    # Single-chunk strategy with 5-minute pre-wake; exit if far from open in task mode
    if in_market_hours(client):
        return
    # Market is closed; release any keep-awake request until pre-open window
    try:
        prevent_system_sleep(False)
    except Exception:
        pass
    try:
        secs_total = seconds_until_next_open(client)
    except Exception:
        secs_total = 300.0
    # Always wait until pre-open/open instead of exiting (avoid restart loops)
    prewake = max(0.0, secs_total - 300.0)
    if prewake > 0:
        log_info("Market closed. Sleeping %.0fs until 5m before open", prewake)
        time.sleep(prewake)
    # If market opened while we were sleeping to prewake, skip remaining sleep
    if in_market_hours(client):
        return
    # Pre-open window: keep system awake and finish remaining sleep until open
    try:
        remain = max(0.0, seconds_until_next_open(client))
    except Exception:
        remain = 300.0
    if remain <= 0.0 or in_market_hours(client):
        return
    log_info("Pre-open window: sleeping %.0fs until open (keeping system awake)", remain)
    prevent_system_sleep(True)
    time.sleep(remain)
    prevent_system_sleep(False)

# Main
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--time", type=float, default=(config.DEFAULT_INTERVAL_SECONDS / 3600.0), help="Polling interval hours (e.g., 2 => 2h, .001 => ~3.6s)")
    parser.add_argument("-s", "--symbol", type=str, default=config.DEFAULT_TICKER, help="Ticker symbol")
    parser.add_argument("-m", "--max-cap", type=float, default=config.MAX_CAP_USD, help="Max cap per symbol (USD)")
    parser.add_argument("--tp", type=float, default=config.TAKE_PROFIT_PERCENT, help="Override take-profit percent")
    parser.add_argument("--sl", type=float, default=config.STOP_LOSS_PERCENT, help="Override stop-loss percent")
    parser.add_argument("--frac", type=float, default=config.TRADE_SIZE_FRAC_OF_CAP, help="Override base trade fraction of cap")
    parser.add_argument("--fixed-usd", type=float, default=config.FIXED_TRADE_USD, help="Override fixed USD per buy (0 to disable)")
    parser.add_argument("--no-dynamic", action="store_true", help="Disable dynamic sizing/TP/SL; use provided base values")
    # remove per-buy override to keep sizing purely dynamic
    parser.add_argument("--go-live", action="store_true", help="Enable live trading (requires CONFIRM_GO_LIVE=YES)")
    parser.add_argument("--allow-missing-keys", action="store_true", help="Debug: allow missing API keys")
    args = parser.parse_args()

    # no per-buy override; sizing is dynamic

    try:
        if args.go_live:
            # require CONFIRM_GO_LIVE=YES env var
            if not config.wants_live_mode(cli_flag_go_live=True):
                log_warn("Attempted to go live without CONFIRM_GO_LIVE=YES. Aborting.")
                return 3
            log_warn("LIVE MODE requested by user. Make sure ALPACA_BASE_URL and keys are live.")
        client = make_client(allow_missing=args.allow_missing_keys, go_live=args.go_live)
    except Exception:
        log_exc("Failed to create Alpaca client")
        return 2

    # Convert hours to seconds, but allow very small hour values for sub-minute polling
    interval = max(1.0, float(args.time) * 3600.0)
    symbol = args.symbol.upper()
    max_cap = float(args.max_cap)

    log_info("Starting bot for %s interval=%.1fs (%.3fh) max_cap=$%.2f", symbol, interval, interval/3600.0, max_cap)
    try:
        enforce_log_max_lines(100)
    except Exception:
        pass

    # Base parameters from CLI (outside loop)
    base_tp = float(args.tp)
    base_sl = float(args.sl)
    base_frac = float(args.frac)
    fixed_usd = float(args.fixed_usd)
    dynamic_enabled = (not args.no_dynamic)

    # ensure ledger
    if not os.path.exists(config.PNL_LEDGER_PATH):
        write_json(config.PNL_LEDGER_PATH, {"records": []})

    # initial interval suggestion + daily projection based on history
    try:
        # Use a larger bar window for robust interval suggestion
        bars = max(config.INTERVAL_SUGGESTION_WINDOW_BARS * 3, config.LONG_WINDOW + 200)
        # Build public-history candidates from 23s up to 6.5h using log spacing
        min_secs = 23
        max_secs = int(6.5 * 3600)
        steps = 20
        ratio = (max_secs / float(min_secs)) ** (1.0 / max(1, steps - 1))
        cand = []
        cur = float(min_secs)
        for _ in range(steps):
            cand.append(int(max(5, round(cur))))
            cur *= ratio
        # Ensure we include some common granularities and the current interval
        cand.extend([30, 45, 60, 90, 120, 180, 300, 600, 900, 1800, 3600, 7200, 14400, max_secs, int(max(5, int(interval)))])
        # Snap all candidates to supported sizes to avoid duplicates across non-supported intervals
        snapped = [snap_interval_to_supported_seconds(x) for x in cand]
        candidates = sorted(set(snapped))
        suggested, details = suggest_interval_from_public_history(client, symbol, candidates, bars)
        if details:
            # Log summary from public history at current and nearby intervals
            cur = next((d for d in details if d[0] == int(interval)), None)
            if cur:
                # Re-simulate quickly to get sample stats for the current interval logging
                try:
                    closes_cur = fetch_closes(client, symbol, int(cur[0]), bars)
                    sim_cur = simulate_signals_and_projection(closes_cur, int(cur[0])) if closes_cur else None
                except Exception:
                    sim_cur = None
                if sim_cur and not sim_cur.get("sufficient_samples", True):
                    log_warn("Projection (public hist @%ds): insufficient samples (trials=%s)", cur[0], sim_cur.get("trials", 0))
                else:
                    log_info("Projection (public hist @%ds): trades/day=%.2f expected_daily=$%.2f (trials=%s)", cur[0], cur[1], cur[2], (sim_cur or {}).get("trials", "-"))
            best = next((d for d in details if d[0] == suggested), None)
            if best and best[0] != int(interval):
                # Also show as hours in decimal (e.g., 0.0065)
                best_hours = best[0] / 3600.0
                # Re-simulate best for sample stats
                try:
                    closes_best = fetch_closes(client, symbol, int(best[0]), bars)
                    sim_best = simulate_signals_and_projection(closes_best, int(best[0])) if closes_best else None
                except Exception:
                    sim_best = None
                if sim_best and not sim_best.get("sufficient_samples", True):
                    log_warn("Suggested interval (public history): %ds (%.4fh) (current %ds) — insufficient samples (trials=%s)", best[0], best_hours, int(interval), sim_best.get("trials", 0))
                else:
                    log_warn("Suggested interval (public history): %ds (%.4fh) (current %ds) — expected_daily=$%.2f (trials=%s)", best[0], best_hours, int(interval), best[2], (sim_best or {}).get("trials", "-"))
    except Exception:
        log_exc("Interval suggestion/projection failed (non-fatal)")

    # main loop
    try:
        while True:
            # Always enforce truncation to keep bot.log within cap, even when file writes are via Tee
            try:
                enforce_log_max_lines(100)
            except Exception:
                pass
            start = now_utc()

            if not in_market_hours(client):
                sleep_until_market_open(client)
                continue
            # Ensure the system doesn't sleep during active market hours
            prevent_system_sleep(True)

            try:
                bars_needed = max(config.LONG_WINDOW + 2, config.VOLATILITY_WINDOW + 2)
                closes = fetch_closes(client, symbol, int(interval), bars_needed)
                if not closes or len(closes) < config.LONG_WINDOW:
                    log_info("Not enough bars (%d) for signal (need %d). Sleeping %.1fs.", len(closes), config.LONG_WINDOW, interval)
                    time.sleep(interval)
                    continue
            except Exception:
                backoff = min(60, interval * 2)
                log_exc("Market data fetch error; sleeping %.1fs", backoff)
                time.sleep(backoff)
                continue

            action = decide_action(closes, config.SHORT_WINDOW, config.LONG_WINDOW)
            confidence = compute_confidence(closes)
            last_price = float(closes[-1])

            log_info("Decision=%s confidence=%.4f price=%.4f short_ma=%.4f long_ma=%.4f", action, confidence, last_price, sma(closes, config.SHORT_WINDOW) or 0.0, sma(closes, config.LONG_WINDOW) or 0.0)

            try:
                enforce_safety(client, symbol)
            except Exception:
                log_exc("Safety enforcement error (non-fatal)")

            if action == "buy":
                ok, msg = buy_flow(client, symbol, last_price, max_cap, max(0.0, confidence), base_frac, base_tp, base_sl, dynamic_enabled, fixed_usd_override=fixed_usd, interval_seconds=int(interval))
                log_info("Buy result: ok=%s msg=%s", ok, msg)
            elif action == "sell":
                ok, msg = sell_flow(client, symbol, confidence=min(0.0, confidence))
                log_info("Sell result: ok=%s msg=%s", ok, msg)
            else:
                LOG.debug("No trade this interval")

            try:
                p = get_position(client, symbol)
                if p:
                    log_info("Snapshot: qty=%.6f mkt=%.2f unreal_pl=%.2f", p["qty"], p["market_value"], p["unrealized_pl"])
            except Exception:
                LOG.debug("Snapshot fetch failed")

            # Final trim at end of loop to ensure cap holds even after large write bursts
            try:
                enforce_log_max_lines(100)
            except Exception:
                pass
            elapsed = (now_utc() - start).total_seconds()
            to_sleep = max(0.0, interval - elapsed)
            LOG.debug("Loop took %.2fs sleeping %.2fs", elapsed, to_sleep)
            time.sleep(to_sleep)
    except KeyboardInterrupt:
        log_info("KeyboardInterrupt - exiting")
        try:
            prevent_system_sleep(False)
        except Exception:
            pass
        return 0
    except Exception:
        log_exc("Unhandled exception - exiting")
        try:
            prevent_system_sleep(False)
        except Exception:
            pass
        return 1

if __name__ == "__main__":
    sys.exit(main())
