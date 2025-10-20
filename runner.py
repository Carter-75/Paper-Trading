import argparse
import datetime as dt
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import requests
import math
import pytz
import re

try:
    from alpaca_trade_api import REST, REST as AlpacaREST
    from alpaca_trade_api.rest import APIError as AlpacaAPIError
except Exception:  # pragma: no cover
    # Defer import error until runtime so requirements can be installed first
    REST = None  # type: ignore
    AlpacaREST = None  # type: ignore
    class AlpacaAPIError(Exception):  # type: ignore
        pass

import config as cfg


def utc_now() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)


_PROCESS_HEADER_LINE: Optional[str] = None
_KEEP_AWAKE_ENABLED: bool = False


def _build_process_header_line() -> str:
    started = utc_now().replace(tzinfo=dt.timezone.utc)
    pid = os.getpid()
    # ISO 8601 in Zulu for clarity
    ts = started.strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"=== Bot process start: {ts} pid={pid} ===\n"


def ensure_process_header_printed() -> None:
    global _PROCESS_HEADER_LINE
    if _PROCESS_HEADER_LINE is None:
        _PROCESS_HEADER_LINE = _build_process_header_line()
    # Print once per process at startup
    try:
        sys.stdout.write(_PROCESS_HEADER_LINE)
        sys.stdout.flush()
    except Exception:
        pass


def _is_header_line(s: str) -> bool:
    return s.strip().startswith("=== Bot process start:") and " pid=" in s


def trim_log_at_market_close(path: str, max_lines: int = 100) -> None:
    """Trim the log to exactly max_lines at market close, with header rules:

    - Keep headers for any process that still has body lines in the kept block.
    - Insert missing headers just before that process's first kept body line.
    - If insertion exceeds max_lines, drop earliest lines until length == max_lines.
    - Remove orphan headers (no body lines following them within the kept block).
    """
    try:
        if not path or max_lines <= 0 or not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        total = len(lines)
        if total <= max_lines:
            return

        # Helper: map each absolute line index to its segment header absolute index
        header_abs_indices: List[int] = [i for i, s in enumerate(lines) if _is_header_line(s)]
        def seg_header_for_index(idx: int) -> int:
            # find last header index <= idx
            lo = -1
            for h in header_abs_indices:
                if h <= idx:
                    lo = h
                else:
                    break
            return lo

        # Start with the last max_lines lines
        start_abs = total - max_lines
        end_abs = total
        kept = list(lines[start_abs:end_abs])

        # Determine which segments (by header abs idx) have body lines in kept
        needed_headers: List[int] = []
        first_pos_in_kept: dict[int, int] = {}
        for rel_i, abs_i in enumerate(range(start_abs, end_abs)):
            s = lines[abs_i]
            if _is_header_line(s):
                continue
            seg_h = seg_header_for_index(abs_i)
            if seg_h >= 0:
                if seg_h not in first_pos_in_kept:
                    first_pos_in_kept[seg_h] = rel_i
                if seg_h not in needed_headers:
                    needed_headers.append(seg_h)

        # Insert missing headers for represented segments
        # Do insertions in order of first position to keep relative positions sensible
        for seg_h in sorted(needed_headers, key=lambda h: first_pos_in_kept.get(h, 0)):
            hdr_line = lines[seg_h]
            # Is header already present in kept?
            if any(_is_header_line(x) and x.strip() == hdr_line.strip() for x in kept):
                continue
            insert_at = first_pos_in_kept.get(seg_h, 0)
            kept.insert(insert_at, hdr_line)
            # If we exceeded max_lines, drop from the front
            while len(kept) > max_lines:
                kept.pop(0)
                # Adjust insertion positions for subsequent headers
                for k in list(first_pos_in_kept.keys()):
                    if first_pos_in_kept[k] > 0:
                        first_pos_in_kept[k] -= 1

        # Remove orphan headers (headers with no body lines until next header)
        def drop_orphan_headers(buf: List[str]) -> List[str]:
            hidx = [i for i, s in enumerate(buf) if _is_header_line(s)]
            to_drop_local = set()
            for p, i in enumerate(hidx):
                j = hidx[p + 1] if p + 1 < len(hidx) else len(buf)
                has_body = any((not _is_header_line(buf[t])) and buf[t].strip() for t in range(i + 1, j))
                if not has_body:
                    to_drop_local.add(i)
            if not to_drop_local:
                return buf
            return [s for i, s in enumerate(buf) if i not in to_drop_local]

        kept = drop_orphan_headers(kept)

        # Add ellipsis after headers of segments that were truncated (older content removed)
        # Build map of header text -> absolute index
        header_text_to_abs: dict[str, int] = {lines[h].strip(): h for h in header_abs_indices}

        def build_segments(buf: List[str]) -> List[Tuple[int, int, str]]:
            hidx = [i for i, s in enumerate(buf) if _is_header_line(s)]
            segs: List[Tuple[int, int, str]] = []
            for p, i in enumerate(hidx):
                j = hidx[p + 1] if p + 1 < len(hidx) else len(buf)
                segs.append((i, j, buf[i].strip()))
            return segs

        segs = build_segments(kept)
        # Determine newest segment index (last header in kept)
        newest_seg_idx = len(segs) - 1 if segs else -1

        # Insert ellipsis where needed; if no room, drop older segment entirely
        i_offset = 0
        for si, (start_i, end_i, hdr_text) in enumerate(segs):
            start_i += i_offset
            end_i += i_offset
            abs_header = header_text_to_abs.get(hdr_text, -1)
            truncated = abs_header >= 0 and abs_header < start_abs
            if not truncated:
                continue
            # Only add ellipsis if not already present as first body line
            insert_pos = start_i + 1
            already = insert_pos < len(kept) and kept[insert_pos].strip() == "..."
            if already:
                continue
            if len(kept) + 1 <= max_lines:
                kept.insert(insert_pos, "...\n")
                i_offset += 1
            else:
                # No room to add ellipsis; drop this entire older segment unless it's the newest
                if si != newest_seg_idx:
                    del kept[start_i:end_i]
                    i_offset -= (end_i - start_i)
                # If it's the newest, skip adding ellipsis

        # Final guard: ensure length <= max_lines by trimming front if needed
        if len(kept) > max_lines:
            kept = kept[-max_lines:]
        # If shorter than max_lines, leave as-is

        with open(path, "w", encoding="utf-8", errors="ignore") as f:
            f.writelines(kept)
    except Exception:
        # Best effort; don't interrupt trading
        pass


def trim_log_if_overflow(path: str, threshold_lines: int = 1000, keep_lines: int = 100) -> None:
    """If the log exceeds threshold_lines, trim to keep_lines using header-preserving logic."""
    try:
        if not path or threshold_lines <= 0 or keep_lines <= 0 or not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if len(lines) <= threshold_lines:
            return
        # Reuse market-close trimming logic to preserve headers for segments that remain
        trim_log_at_market_close(path, max_lines=keep_lines)
    except Exception:
        pass


def to_float(value: object, default: float = 0.0) -> float:
    """Robust float conversion handling None and non-numeric strings."""
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip()
        if s == "" or s.lower() in {"none", "null", "nan"}:
            return default
        return float(s)
    except Exception:
        return default


def _set_keep_awake_windows(enable: bool) -> None:
    """Use Windows SetThreadExecutionState to prevent system sleep while enabled.

    ES_CONTINUOUS | ES_SYSTEM_REQUIRED keeps the system awake but allows display to turn off.
    """
    try:
        if sys.platform != "win32":
            return
        import ctypes  # local import to avoid non-Windows issues

        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def set_keep_awake(enable: bool) -> None:
    """Idempotently toggle system sleep prevention (Windows only)."""
    global _KEEP_AWAKE_ENABLED
    try:
        enable_bool = bool(enable)
        if enable_bool == _KEEP_AWAKE_ENABLED:
            return
        _set_keep_awake_windows(enable_bool)
        _KEEP_AWAKE_ENABLED = enable_bool
        if enable_bool:
            print("Keep-awake: enabled (market hours)")
        else:
            print("Keep-awake: disabled (market closed)")
    except Exception:
        pass


class RateLimitError(Exception):
    def __init__(self, wait_seconds: int, message: str = ""):
        super().__init__(message or f"Rate limited. Wait {wait_seconds}s")
        self.wait_seconds = int(max(1, wait_seconds))


def compute_ema(values: List[float], span: int) -> float:
    if span <= 0:
        raise ValueError("EMA span must be positive")
    if not values:
        raise ValueError("Cannot compute EMA of empty sequence")
    k = 2.0 / (span + 1.0)
    ema: float = values[0]
    for price in values[1:]:
        ema = price * k + ema * (1.0 - k)
    return ema


def compute_percent_returns(values: List[float]) -> List[float]:
    if not values or len(values) < 2:
        return []
    returns: List[float] = []
    prev = values[0]
    for v in values[1:]:
        try:
            if prev > 0:
                returns.append((v - prev) / prev)
        except Exception:
            pass
        prev = v
    return returns


def compute_volatility(values: List[float]) -> float:
    """Rough volatility as std dev of percent returns."""
    rets = compute_percent_returns(values)
    try:
        n = len(rets)
        if n < 2:
            return 0.0
        mean = sum(rets) / n
        var = sum((r - mean) * (r - mean) for r in rets) / (n - 1)
        if var < 0:
            return 0.0
        import math as _math
        return float(_math.sqrt(var))
    except Exception:
        return 0.0


def compute_trend_confidence(values: List[float], short_window: int, long_window: int) -> float:
    if len(values) < max(short_window, long_window):
        return 0.0
    short_ema = compute_ema(values[-short_window:], short_window)
    long_ema = compute_ema(values[-long_window:], long_window)
    spread = short_ema - long_ema
    if long_ema <= 0:
        return 0.0
    ratio = spread / long_ema
    # Map ratio into [0,1] band; >1% -> strong
    if ratio <= 0:
        return 0.0
    if ratio >= 0.01:
        return 1.0
    return max(0.0, min(1.0, ratio / 0.01))


def decide_action(closes: List[float], short_window: int, long_window: int) -> str:
    if len(closes) >= max(short_window, long_window):
        short_ema = compute_ema(closes[-short_window:], short_window)
        long_ema = compute_ema(closes[-long_window:], long_window)
        return "buy" if short_ema > long_ema else "sell"
    if len(closes) >= 2:
        return "buy" if closes[-1] > closes[-2] else "sell"
    return "buy"


def compute_buy_fraction(closes: List[float], short_window: int, long_window: int) -> float:
    """Return fraction of remaining-to-cap to buy based on EMA crossover confidence.

    0.0 => hold; 0.1 => buy 10% of remaining; 0.5 => buy 50%; 1.0 => buy the computed dynamic amount
    """
    if len(closes) < max(short_window, long_window):
        return 0.0
    short_ema = compute_ema(closes[-short_window:], short_window)
    long_ema = compute_ema(closes[-long_window:], long_window)
    spread = short_ema - long_ema
    if spread <= 0:
        return 0.0
    # Normalize by long EMA to get a relative confidence
    ratio = spread / max(1e-9, long_ema)
    if ratio < 0.001:  # <0.10%
        return 0.0
    if ratio < 0.003:  # <0.30%
        return 0.10
    if ratio < 0.01:   # <1.00%
        return 0.50
    return 1.0


def _estimate_fee_fraction(interval_hours: float) -> float:
    """Estimated all-in trading cost fraction per round-trip considering short intervals.

    Paper trading is often zero-commission, but we account for slippage/fees.
    Heuristic: 6 bps baseline plus extra for very short intervals.
    """
    try:
        base = 0.0006
        if interval_hours <= 0:
            return base
        import math as _math
        # Extra cost rises as intervals shrink below 1h
        extra = 0.001 * max(0.0, (1.0 / max(1e-6, interval_hours)) ** 0.3 - 1.0)
        return min(0.01, base + extra)
    except Exception:
        return 0.001


def _dynamic_windows(short_w: int, long_w: int, volatility: float, overrides: Tuple[bool, bool]) -> Tuple[int, int]:
    """Adjust EMA windows when not overridden: higher vol -> longer windows for stability."""
    short_overridden, long_overridden = overrides
    s, l = int(short_w), int(long_w)
    if not short_overridden or not long_overridden:
        # volatility ~ 0.0..0.05 typical per-hour; scale modestly
        vol_norm = max(0.0, min(1.5, volatility / 0.02))  # around 2% per interval as reference
        scale_up = 1.0 + 0.5 * vol_norm  # up to +50%
        scale_down = 1.0 - 0.3 * max(0.0, 1.0 - vol_norm)  # mild tightening when very low vol
        if not short_overridden:
            s = max(2, int(round(s * (scale_up if volatility > 0.02 else scale_down))))
        if not long_overridden:
            l = max(s + 1, int(round(l * (scale_up if volatility > 0.02 else scale_down))))
    return s, l


def compute_dynamic_trade_params(
    closes: List[float],
    interval_hours: float,
    current_value: float,
    remaining_to_cap: float,
    base_tp_pct: float,
    base_sl_pct: float,
    base_trail_pct: float,
    short_w: int,
    long_w: int,
    tp_overridden: bool,
    sl_overridden: bool,
    trail_overridden: bool,
    short_overridden: bool,
    long_overridden: bool,
) -> Tuple[float, float, float, float, int, int, str]:
    """Return (per_buy_usd, tp_pct, sl_pct, trail_pct, eff_short, eff_long, rationale).

    Sizing scales to remaining_to_cap based on confidence (increase) and volatility (decrease).
    If a per-buy override is provided, use it (clamped to remaining_to_cap).
    """
    vol = compute_volatility(closes)
    eff_short, eff_long = _dynamic_windows(short_w, long_w, vol, (short_overridden, long_overridden))
    confidence = compute_trend_confidence(closes, eff_short, eff_long)
    fee_frac = _estimate_fee_fraction(interval_hours)

    # Position sizing: fraction of remaining_to_cap
    vol_norm = max(0.0, min(1.0, vol / 0.02))  # ~2% per interval reference
    size_fraction = 0.15 + 0.85 * confidence - 0.5 * vol_norm
    size_fraction = max(0.05, min(1.0, size_fraction))
    per_buy_dyn = min(remaining_to_cap, remaining_to_cap * size_fraction)
    # Avoid tiny trades; enforce $1 only if there is room under cap
    if per_buy_dyn < 1.0 and remaining_to_cap >= 1.0:
        per_buy_dyn = 1.0

    # Risk params
    tp_pct = base_tp_pct
    sl_pct = base_sl_pct
    trail_pct = base_trail_pct
    if not sl_overridden:
        # High vol -> wider stop; strong trend -> tighter stop
        sl_scale = 1.0 + 1.5 * vol_norm - 0.5 * confidence
        sl_pct = max(0.5, round(max(0.1, base_sl_pct) * sl_scale, 2))
    if not tp_overridden:
        # High vol -> allow wider take; strong trend -> slightly tighter TP to realize gains sooner
        tp_scale = 1.0 + 1.0 * vol_norm - 0.2 * confidence
        tp_pct = max(0.5, round(max(0.1, base_tp_pct) * tp_scale, 2))
    if not trail_overridden:
        # Trail follows similar logic; ensure reasonable bounds
        tr_scale = 1.0 + 1.0 * vol_norm - 0.3 * confidence
        trail_pct = max(0.0, round(max(0.0, base_trail_pct) * tr_scale, 2))

    rationale = (
        f"vol={vol:.4f} conf={confidence:.2f} fee~{fee_frac*100:.2f}% "
        f"size_frac={size_fraction:.2f} remaining=${remaining_to_cap:.2f} -> buy=${per_buy_dyn:.2f} "
        f"TP={tp_pct:.2f}% SL={sl_pct:.2f}% TR={trail_pct:.2f}% EMAs={eff_short}/{eff_long}"
    )
    return per_buy_dyn, tp_pct, sl_pct, trail_pct, eff_short, eff_long, rationale


def _polygon_rate_limit_wait_seconds(headers: dict) -> int:
    try:
        # Prefer X-RateLimit-Reset (unix epoch seconds)
        reset_raw = headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset")
        if reset_raw is not None:
            reset_epoch = int(float(str(reset_raw)))
            now_epoch = int(time.time())
            delta = reset_epoch - now_epoch
            if delta > 0:
                return min(delta, 3600)
        retry_raw = headers.get("Retry-After") or headers.get("retry-after")
        if retry_raw is not None:
            ra = int(float(str(retry_raw)))
            if ra > 0:
                return min(ra, 3600)
    except Exception:
        pass
    # Fallback to 5 minutes
    return 300


def polygon_fetch_hourly_closes(ticker: str, hours_back: int = 72) -> List[float]:
    if hours_back < 2:
        hours_back = 2
    now = utc_now()
    start = now - dt.timedelta(hours=hours_back)
    # Polygon aggregates v2 endpoint for hourly bars
    base_url = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour/{start}/{end}"
    url = base_url.format(
        ticker=ticker.upper(),
        start=start.strftime("%Y-%m-%d"),
        end=now.strftime("%Y-%m-%d"),
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
        "apiKey": cfg.POLYGON_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 429:
        wait_s = _polygon_rate_limit_wait_seconds(resp.headers)
        raise RateLimitError(wait_s, "Polygon rate limit reached")
    if resp.status_code != 200:
        raise RuntimeError(f"Polygon error {resp.status_code}: {resp.text}")
    data = resp.json()
    # Some Polygon responses may omit status or set non-OK while still including results.
    # Only raise if an explicit error is provided or HTTP failed (handled above).
    if isinstance(data, dict) and data.get("error"):
        err_msg = str(data.get("error"))
        if "rate" in err_msg.lower():
            wait_s = _polygon_rate_limit_wait_seconds(resp.headers)
            raise RateLimitError(wait_s, err_msg)
        raise RuntimeError(f"Polygon error: {err_msg}")
    results = data.get("results") or []
    closes: List[float] = [float(item["c"]) for item in results if "c" in item]
    return closes


def estimate_interval_efficiency(
    hourly_closes: List[float],
    current_interval_hours: float,
    notional_usd: float,
) -> Tuple[str, List[Tuple[str, float]]]:
    """Estimate daily return efficiency across intervals using volatility scaling.

    Returns (advice_line, [(label, est_net_per_day_usd), ...])
    """
    try:
        import math as _math
        # Hourly volatility as base
        vol_1h = compute_volatility(hourly_closes)
        market_hours = 6.5
        candidates = [
            ("15s", 15.0 / 3600.0),
            ("23s", 23.0 / 3600.0),
            ("30s", 30.0 / 3600.0),
            ("45s", 45.0 / 3600.0),
            ("1m", 1.0 / 60.0),
            ("2m", 2.0 / 60.0),
            ("5m", 5.0 / 60.0),
            ("10m", 10.0 / 60.0),
            ("15m", 15.0 / 60.0),
            ("30m", 30.0 / 60.0),
            ("45m", 45.0 / 60.0),
            ("1h", 1.0),
            ("90m", 1.5),
            ("2h", 2.0),
            ("3h", 3.0),
            ("4h", 4.0),
        ]
        estimates: List[Tuple[str, float]] = []
        best_label = ""
        best_net = -1e9
        for label, h in candidates:
            # Scale vol by sqrt(time)
            h_eff = max(1e-6, h)
            vol_h = vol_1h * _math.sqrt(h_eff / 1.0)
            trades_per_day = max(1.0, market_hours / h_eff)
            # Expected per-trade edge proportional to vol and signal reliability proxy
            reliability = max(0.1, min(0.9, 0.5 + 0.5 * min(1.0, vol_1h / 0.02)))
            exp_ret_per_trade = 0.25 * vol_h * reliability * notional_usd
            fee_frac = _estimate_fee_fraction(h_eff)
            fee_per_trade = fee_frac * notional_usd
            net_per_day = trades_per_day * max(0.0, exp_ret_per_trade - fee_per_trade)
            estimates.append((label, float(round(net_per_day, 2))))
            if net_per_day > best_net:
                best_net = net_per_day
                best_label = label
        current_label = next((lbl for lbl, h in candidates if abs(h - current_interval_hours) < 1e-9), None)
        # Build advice
        parts = ", ".join([f"{lbl} ~ ${val}/day" for lbl, val in estimates])
        advice = f"Interval efficiency: {parts}."
        if current_label is not None and best_label and best_label != current_label:
            advice += f" Consider switching from {current_label} to {best_label} for potentially higher net."
        return advice, estimates
    except Exception:
        return "", []


def choose_best_interval(hourly_closes: List[float], current_interval_hours: float, notional_usd: float) -> Tuple[float, str]:
    advice, estimates = estimate_interval_efficiency(hourly_closes, current_interval_hours, notional_usd)
    if not estimates:
        return current_interval_hours, advice
    # Pick max net/day
    best_label, best_net = max(estimates, key=lambda kv: kv[1])
    # Find current net
    current_entry = None
    for lbl, val in estimates:
        # map label to hours similar to estimate list
        label_to_hours = {
            "15s": 15.0 / 3600.0,
            "23s": 23.0 / 3600.0,
            "30s": 30.0 / 3600.0,
            "45s": 45.0 / 3600.0,
            "1m": 1.0 / 60.0,
            "2m": 2.0 / 60.0,
            "5m": 5.0 / 60.0,
            "10m": 10.0 / 60.0,
            "15m": 15.0 / 60.0,
            "30m": 30.0 / 60.0,
            "45m": 45.0 / 60.0,
            "1h": 1.0,
            "90m": 1.5,
            "2h": 2.0,
            "3h": 3.0,
            "4h": 4.0,
        }
        if abs(label_to_hours.get(lbl, -1.0) - current_interval_hours) < 1e-9:
            current_entry = (lbl, val)
            break
    if current_entry is None:
        return current_interval_hours, advice
    _, current_net = current_entry
    # Only switch if improvement is meaningful (>=20% and absolute gain >= $0.50/day)
    if best_net > current_net * 1.2 and (best_net - current_net) >= 0.5:
        # Translate best_label back to hours
        mapping = {
            "15s": 15.0 / 3600.0,
            "23s": 23.0 / 3600.0,
            "30s": 30.0 / 3600.0,
            "45s": 45.0 / 3600.0,
            "1m": 1.0 / 60.0,
            "2m": 2.0 / 60.0,
            "5m": 5.0 / 60.0,
            "10m": 10.0 / 60.0,
            "15m": 15.0 / 60.0,
            "30m": 30.0 / 60.0,
            "45m": 45.0 / 60.0,
            "1h": 1.0,
            "90m": 1.5,
            "2h": 2.0,
            "3h": 3.0,
            "4h": 4.0,
        }
        return mapping.get(best_label, current_interval_hours), advice
    return current_interval_hours, advice


def build_alpaca_client() -> AlpacaREST:
    if REST is None:
        raise RuntimeError(
            "alpaca_trade_api is not installed. Run: pip install -r requirements.txt"
        )
    base_url = cfg.ALPACA_BASE_URL or "https://paper-api.alpaca.markets"
    return REST(key_id=cfg.ALPACA_API_KEY, secret_key=cfg.ALPACA_API_SECRET, base_url=base_url)


def get_tsla_position(client: AlpacaREST, ticker: str) -> Optional[dict]:
    try:
        pos = client.get_position(ticker)
        # Convert to dict with fields we need
        return {
            "qty": to_float(getattr(pos, "qty", 0.0), 0.0),
            "market_value": to_float(getattr(pos, "market_value", 0.0), 0.0),
            "avg_entry_price": to_float(getattr(pos, "avg_entry_price", 0.0), 0.0),
        }
    except Exception:
        return None


def submit_notional_buy(client: AlpacaREST, ticker: str, notional_usd: float) -> Optional[str]:
    """Backward-compatible buy using static config TP/SL/trailing."""
    take_pct = float(getattr(cfg, "TAKE_PROFIT_PERCENT", 0.0) or 0.0)
    stop_pct = float(getattr(cfg, "STOP_LOSS_PERCENT", 0.0) or 0.0)
    trail_pct = float(getattr(cfg, "TRAILING_STOP_PERCENT", 0.0) or 0.0)
    return submit_dynamic_buy(client, ticker, notional_usd, take_pct, stop_pct, trail_pct)


def submit_dynamic_buy(
    client: AlpacaREST,
    ticker: str,
    notional_usd: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    trailing_pct: float,
) -> Optional[str]:
    order_kwargs = dict(
        symbol=ticker,
        side="buy",
        time_in_force="day",
        notional=round(float(notional_usd), 2),
    )
    tp = float(take_profit_pct or 0.0)
    sl = float(stop_loss_pct or 0.0)
    tr = float(trailing_pct or 0.0)

    if tp > 0.0 and sl > 0.0:
        try:
            last_trade = client.get_last_trade(ticker)
            px = to_float(getattr(last_trade, "price", None), 0.0)
        except Exception:
            px = 0.0
        if px > 0.0:
            tp_price = round(px * (1.0 + tp / 100.0), 2)
            sl_price = round(px * (1.0 - sl / 100.0), 2)
            order_kwargs.update({
                "type": "market",
                "order_class": "bracket",
                "take_profit": {"limit_price": tp_price},
                "stop_loss": {"stop_price": sl_price},
            })
        else:
            order_kwargs.update({"type": "market"})
    else:
        order_kwargs.update({"type": "market"})

    # Attempt trailing attachment via OTO if configured and supported
    if tr > 0.0 and ("order_class" not in order_kwargs or order_kwargs.get("order_class") != "bracket"):
        try:
            tmp = dict(order_kwargs)
            tmp.update({
                "order_class": "oto",
                "take_profit": None,
                "stop_loss": None,
                "trail_percent": tr,
            })
            order = client.submit_order(**tmp)
            return str(getattr(order, "id", None)) if order else None
        except Exception:
            pass

    order = client.submit_order(**order_kwargs)
    try:
        return str(order.id)
    except Exception:
        return None


def calendar_is_open_now(client: AlpacaREST) -> bool:
    try:
        ny_tz = pytz.timezone("America/New_York")
        now_ny = utc_now().astimezone(ny_tz)
        today = now_ny.date()
        cal = client.get_calendar(start=str(today), end=str(today))
        entries = list(cal) if isinstance(cal, (list, tuple)) else [cal]
        if not entries:
            return False
        c = entries[0]
        cal_date = getattr(c, "date", today)
        if isinstance(cal_date, str):
            cal_date = dt.date.fromisoformat(cal_date)
        open_str = str(getattr(c, "open", "09:30"))
        close_str = str(getattr(c, "close", "16:00"))
        open_h, open_m = map(int, open_str[:5].split(":"))
        close_h, close_m = map(int, close_str[:5].split(":"))
        open_dt = ny_tz.localize(dt.datetime(cal_date.year, cal_date.month, cal_date.day, open_h, open_m))
        close_dt = ny_tz.localize(dt.datetime(cal_date.year, cal_date.month, cal_date.day, close_h, close_m))
        return open_dt <= now_ny < close_dt
    except Exception:
        return False


def is_market_open(client: AlpacaREST) -> bool:
    try:
        clock = client.get_clock()
        if bool(getattr(clock, "is_open", False)):
            return True
    except Exception:
        pass
    # Fallback to calendar window check
    return calendar_is_open_now(client)


def ny_now() -> dt.datetime:
    tz = pytz.timezone("America/New_York")
    return utc_now().astimezone(tz)


def is_us_equity_market_open_now_hardcoded() -> bool:
    now_et = ny_now()
    if now_et.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= now_et < close_time


def next_us_equity_open_hardcoded() -> dt.datetime:
    tz = pytz.timezone("America/New_York")
    now_et = ny_now()
    # If before today's open on a weekday
    candidate = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_et.weekday() < 5 and now_et < candidate:
        return candidate
    # Otherwise move to next weekday at 9:30
    days_ahead = 1
    while True:
        d = now_et + dt.timedelta(days=days_ahead)
        if d.weekday() < 5:
            return tz.localize(dt.datetime(d.year, d.month, d.day, 9, 30, 0))
        days_ahead += 1


def wait_until_market_open(client: AlpacaREST, poll_seconds: int = 60) -> None:
    """Block until both hardcoded hours and Alpaca clock report market open.

    Behavior:
    - Sleep the full remaining time to the next open.
    - If that span crosses a weekday 09:25 ET checkpoint, first sleep until 09:25,
      then recompute and sleep the remaining full span to open. This ensures a
      fresh calculation near the session start without burst polling.
    """
    printed: bool = False
    while True:
        hard_open = is_us_equity_market_open_now_hardcoded()
        api_open = False
        try:
            api_open = is_market_open(client)
        except Exception:
            api_open = False
        if hard_open or api_open:
            return

        now_et = ny_now()
        nxt_open = next_us_equity_open_hardcoded()
        # Compute next 09:25 ET on a weekday
        tz = pytz.timezone("America/New_York")
        cand = now_et.replace(hour=9, minute=25, second=0, microsecond=0)
        if now_et.weekday() < 5 and now_et < cand:
            next_925 = cand
        else:
            days_ahead = 1
            while True:
                d = now_et + dt.timedelta(days=days_ahead)
                if d.weekday() < 5:
                    next_925 = tz.localize(dt.datetime(d.year, d.month, d.day, 9, 25, 0))
                    break
                days_ahead += 1

        # Choose sleep target: first hit 09:25 if it comes before the open; else sleep to open
        target = next_925 if next_925 < nxt_open else nxt_open
        remaining = max(1, int((target - ny_now()).total_seconds()))
        if not printed or target is next_925:
            # Print full time-to-open for transparency
            total_remaining_to_open = max(1, int((nxt_open - ny_now()).total_seconds()))
            print(f"Market closed. Next open: {nxt_open} (waiting {total_remaining_to_open}s)...")
            printed = True
        time.sleep(remaining)


def has_open_order(client: AlpacaREST, ticker: str, side: Optional[str] = None) -> bool:
    try:
        orders = client.get_orders(status="open", symbols=[ticker], limit=100)
    except Exception:
        return False
    for o in orders:
        try:
            if side and getattr(o, "side", None) != side:
                continue
            status = (getattr(o, "status", "") or "").lower()
            if status not in {"filled", "canceled", "expired", "rejected", "stopped"}:
                return True
        except Exception:
            continue
    return False


def poll_order_fill(client: AlpacaREST, order_id: Optional[str], timeout_seconds: int = 20, interval_seconds: int = 2) -> None:
    if not order_id:
        return
    deadline = time.time() + max(1, timeout_seconds)
    while time.time() < deadline:
        try:
            o = client.get_order(order_id)
            status = (getattr(o, "status", "") or "").lower()
            if status in {"filled", "partially_filled"}:
                return
        except Exception:
            pass
        time.sleep(max(1, interval_seconds))


def sell_all_position(client: AlpacaREST, ticker: str) -> None:
    try:
        client.close_position(ticker)
    except AlpacaAPIError as e:  # pragma: no cover
        # If no open position, ignore
        if "position does not exist" in str(e).lower():
            return
        raise


def submit_fractional_sell(client: AlpacaREST, ticker: str, fraction: float) -> Optional[str]:
    pos = get_tsla_position(client, ticker)
    if not pos or pos["qty"] <= 0.0:
        return None
    qty = float(pos["qty"]) * max(0.0, min(1.0, float(fraction)))
    if qty <= 0:
        return None
    order = client.submit_order(
        symbol=ticker,
        side="sell",
        type="market",
        time_in_force="day",
        qty=str(qty),
    )
    try:
        return str(order.id)
    except Exception:
        return None


def ensure_initial_allocation(client: AlpacaREST, ticker: str, initial_notional: float, max_cap_usd: float) -> None:
    position = get_tsla_position(client, ticker)
    if position is not None and position["market_value"] > 0.0:
        print(
            f"Existing {ticker} position detected: qty={position['qty']}, market_value=${position['market_value']:.2f}"
        )
        return
    if has_open_order(client, ticker, side="buy"):
        print(f"Open BUY order detected for {ticker}. Waiting for fill.")
        return
    remaining_to_cap = max(0.0, float(max_cap_usd) - 0.0)
    buy_amount = min(float(initial_notional), remaining_to_cap)
    if buy_amount < 1.0:
        print(f"Initial allocation below $1 after cap; skipping initial buy.")
        return
    print(f"No existing {ticker} position. Buying initial ${buy_amount:.2f} (cap ${max_cap_usd:.2f})...")
    order_id = submit_notional_buy(client, ticker, buy_amount)
    print("Initial buy submitted; polling for fill briefly...")
    poll_order_fill(client, order_id)


def sleep_interval(interval_hours: float) -> None:
    seconds_float = max(0.0, interval_hours * 3600.0)
    seconds = int(math.floor(seconds_float + 0.5))
    if seconds < 1:
        seconds = 1
    print(f"Sleeping {seconds}s until next check...")
    time.sleep(seconds)


def trade_cycle(
    client: AlpacaREST,
    ticker: str,
    max_cap_usd: float,
    short_window: int,
    long_window: int,
    trailing_pct: float,
    max_drawdown_pct: float,
) -> Tuple[str, Optional[str]]:
    closes = polygon_fetch_hourly_closes(ticker, hours_back=cfg.HOURS_BACK_FOR_TREND)
    # Always print interval efficiency advisory on SELL path as well
    try:
    advice, _est = estimate_interval_efficiency(
            closes,
            float(getattr(trade_cycle, "_interval_hours", 1.0)),
        max(1.0, remaining_to_cap if 'remaining_to_cap' in locals() else 100.0),
        )
        if advice:
            print(advice)
    except Exception:
        pass
    action = decide_action(closes, short_window, long_window)
    if action == "buy":
        if has_open_order(client, ticker, side="buy"):
            return action, "Open BUY order exists; skipping duplicate"
        pos = get_tsla_position(client, ticker)
        current_value = float(pos["market_value"]) if pos else 0.0
        if current_value >= float(max_cap_usd) - 1e-9:
            return action, f"At/above cap (${max_cap_usd:.2f}); skipping BUY"
        remaining_to_cap = max(0.0, float(max_cap_usd) - current_value)
        # Dynamic per-trade parameters (respect CLI overrides only for interval and max cap externally)
        base_tp = float(getattr(cfg, 'TAKE_PROFIT_PERCENT', 0.0) or 0.0)
        base_sl = float(getattr(cfg, 'STOP_LOSS_PERCENT', 0.0) or 0.0)
        base_tr = float(getattr(cfg, 'TRAILING_STOP_PERCENT', 0.0) or 0.0)
        # Detect if overrides were provided via CLI by comparing to globals stored on main() closure through attributes
        tp_overridden = bool(getattr(trade_cycle, "_tp_overridden", False))
        sl_overridden = bool(getattr(trade_cycle, "_sl_overridden", False))
        trail_overridden = bool(getattr(trade_cycle, "_trail_overridden", False))
        short_overridden = bool(getattr(trade_cycle, "_short_overridden", False))
        long_overridden = bool(getattr(trade_cycle, "_long_overridden", False))

        per_buy_dyn, tp_pct, sl_pct, tr_pct, eff_short, eff_long, why = compute_dynamic_trade_params(
            closes,
            float(getattr(trade_cycle, "_interval_hours", 1.0)),
            current_value,
            remaining_to_cap,
            base_tp,
            base_sl,
            base_tr,
            int(short_window),
            int(long_window),
            tp_overridden,
            sl_overridden,
            trail_overridden,
            short_overridden,
            long_overridden,
        )
        if per_buy_dyn < 1.0:
            return action, f"Low confidence/size; HOLD (calc: {why})"
        order_id = submit_dynamic_buy(client, ticker, per_buy_dyn, tp_pct, sl_pct, tr_pct)
        advice, _ = estimate_interval_efficiency(closes, float(getattr(trade_cycle, "_interval_hours", 1.0)), per_buy_dyn)
        if advice:
            print(advice)
        return action, f"Submitted BUY ${per_buy_dyn:.2f} | {why}"
    # action == "sell"
    position = get_tsla_position(client, ticker)
    if position is None or position["qty"] <= 0.0:
        if has_open_order(client, ticker, side="buy"):
            return action, "Pending BUY order; not selling"
        # Nothing to sell; skip but report
        return action, "No position to sell"
    # Confidence-based partial sell mirrors buy using inverse signal (long_ema > short_ema)
    closes = polygon_fetch_hourly_closes(ticker, hours_back=cfg.HOURS_BACK_FOR_TREND)
    eff_short = int(short_window)
    eff_long = int(long_window)
    vol = compute_volatility(closes)
    eff_short, eff_long = _dynamic_windows(eff_short, eff_long, vol, (
        bool(getattr(trade_cycle, "_short_overridden", False)),
        bool(getattr(trade_cycle, "_long_overridden", False)),
    ))
    if len(closes) >= max(eff_short, eff_long):
        short_ema = compute_ema(closes[-eff_short:], eff_short)
        long_ema = compute_ema(closes[-eff_long:], eff_long)
        spread = long_ema - short_ema
        ratio = spread / max(1e-9, long_ema)
        # Use defaults unless strong reversal appears
        if ratio < 0.001:
            return action, "Weak SELL signal; holding position"
        if ratio < 0.003:
            submit_fractional_sell(client, ticker, 0.10)
            return action, "Submitted SELL 10% (weak signal)"
        if ratio < 0.01:
            submit_fractional_sell(client, ticker, 0.50)
            return action, "Submitted SELL 50% (moderate signal)"
        # Strong signal: sell all
        sell_all_position(client, ticker)
        return action, "Submitted SELL ALL (strong signal)"
    # Risk exits: drawdown or position age
    try:
        max_dd = float(max_drawdown_pct)
        max_age = float(getattr(cfg, "MAX_POSITION_AGE_HOURS", 0.0) or 0.0)
    except Exception:
        max_dd, max_age = 0.0, 0.0
    if max_dd > 0.0:
        try:
            avg_entry = float(position["avg_entry_price"]) if position else 0.0
            # Estimate drawdown based on last close
            closes = polygon_fetch_hourly_closes(ticker, hours_back=2)
            last = closes[-1] if closes else avg_entry
            if avg_entry > 0.0 and last > 0.0:
                dd_pct = max(0.0, (avg_entry - last) / avg_entry) * 100.0
                if dd_pct >= max_dd:
                    sell_all_position(client, ticker)
                    return action, f"Drawdown {dd_pct:.2f}% >= {max_dd:.2f}%: SELL ALL"
        except Exception:
            pass
    if max_age > 0.0:
        # Approximate position age by checking last transaction time is not directly available; skip here or integrate fills history
        pass

    # Fallback: sell all if we can't compute EMAs reliably
    sell_all_position(client, ticker)
    return action, f"Submitted SELL ALL for {ticker}"


def parse_interval_from_argv(argv: List[str]) -> Tuple[float, bool, float, bool, List[str]]:
    """Pass-through pre-parse: argparse will handle all flags; no bare numerics allowed."""
    interval_hours = 1.0
    run_once = False
    max_cap_usd = float(cfg.INITIAL_NOTIONAL_USD)
    interval_specified = False
    remaining: List[str] = list(argv or [])
    return interval_hours, run_once, max_cap_usd, interval_specified, remaining


def normalize_argv_flags(argv: List[str]) -> List[str]:
    """Normalize option case so flags are case-insensitive; supports attached values like -T0.5.

    Recognized short flags: t,m,p,s,l,r,d,h; recognized long flags are lowercased.
    """
    out: List[str] = []
    i = 0
    short_flags = {"-t", "-m", "-s", "-l", "-r", "-d", "-h"}
    long_flags = {"--time", "--max", "--short", "--long", "--trail", "--drawdown"}

    def merge_numeric(parts: List[str]) -> Tuple[str, int]:
        # Merge tokens into a single numeric string (handles cases like "10." "0" -> "10.0", ".25" etc.)
        if not parts:
            return "", 0
        s = parts[0]
        consumed = 1
        dot_used = "." in s
        while consumed < len(parts):
            t = parts[consumed]
            if s.endswith(".") and t.isdigit():
                s += t
                consumed += 1
                continue
            if t == "." and not dot_used:
                s += t
                dot_used = True
                consumed += 1
                continue
            if t.startswith(".") and t[1:].isdigit() and not dot_used:
                s += t
                dot_used = True
                consumed += 1
                continue
            if t.isdigit():
                s += t
                consumed += 1
                continue
            break
        return s, consumed

    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            flag = tok.lower()
            out.append(flag)
            # If this is a recognized long flag expecting a number, attempt to merge numeric fragments
            if flag in long_flags and (i + 1) < len(argv):
                val, used = merge_numeric(argv[i + 1 : i + 6])  # look ahead up to 5 tokens
                if used > 0:
                    out.append(val)
                    i += 1 + used
                    continue
            i += 1
            continue
        if tok.startswith("-") and len(tok) >= 2 and tok[1].isalpha():
            # Lowercase the flag letter
            norm_flag = "-" + tok[1].lower()
            rest = tok[2:]
            out.append(norm_flag)
            # Handle attached value e.g., -t0.5
            if rest:
                out.append(rest)
            # Handle separated numeric fragments after recognized short flags
            if norm_flag in short_flags and (i + 1) < len(argv):
                # If we already had an attached value, also allow merging with following fragments
                start_idx = i + 1
                if rest:
                    # Include the attached 'rest' as the first part
                    parts = [rest] + argv[start_idx : start_idx + 5]
                    merged, used = merge_numeric(parts)
                    out[-1] = merged  # replace last appended value
                    i += 1 + max(0, used - 1)
                    continue
                else:
                    merged, used = merge_numeric(argv[start_idx : start_idx + 5])
                    if used > 0:
                        out.append(merged)
                        i += 1 + used
                        continue
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def pretokenize(argv: List[str]) -> List[str]:
    res: List[str] = []
    for tok in argv:
        m = re.match(r'^(\d+(?:\.\d*)?|\.\d+)(-[A-Za-z].*)$', tok)
        if m:
            res.append(m.group(1))
            res.append(m.group(2))
        else:
            res.append(tok)
    return res


def parse_args(argv: Optional[List[str]] = None) -> Tuple[argparse.Namespace, float, bool, float, bool]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    pre = pretokenize(raw_argv)
    normalized = normalize_argv_flags(pre)
    interval_hours, run_once, max_cap_usd, interval_specified, rest = parse_interval_from_argv(normalized)
    parser = argparse.ArgumentParser(description="Alpaca paper trading bot for TSLA", add_help=False)
    parser.add_argument("--help", "-?", action="help", help="Show this help message and exit")
    parser.add_argument(
        "-t", "--time",
        dest="interval_hours_override",
        type=float,
        default=None,
        help="Interval hours between checks (e.g., 1, 0.5, .001)",
    )
    parser.add_argument(
        "-m", "--max",
        dest="max_cap_override",
        type=float,
        default=None,
        help="Max USD cap for TSLA total position value",
    )
    # Removed per-buy; sizing is dynamic from remaining cap
    parser.add_argument(
        "-s", "--short",
        dest="short_ema_override",
        type=float,
        default=None,
        help="Short EMA hours",
    )
    parser.add_argument(
        "-l", "--long",
        dest="long_ema_override",
        type=float,
        default=None,
        help="Long EMA hours",
    )
    parser.add_argument(
        "-r", "--trail",
        dest="trail_override",
        type=float,
        default=None,
        help="Trailing stop percent",
    )
    parser.add_argument(
        "-d", "--drawdown",
        dest="drawdown_override",
        type=float,
        default=None,
        help="Max drawdown percent before exit",
    )
    parser.add_argument(
        "-h",
        dest="max_hours",
        type=float,
        default=None,
        help="Maximum runtime hours before exiting (omit for unlimited).",
    )
    args = parser.parse_args(rest)
    if args.interval_hours_override is not None:
        interval_hours = abs(float(args.interval_hours_override))
        interval_specified = True
    if args.max_cap_override is not None:
        max_cap_usd = abs(float(args.max_cap_override))
    return args, interval_hours, run_once, max_cap_usd, interval_specified


def main(argv: Optional[List[str]] = None) -> int:
    args, interval_hours, run_once, max_cap_usd, interval_specified = parse_args(argv)
    # Print process header
    ensure_process_header_printed()
    required = {
        "ALPACA_API_KEY": cfg.ALPACA_API_KEY,
        "ALPACA_API_SECRET": cfg.ALPACA_API_SECRET,
        "POLYGON_API_KEY": cfg.POLYGON_API_KEY,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(
            "Missing required API keys in config.py: " + ", ".join(missing)
        )
        return 2

    client = build_alpaca_client()
    ticker = cfg.TICKER

    print(f"Alpaca base URL: {cfg.ALPACA_BASE_URL}")
    print(f"Starting paper trading for {ticker}...")
    try:
        requested_seconds_f = float(interval_hours) * 3600.0
        interval_seconds = int(math.floor(requested_seconds_f + 0.5))
        if interval_seconds < 1:
            print(
                f"Requested interval ~{requested_seconds_f:.3f}s < 1s; using 1 second minimum"
            )
            interval_seconds = 1
    except Exception:
        interval_seconds = 3600
    # Startup summary (concise 4-6 words each)
    # Resolve effective settings with overrides
    per_buy = 0.0  # unused placeholder; sizing is dynamic
    short_ema = int(round(float(args.short_ema_override))) if args.short_ema_override is not None else int(cfg.SHORT_EMA_HOURS)
    long_ema = int(round(float(args.long_ema_override))) if args.long_ema_override is not None else int(cfg.LONG_EMA_HOURS)
    trail_pct = float(args.trail_override) if args.trail_override is not None else float(getattr(cfg,'TRAILING_STOP_PERCENT',0.0))
    dd_pct = float(args.drawdown_override) if args.drawdown_override is not None else float(getattr(cfg,'MAX_DRAWDOWN_PERCENT',0.0))

    # Startup summary including flag letters
    max_run_str = f"{args.max_hours}h" if args.max_hours is not None else "unlimited"
    interval_summary = "one-shot" if not interval_specified else f"{interval_hours}h ({interval_seconds}s)"
    print(
        f"-t Interval: {interval_summary} | -m Cap: ${max_cap_usd:.2f} | "
        f"-s/-l EMAs: {short_ema}/{long_ema}h | -r Trail: {trail_pct:.2f}% | -d DD: {dd_pct:.2f}% | -h MaxRun: {max_run_str}"
    )
    # Default to one-shot when no -t/--time provided
    run_once = not interval_specified

    if run_once:
        if not (is_us_equity_market_open_now_hardcoded() and is_market_open(client)):
            wait_until_market_open(client, poll_seconds=30)
        # During market hours, prevent system sleep
        set_keep_awake(True)
        while True:
            try:
                # Annotate override flags for dynamic logic
                # per-buy override removed
                setattr(trade_cycle, "_tp_overridden", False)  # no CLI flag for TP; treat as not overridden
                setattr(trade_cycle, "_sl_overridden", False)  # no CLI flag for SL; treat as not overridden
                setattr(trade_cycle, "_trail_overridden", args.trail_override is not None)
                setattr(trade_cycle, "_short_overridden", args.short_ema_override is not None)
                setattr(trade_cycle, "_long_overridden", args.long_ema_override is not None)
                setattr(trade_cycle, "_interval_hours", interval_hours)
                action, detail = trade_cycle(
                    client, ticker, max_cap_usd, short_ema, long_ema, trail_pct, dd_pct
                )
                print(f"Decision: {action.upper()} - {detail}")
                set_keep_awake(False)
                return 0
            except RateLimitError as rl:
                print(f"Polygon rate limited. Waiting {rl.wait_seconds}s before retry...")
                time.sleep(int(rl.wait_seconds))
                continue
            except Exception as e:
                print(f"Error during one-shot trade: {e}")
                set_keep_awake(False)
                return 1

    started = utc_now()
    # Track PnL using account equity deltas
    try:
        acct = client.get_account()
        last_equity = float(acct.equity)
    except Exception:
        last_equity = 0.0
    total_pnl = 0.0
    closed_notice_printed = False
    while True:
        cycle_started = utc_now()
        try:
            # Global overflow guard: if log grew huge, trim down to 100 lines
            try:
                trim_log_if_overflow(getattr(cfg, "LOG_PATH", "bot.log"), threshold_lines=1000, keep_lines=100)
            except Exception:
                pass
            # Skip trading when market is closed; print notice once per closed period
            if not is_us_equity_market_open_now_hardcoded() and not is_market_open(client):
                if not closed_notice_printed:
                    closed_notice_printed = True
                    set_keep_awake(False)
                # Wait until market open (ignore -t while closed)
                if args.max_hours is not None:
                    elapsed_hours = (utc_now() - started).total_seconds() / 3600.0
                    if elapsed_hours >= float(args.max_hours):
                        print("Reached maximum runtime. Exiting.")
                        return 0
                # Poll less frequently while closed
                try:
                    poll_seconds = max(10, min(300, int(interval_seconds)))
                except Exception:
                    poll_seconds = 60
                # On first entry to closed period, if log grows too large, trim
                try:
                    trim_log_at_market_close(getattr(cfg, "LOG_PATH", "bot.log"), max_lines=100)
                except Exception:
                    pass
                wait_until_market_open(client, poll_seconds=poll_seconds)
                continue

            # Market is open: clear notice flag, ensure allocation, then trade
            if closed_notice_printed:
                print("Market open. Resuming trading.")
                closed_notice_printed = False
            set_keep_awake(True)

            try:
                # per-buy override removed
                setattr(trade_cycle, "_tp_overridden", False)
                setattr(trade_cycle, "_sl_overridden", False)
                setattr(trade_cycle, "_trail_overridden", args.trail_override is not None)
                setattr(trade_cycle, "_short_overridden", args.short_ema_override is not None)
                setattr(trade_cycle, "_long_overridden", args.long_ema_override is not None)
                # Advise the best interval but do not auto-switch; keep user-chosen interval
                hourly = polygon_fetch_hourly_closes(ticker, hours_back=cfg.HOURS_BACK_FOR_TREND)
                _next_interval, advice = choose_best_interval(hourly, interval_hours, max(1.0, float(max_cap_usd)))
                if advice:
                    print(advice)
                setattr(trade_cycle, "_interval_hours", interval_hours)
                action, detail = trade_cycle(
                    client, ticker, max_cap_usd, short_ema, long_ema, trail_pct, dd_pct
                )
            except RateLimitError as rl:
                print(f"Polygon rate limited. Waiting {rl.wait_seconds}s before retry...")
                time.sleep(int(rl.wait_seconds))
                continue
            # Fetch current equity to compute PnL metrics
            try:
                acct_now = client.get_account()
                equity_now = float(acct_now.equity)
            except Exception:
                equity_now = last_equity

            last_interval_pnl = equity_now - last_equity
            total_pnl += last_interval_pnl
            last_equity = equity_now
            elapsed_since_start_hours = max(1e-9, (utc_now() - started).total_seconds() / 3600.0)
            avg_hourly_pnl = total_pnl / elapsed_since_start_hours

            print(
                f"{cycle_started.isoformat()} Decision: {action.upper()} - {detail} | "
                f"Last interval PnL=${last_interval_pnl:.2f} | Avg/hr=${avg_hourly_pnl:.2f} | Total=${total_pnl:.2f}"
            )
        except Exception as e:  # pragma: no cover
            print(f"Error during trade cycle: {e}")

        if args.max_hours is not None:
            elapsed_hours = (utc_now() - started).total_seconds() / 3600.0
            if elapsed_hours >= float(args.max_hours):
                print("Reached maximum runtime. Exiting.")
                return 0

        # Sleep using the user-chosen interval (no auto-switch)
        sleep_interval(interval_hours)


if __name__ == "__main__":
    sys.exit(main())


