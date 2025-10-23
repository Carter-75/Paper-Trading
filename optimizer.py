import sys
from typing import List, Optional, Tuple

import pytz

import config
from runner import (
    make_client,
    fetch_closes,
    simulate_signals_and_projection,
    snap_interval_to_supported_seconds,
    pct_stddev,
)


def _build_candidate_intervals() -> List[int]:
    # Log-spaced seconds from ~23s to 6.5h, snapped to supported sizes (1/5/15/60m)
    min_secs = 23
    max_secs = int(6.5 * 3600)
    steps = 24
    ratio = (max_secs / float(min_secs)) ** (1.0 / max(1, steps - 1))
    cand: List[int] = []
    cur = float(min_secs)
    for _ in range(steps):
        cand.append(int(max(5, round(cur))))
        cur *= ratio
    cand.extend([60, 300, 900, 3600])
    snapped = [snap_interval_to_supported_seconds(x) for x in cand]
    return sorted(set(snapped))


def _bars_for_one_year(interval_seconds: int) -> int:
    # Approximate trading days per year and bars/day per interval
    trading_days = 252
    if interval_seconds <= 60:
        bars_per_day = 390
    elif interval_seconds <= 300:
        bars_per_day = 78
    elif interval_seconds <= 900:
        bars_per_day = 26
    else:
        bars_per_day = 7  # 60m ≈ 6.5 → 7 safe
    total = trading_days * bars_per_day
    # Avoid very heavy loads; let fetch_closes fallback handle start-based fetch if needed
    return min(total, 8000)


def _cap_grid() -> List[float]:
    # Candidate max caps in USD (wide range)
    return [25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000]


def _tp_grid() -> List[float]:
    lo = max(0.1, config.MIN_TAKE_PROFIT_PERCENT)
    hi = min(15.0, config.MAX_TAKE_PROFIT_PERCENT)
    grid = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    return [x for x in grid if lo <= x <= hi]


def _sl_grid() -> List[float]:
    lo = max(0.1, config.MIN_STOP_LOSS_PERCENT)
    hi = min(15.0, config.MAX_STOP_LOSS_PERCENT)
    grid = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    return [x for x in grid if lo <= x <= hi]


def _trade_frac_grid() -> List[float]:
    lo = max(0.01, config.MIN_TRADE_SIZE_FRAC)
    hi = min(0.95, max(config.MAX_TRADE_SIZE_FRAC, lo))
    grid = [0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.66, 0.75, 0.85]
    # If risky mode is on, allow up to risky cap; else cap at MAX_TRADE_SIZE_FRAC
    if config.RISKY_MODE_ENABLED:
        hi = min(max(hi, config.RISKY_MAX_FRAC_CAP), 0.95)
    else:
        hi = min(hi, config.MAX_TRADE_SIZE_FRAC)
    return [f for f in grid if lo <= f <= hi]


def _expected_return_per_trade(win_rate: float, tp_pct: float, sl_pct: float) -> float:
    # Expected fractional return per trade
    return ((tp_pct / 100.0) * max(0.0, min(1.0, win_rate))) - ((sl_pct / 100.0) * (1.0 - max(0.0, min(1.0, win_rate))))


def optimize(symbol: str) -> Tuple[int, float, float]:
    client = make_client(allow_missing=False, go_live=False)
    candidates = _build_candidate_intervals()

    best_interval: Optional[int] = None
    best_cap_usd: float = 0.0
    best_expected_daily: float = -1e18

    for secs in candidates:
        bars = _bars_for_one_year(secs)
        try:
            closes = fetch_closes(client, symbol, int(secs), bars)
            if not closes or len(closes) < max(config.LONG_WINDOW + 10, 30):
                continue
            # Simple risk constraint: skip intervals with excessive recent volatility (relaxed x1.5)
            vol_pct = pct_stddev(closes[-max(config.VOLATILITY_WINDOW, 10):])
            if vol_pct >= (1.5 * config.VOLATILITY_PCT_THRESHOLD):
                continue

            # Remove strong gating to truly maximize; evaluate a grid of TP/SL/size and caps
            for tp_eval in _tp_grid():
                for sl_eval in _sl_grid():
                    sim = simulate_signals_and_projection(
                        closes,
                        int(secs),
                        override_tp_pct=float(tp_eval),
                        override_sl_pct=float(sl_eval),
                    )
                    trades_per_day = float(sim["expected_trades_per_day"])
                    if trades_per_day <= 0:
                        continue
                    for trade_frac_eval in _trade_frac_grid():
                        for cap in _cap_grid():
                            expected_daily = simulate_signals_and_projection(
                                closes,
                                int(secs),
                                override_tp_pct=float(tp_eval),
                                override_sl_pct=float(sl_eval),
                                override_trade_frac=float(trade_frac_eval),
                                override_cap_usd=float(cap),
                            )["expected_daily_usd"]
                            if expected_daily > best_expected_daily:
                                best_expected_daily = float(expected_daily)
                                best_interval = int(secs)
                                best_cap_usd = float(cap)
        except Exception:
            continue

    if best_interval is None:
        # Fallback to 60m, $100
        best_interval = 3600
        best_cap_usd = 100.0
        best_expected_daily = 0.0

    return best_interval, round(best_cap_usd, 2), best_expected_daily


def main() -> int:
    symbol = (config.DEFAULT_TICKER or "TSLA").upper()
    best_interval, best_cap_usd, expected_daily = optimize(symbol)
    print(f"{best_interval} {best_cap_usd} {round(expected_daily, 2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


