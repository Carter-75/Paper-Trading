# COMPREHENSIVE BUG FIX REQUEST - Paper Trading Bot

## EXECUTIVE SUMMARY

I have a Python paper trading bot for stocks (Alpaca API + Polygon data). A code review identified **20 bugs** ranging from critical (will crash) to low severity (edge cases). I need you to **systematically fix ALL bugs** working through 4 priority phases. This is a complete implementation guide with exact code changes needed.

---

## REPOSITORY OVERVIEW

```
Paper-Trading/
├── runner.py            # Main bot - 1269 lines (trading logic, market data, safety checks)
├── config.py            # Configuration - 118 lines (env vars, risk parameters)
├── stock_scanner.py     # Stock ranking - 290 lines (scans top 100 stocks by market cap)
├── portfolio_manager.py # Position tracking - 110 lines (manages multi-stock portfolio)
├── optimizer.py         # Parameter optimizer - 314 lines (binary search for best interval/capital)
├── requirements.txt     # Dependencies (alpaca-trade-api, yfinance, polygon, etc.)
└── .env                # User secrets (API keys)
```

**Tech Stack:** Python 3.9+, Alpaca Paper Trading API, Polygon free tier (5 calls/min), yfinance  
**Trading Style:** Simple moving average crossover (9/21 SMA), dynamic position sizing, multi-stock portfolio  
**Current State:** Works on paper trading but has critical bugs preventing safe live use

---

## BUGS IDENTIFIED (20 Total)

### 🔴 CRITICAL - Will Cause Crashes/Failures (Fix in Phase 1)

**BUG-001:** Missing `MAX_DAILY_LOSS_PERCENT` in config.py  
- `runner.py:534` uses `config.MAX_DAILY_LOSS_PERCENT` but variable doesn't exist → AttributeError crash

**BUG-002:** Fractional share quantity loss in sell_flow  
- `runner.py:486` converts `qty = int(pos["qty"])` losing fractional shares (0.5 → 0)

**BUG-003:** Wrong time-in-force for fractional sells  
- `runner.py:499` uses 'gtc' for all sells, but Alpaca requires 'day' for fractional shares → API rejection

**BUG-004:** No idempotency protection  
- Network retry can submit duplicate orders → double positions, violated risk limits

**BUG-005:** No order fill reconciliation  
- Assumes instant fills. Live trading has partial fills → position state mismatch

### 🟠 HIGH SEVERITY - Silent Failures/Logic Flaws (Fix in Phase 2)

**BUG-006:** No Polygon API rate limiting  
- Free tier = 5 calls/min. Multi-stock scanner hits 15-100 stocks → throttling, partial data, selection bias

**BUG-007:** No data timestamp validation  
- Could trade on stale data if API delayed → buying yesterday's winners (momentum exhausted)

**BUG-008:** Simulation look-ahead bias  
- Backtests on same data used for forward decisions → overfitted, inflated expected returns

**BUG-009:** No slippage modeling  
- Paper fills at exact price, live has 0.05-0.1% slippage → 20-40% performance degradation

**BUG-010:** Risk config validation missing  
- RISKY_MODE multipliers can push position size beyond MAX_TRADE_SIZE_FRAC → 120% of capital in one stock

### 🟡 MEDIUM SEVERITY - Suboptimal Behavior (Fix in Phase 3)

**BUG-011:** Smart allocation sells mid-session positions  
- Can rebalance profitable positions due to temporary dip → whipsaw, transaction costs

**BUG-012:** No limit order option  
- All market orders → vulnerable to flash crashes, no price protection

**BUG-013:** Contradictory stop-loss configs  
- Has DAILY_LOSS_LIMIT_USD, MAX_DAILY_LOSS_PERCENT, MAX_DRAWDOWN_PERCENT → unclear precedence

**BUG-014:** Profitability freeze detection missing  
- If all stocks unprofitable, bot runs 24/7 doing nothing → user thinks it's working

**BUG-015:** No unit tests  
- Zero test coverage → can't validate fixes

### 🟢 LOW SEVERITY - Edge Cases (Fix in Phase 4)

**BUG-016:** Market hours fallback uses wrong timezone assumptions  
**BUG-017:** No volume filter (could select illiquid stocks)  
**BUG-018:** No bid-ask spread consideration  
**BUG-019:** Over-aggressive default settings (75% position size, risky mode ON by default)  
**BUG-020:** Missing risk disclosure in documentation  

---

# IMPLEMENTATION GUIDE

Work through each phase sequentially. Test after each fix. Do NOT skip ahead.

---

## PHASE 1: CRITICAL BUGS (Must Fix Before Any Live Trading)

### FIX 1.1 - Add MAX_DAILY_LOSS_PERCENT Configuration

**File:** `config.py`  
**Location:** After line 62 (after DAILY_LOSS_LIMIT_USD)

```python
# Safety
MAX_DRAWDOWN_PERCENT: float = float(os.getenv("MAX_DRAWDOWN_PERCENT", "10.0") or 0.0)
MAX_POSITION_AGE_HOURS: float = float(os.getenv("MAX_POSITION_AGE_HOURS", "72.0") or 0.0)
DAILY_LOSS_LIMIT_USD: float = float(os.getenv("DAILY_LOSS_LIMIT_USD", "100.0") or 0.0)
MAX_DAILY_LOSS_PERCENT: float = float(os.getenv("MAX_DAILY_LOSS_PERCENT", "5.0"))  # <-- ADD THIS LINE
```

**Explanation:** This prevents the AttributeError at runtime. Default 5% daily loss limit.

---

### FIX 1.2 - Fix Fractional Share Handling in Sell Flow

**File:** `runner.py`  
**Location:** Line 486

**OLD CODE:**
```python
def sell_flow(client, symbol: str, confidence: float = 0.0):
    pos = get_position(client, symbol)
    if not pos:
        return (False, "No position")
    
    qty = int(pos["qty"])  # <-- BUG: Loses fractional shares!
    realized_pl = pos["unrealized_pl"]
```

**NEW CODE:**
```python
def sell_flow(client, symbol: str, confidence: float = 0.0):
    pos = get_position(client, symbol)
    if not pos:
        return (False, "No position")
    
    qty = float(pos["qty"])  # FIXED: Keep fractional shares
    realized_pl = pos["unrealized_pl"]
```

---

### FIX 1.3 - Fix Time-in-Force for Fractional Sells

**File:** `runner.py`  
**Location:** Line 499

**OLD CODE:**
```python
# Market sell
client.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
```

**NEW CODE:**
```python
# Market sell (use 'day' for fractional shares, 'gtc' for whole shares)
is_fractional = (qty % 1 != 0)
time_in_force = 'day' if is_fractional else 'gtc'
client.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force=time_in_force)
```

**Explanation:** Alpaca requires 'day' time-in-force for fractional share orders.

---

### FIX 1.4 - Add Idempotency Protection

**File:** `runner.py`

**Step 1:** Add imports and global tracker at top (after line 24):
```python
from typing import List, Optional, Tuple, Dict
import uuid  # <-- ADD THIS
from dotenv import load_dotenv
```

**Step 2:** Add global order tracker after imports (around line 36):
```python
load_dotenv()

# Idempotency: Track orders submitted this cycle to prevent duplicates
_order_ids_submitted_this_cycle = set()

# ===== Logging Setup =====
```

**Step 3:** In buy_flow, before FIRST submit_order (around line 425), add:
```python
def buy_flow(client, symbol: str, last_price: float, available_cap: float,
             confidence: float, base_frac: float, base_tp: float, base_sl: float,
             dynamic_enabled: bool = True, interval_seconds: int = None):
    """
    Buy a stock using available capital.
    """
    # ... existing code up to try: block ...
    
    try:
        tp_price = last_price * (1 + tp / 100.0)
        sl_price = last_price * (1 - sl / 100.0)
        
        # Generate unique client order ID for idempotency
        client_order_id = f"{symbol}_BUY_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Check if already submitted this cycle (network retry protection)
        if client_order_id in _order_ids_submitted_this_cycle:
            return (False, "Order already submitted this cycle")
        
        _order_ids_submitted_this_cycle.add(client_order_id)
        
        # Check if fractional
        is_fractional = (qty % 1 != 0)
        
        if is_fractional:
            # Fractional shares: Must use 'day' order, cannot use bracket orders
            client.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day',
                client_order_id=client_order_id  # <-- ADD THIS
            )
            # ... rest of fractional logic ...
```

**Step 4:** Update ALL submit_order calls in buy_flow to include `client_order_id` parameter (lines 434, 444, 453, 465)

**Step 5:** Similarly update sell_flow (around line 499):
```python
def sell_flow(client, symbol: str, confidence: float = 0.0):
    # ... existing code ...
    
    try:
        # Cancel bracket orders
        orders = client.list_orders(status='open', symbols=[symbol])
        for order in orders:
            try:
                client.cancel_order(order.id)
            except:
                pass
        
        # Generate unique client order ID
        client_order_id = f"{symbol}_SELL_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        if client_order_id in _order_ids_submitted_this_cycle:
            return (False, "Order already submitted this cycle")
        
        _order_ids_submitted_this_cycle.add(client_order_id)
        
        # Market sell
        is_fractional = (qty % 1 != 0)
        time_in_force = 'day' if is_fractional else 'gtc'
        client.submit_order(
            symbol=symbol, 
            qty=qty, 
            side='sell', 
            type='market', 
            time_in_force=time_in_force,
            client_order_id=client_order_id  # <-- ADD THIS
        )
```

**Step 6:** Clear tracker at start of each main loop iteration. In main(), find the main loop (line 998) and add at start of loop (line 1009):
```python
while True:
    iteration += 1
    
    # Clear order ID tracker for new iteration (idempotency protection)
    _order_ids_submitted_this_cycle.clear()
    
    if not in_market_hours(client):
```

---

### FIX 1.5 - Add Order Fill Reconciliation

**File:** `runner.py`

**In buy_flow**, after EACH submit_order call, add fill confirmation logic.

**For fractional orders (around line 434):**
```python
# Fractional shares: Must use 'day' order, cannot use bracket orders
order = client.submit_order(
    symbol=symbol,
    qty=qty,
    side='buy',
    type='market',
    time_in_force='day',
    client_order_id=client_order_id
)

# Wait for fill confirmation (market orders should fill quickly)
max_wait_seconds = 30
fill_confirmed = False
actual_filled_qty = 0

for wait_iter in range(max_wait_seconds):
    try:
        order_status = client.get_order(order.id)
        if order_status.status == 'filled':
            actual_filled_qty = float(order_status.filled_qty)
            fill_confirmed = True
            break
        elif order_status.status == 'partially_filled':
            actual_filled_qty = float(order_status.filled_qty)
            # For fractional, partial is common - accept it
            fill_confirmed = True
            break
    except:
        pass
    time.sleep(1)

if not fill_confirmed:
    return (False, f"Order not filled after {max_wait_seconds}s")

# Place separate TP and SL orders using ACTUAL filled quantity
try:
    client.submit_order(
        symbol=symbol,
        qty=actual_filled_qty,  # <-- Use actual filled qty
        side='sell',
        type='limit',
        time_in_force='day',
        limit_price=round(tp_price, 2)
    )
    client.submit_order(
        symbol=symbol,
        qty=actual_filled_qty,  # <-- Use actual filled qty
        side='sell',
        type='stop',
        time_in_force='day',
        stop_price=round(sl_price, 2)
    )
except:
    pass  # TP/SL orders are optional for fractional shares

shares_text = f"{actual_filled_qty:.6f}".rstrip('0').rstrip('.') if actual_filled_qty < 1 else f"{actual_filled_qty:.2f}"
return (True, f"Bought {shares_text} shares @ ${last_price:.2f} (TP:{tp:.2f}% SL:{sl:.2f}%)")
```

**For whole share bracket orders (around line 465):**
```python
# Whole shares: Use bracket orders with GTC
order = client.submit_order(
    symbol=symbol,
    qty=int(qty),
    side='buy',
    type='market',
    time_in_force='gtc',
    order_class='bracket',
    take_profit={'limit_price': round(tp_price, 2)},
    stop_loss={'stop_price': round(sl_price, 2)},
    client_order_id=client_order_id
)

# Wait for fill confirmation
max_wait_seconds = 30
fill_confirmed = False
actual_filled_qty = 0

for wait_iter in range(max_wait_seconds):
    try:
        order_status = client.get_order(order.id)
        if order_status.status in ['filled', 'partially_filled']:
            actual_filled_qty = float(order_status.filled_qty)
            fill_confirmed = True
            if order_status.status == 'filled':
                break
    except:
        pass
    time.sleep(1)

if not fill_confirmed:
    return (False, f"Order not filled after {max_wait_seconds}s")
elif actual_filled_qty < qty * 0.9:  # Less than 90% filled
    log_warn(f"Partial fill: {actual_filled_qty}/{qty} shares")
    # Continue anyway - we got something

shares_text = f"{actual_filled_qty:.6f}".rstrip('0').rstrip('.') if actual_filled_qty < 1 else f"{actual_filled_qty:.2f}"
return (True, f"Bought {shares_text} shares @ ${last_price:.2f} (TP:{tp:.2f}% SL:{sl:.2f}%)")
```

**Apply similar logic to sell_flow** (after line 499).

---

## PHASE 2: HIGH SEVERITY BUGS (Fix Before Scaling Capital)

### FIX 2.1 - Add Polygon API Rate Limiting

**File:** `runner.py`

**Step 1:** Add RateLimiter class before fetch_closes (around line 190):
```python
# ===== API Client =====
def make_client(allow_missing: bool = False, go_live: bool = False):
    # ... existing code ...

# ===== Polygon Rate Limiting =====
class PolygonRateLimiter:
    """Enforces Polygon free tier limit: 5 calls per minute with exponential backoff"""
    def __init__(self, calls_per_minute=5):
        self.calls = []
        self.limit = calls_per_minute
    
    def wait_if_needed(self):
        """Block if rate limit would be exceeded"""
        now = time.time()
        # Remove calls older than 60 seconds
        self.calls = [t for t in self.calls if now - t < 60]
        
        if len(self.calls) >= self.limit:
            # Calculate sleep time
            sleep_time = 60 - (now - self.calls[0]) + 0.1  # +0.1s buffer
            if sleep_time > 0:
                log_info(f"Polygon rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            # Re-clean after sleep
            now = time.time()
            self.calls = [t for t in self.calls if now - t < 60]
        
        self.calls.append(now)

# Global instance
_polygon_rate_limiter = PolygonRateLimiter(calls_per_minute=5)

# ===== Market Data =====
```

**Step 2:** Add random import at top (line 22):
```python
import sys
import time
import datetime as dt
import random  # <-- ADD THIS for jitter
from typing import List, Optional, Tuple, Dict
```

**Step 3:** In fetch_closes, update Polygon section (lines 267-302):
```python
# Fallback to Polygon
try:
    polygon_key = config.POLYGON_API_KEY
    if not polygon_key:
        return []
    
    # Enforce rate limit BEFORE making call
    _polygon_rate_limiter.wait_if_needed()
    
    snap = snap_interval_to_supported_seconds(interval_seconds)
    multiplier = 1
    timespan = "minute"
    
    if snap == 60:
        multiplier, timespan = 1, "minute"
    elif snap == 300:
        multiplier, timespan = 5, "minute"
    elif snap == 900:
        multiplier, timespan = 15, "minute"
    elif snap == 3600:
        multiplier, timespan = 1, "hour"
    else:
        multiplier, timespan = 4, "hour"
    
    end_date = dt.datetime.now(pytz.UTC)
    start_date = end_date - dt.timedelta(days=365)
    
    url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
           f"{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/"
           f"{end_date.strftime('%Y-%m-%d')}")
    
    # Retry with exponential backoff for 429 errors
    max_retries = 3
    for attempt in range(max_retries):
        resp = requests.get(url, params={"apiKey": polygon_key, "limit": limit_bars * 2}, timeout=15)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get("results"):
                closes = [float(r["c"]) for r in data["results"]]
                return closes[-limit_bars:] if len(closes) > limit_bars else closes
            break
        elif resp.status_code == 429:  # Rate limited
            if attempt < max_retries - 1:
                backoff = (2 ** attempt) + random.uniform(0, 1)  # Exponential + jitter
                log_warn(f"Polygon 429 rate limit, retry {attempt+1}/{max_retries} in {backoff:.1f}s")
                time.sleep(backoff)
            else:
                log_warn(f"Polygon rate limit: max retries exceeded for {symbol}")
                return []
        else:
            log_warn(f"Polygon error {resp.status_code} for {symbol}")
            break
            
except Exception as e:
    log_warn(f"Polygon fetch failed for {symbol}: {e}")
    pass

return []
```

---

### FIX 2.2 - Add Data Timestamp Validation

**File:** `runner.py`

**In fetch_closes**, add timestamp validation for yfinance (primary source). Update lines 230-238:

```python
if not hist.empty and 'Close' in hist.columns:
    closes = list(hist['Close'].values)
    
    # VALIDATE DATA FRESHNESS
    last_timestamp = hist.index[-1]
    now = dt.datetime.now(pytz.UTC)
    
    # Convert to UTC if needed
    if last_timestamp.tzinfo is None:
        last_timestamp = pytz.UTC.localize(last_timestamp)
    else:
        last_timestamp = last_timestamp.astimezone(pytz.UTC)
    
    age_minutes = (now - last_timestamp).total_seconds() / 60
    
    # During market hours, data should be recent. Outside hours, stale is OK.
    try:
        is_market_hours_now = in_market_hours(client)
    except:
        # If can't check market hours, be conservative
        is_market_hours_now = True
    
    max_age_minutes = 30 if is_market_hours_now else 1440  # 30min vs 24hr
    
    if age_minutes > max_age_minutes:
        # Data too old during market hours - don't use it
        if is_market_hours_now:
            # Fall through to next data source
            pass
        else:
            # Outside market hours, stale data is acceptable
            result = closes[-limit_bars:] if len(closes) > limit_bars else closes
            if len(result) > 0:
                return result
    else:
        # Data is fresh - use it
        result = closes[-limit_bars:] if len(closes) > limit_bars else closes
        if len(result) > 0:
            return result
```

**Add logging in buy_flow** to show data age (around line 403):
```python
closes = fetch_closes(client, symbol, interval_seconds, config.LONG_WINDOW + 50)
if closes:
    # Log data info for transparency
    log_info(f"  Using {len(closes)} bars for analysis")
```

---

### FIX 2.3 - Fix Simulation Look-Ahead Bias

**File:** `runner.py`

**Update simulate_signals_and_projection function** (lines 542-621). Change signature and add walk-forward logic:

```python
def simulate_signals_and_projection(
    closes: List[float],
    interval_seconds: int,
    override_tp_pct: Optional[float] = None,
    override_sl_pct: Optional[float] = None,
    override_trade_frac: Optional[float] = None,
    override_cap_usd: Optional[float] = None,
    use_walk_forward: bool = True  # NEW PARAMETER
) -> dict:
    
    # Minimum bars needed for SMA calculation
    min_bars = config.LONG_WINDOW + 2
    if len(closes) < min_bars:
        # Not enough data - return conservative estimates
        return {
            "win_rate": 0.5,
            "expected_trades_per_day": 2.0,
            "expected_daily_usd": 0.0
        }
    
    # Walk-forward validation to avoid look-ahead bias
    if use_walk_forward and len(closes) >= min_bars * 2:
        # Split: 70% train, 30% test
        split_point = int(len(closes) * 0.7)
        train_closes = closes[:split_point]
        test_closes = closes[split_point:]
        
        # Recursively simulate on both sets (with walk_forward=False to avoid infinite recursion)
        train_sim = simulate_signals_and_projection(
            train_closes, interval_seconds, override_tp_pct, override_sl_pct,
            override_trade_frac, override_cap_usd, use_walk_forward=False
        )
        
        test_sim = simulate_signals_and_projection(
            test_closes, interval_seconds, override_tp_pct, override_sl_pct,
            override_trade_frac, override_cap_usd, use_walk_forward=False
        )
        
        # Return weighted average (70% test, 30% train - emphasize out-of-sample)
        # This is more conservative and realistic
        return {
            "win_rate": train_sim["win_rate"] * 0.3 + test_sim["win_rate"] * 0.7,
            "expected_trades_per_day": train_sim["expected_trades_per_day"] * 0.3 + test_sim["expected_trades_per_day"] * 0.7,
            "expected_daily_usd": train_sim["expected_daily_usd"] * 0.3 + test_sim["expected_daily_usd"] * 0.7,
            # Include individual results for debugging
            "_train_return": train_sim["expected_daily_usd"],
            "_test_return": test_sim["expected_daily_usd"]
        }
    
    # Rest of existing simulation logic (unchanged)
    tp_pct = override_tp_pct if override_tp_pct is not None else config.TAKE_PROFIT_PERCENT
    sl_pct = override_sl_pct if override_sl_pct is not None else config.STOP_LOSS_PERCENT
    frac = override_trade_frac if override_trade_frac is not None else config.TRADE_SIZE_FRAC_OF_CAP
    cap = override_cap_usd if override_cap_usd is not None else config.MAX_CAP_USD
    
    # ... rest of existing code unchanged ...
```

**Add overfitting warning** in buy_flow (after simulation, around line 406):
```python
sim = simulate_signals_and_projection(closes, interval_seconds, override_cap_usd=available_cap)
expected_daily = float(sim.get("expected_daily_usd", 0.0))

# Check for potential overfitting
if "_train_return" in sim and "_test_return" in sim:
    train_ret = sim["_train_return"]
    test_ret = sim["_test_return"]
    if train_ret > test_ret * 1.5:
        log_warn(f"Possible overfitting: train=${train_ret:.2f} >> test=${test_ret:.2f}")
```

---

### FIX 2.4 - Add Slippage Simulation

**File:** `config.py`

Add after line 83:
```python
# Behavior
ALLOW_MISSING_KEYS_FOR_DEBUG: bool = os.getenv("ALLOW_MISSING_KEYS_FOR_DEBUG", "0") in ("1", "true", "True")
ENABLE_MARKET_HOURS_ONLY: bool = os.getenv("ENABLE_MARKET_HOURS_ONLY", "1") in ("1", "true", "True")

# Paper trading slippage simulation (to match live expectations)
SIMULATE_SLIPPAGE_ENABLED: bool = os.getenv("SIMULATE_SLIPPAGE_ENABLED", "1") in ("1", "true", "True")
SLIPPAGE_PERCENT: float = float(os.getenv("SLIPPAGE_PERCENT", "0.05"))  # 5 basis points
```

**File:** `runner.py`

In buy_flow, adjust price calculation (around line 421):
```python
qty = compute_order_qty_from_remaining(last_price, available_cap, frac)
if qty < 0.001:  # Minimum fractional share
    return (False, "Insufficient capital")

# Apply slippage simulation for paper trading to match live expectations
effective_price = last_price
if config.SIMULATE_SLIPPAGE_ENABLED and "paper" in config.ALPACA_BASE_URL.lower():
    effective_price = last_price * (1 + config.SLIPPAGE_PERCENT / 100)
    # Recalculate qty with slippage-adjusted price
    qty = compute_order_qty_from_remaining(effective_price, available_cap, frac)
    if qty < 0.001:
        return (False, "Insufficient capital after slippage")

try:
    # Use effective_price for TP/SL calculations
    tp_price = effective_price * (1 + tp / 100.0)
    sl_price = effective_price * (1 - sl / 100.0)
```

In sell_flow, add logging (around line 488):
```python
qty = float(pos["qty"])
realized_pl = pos["unrealized_pl"]

# Log expected slippage cost (can't change paper fills, but document it)
if config.SIMULATE_SLIPPAGE_ENABLED and "paper" in config.ALPACA_BASE_URL.lower():
    slippage_loss = pos["market_value"] * (config.SLIPPAGE_PERCENT / 100)
    log_info(f"Expected slippage: -${slippage_loss:.2f}")
```

---

### FIX 2.5 - Validate Risk Configuration

**File:** `config.py`

Add at end of file (after line 117):
```python
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
    
    # Check for redundant settings
    if PROFITABILITY_MIN_EXPECTED_USD > 0 and EXIT_ON_NEGATIVE_PROJECTION:
        errors.append(
            "Both PROFITABILITY_MIN_EXPECTED_USD and EXIT_ON_NEGATIVE_PROJECTION enabled (redundant)"
        )
    
    if errors:
        return "Configuration validation failed:\n  - " + "\n  - ".join(errors)
    return None

# Auto-validate on import (warn only, don't crash)
_validation_error = validate_risk_config()
if _validation_error and not ALLOW_MISSING_KEYS_FOR_DEBUG:
    import sys
    print(f"\n⚠️  WARNING: {_validation_error}\n", file=sys.stderr)
```

**File:** `runner.py`

In main(), add validation check (around line 949):
```python
try:
    client = make_client(allow_missing=args.allow_missing_keys, go_live=args.go_live)
except Exception as e:
    log_warn(f"Failed to create client: {e}")
    return 2

# Validate configuration
config_error = config.validate_risk_config()
if config_error:
    log_error(config_error)
    if not SCHEDULED_TASK_MODE:
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return 1
```

---

## PHASE 3: MEDIUM SEVERITY (Improve Reliability)

### FIX 3.1 - Add Holding Period to Rebalancing

**File:** `config.py`

Add after line 100:
```python
# Profitability/Confidence gates
PROFITABILITY_GATE_ENABLED: bool = os.getenv("PROFITABILITY_GATE_ENABLED", "1") in ("1", "true", "True")
PROFITABILITY_MIN_EXPECTED_USD: float = float(os.getenv("PROFITABILITY_MIN_EXPECTED_USD", "0.01") or 0.0)
STRONG_CONFIDENCE_THRESHOLD: float = float(os.getenv("STRONG_CONFIDENCE_THRESHOLD", "0.08"))
STRONG_CONFIDENCE_BYPASS_ENABLED: bool = os.getenv("STRONG_CONFIDENCE_BYPASS_ENABLED", "1") in ("1", "true", "True")

# Rebalancing constraints
MIN_HOLDING_PERIOD_HOURS: float = float(os.getenv("MIN_HOLDING_PERIOD_HOURS", "2.0"))
REBALANCE_THRESHOLD_PERCENT: float = float(os.getenv("REBALANCE_THRESHOLD_PERCENT", "15.0"))
```

**File:** `portfolio_manager.py`

Update update_position to track first_opened (around line 46):
```python
def update_position(self, symbol: str, qty: float, avg_entry: float, 
                   market_value: float, unrealized_pl: float):
    """Update or add a position."""
    # Preserve first_opened if position already exists
    first_opened = self.positions.get(symbol, {}).get("first_opened")
    if first_opened is None:
        first_opened = datetime.now(pytz.UTC).isoformat()
    
    self.positions[symbol] = {
        "qty": qty,
        "avg_entry": avg_entry,
        "market_value": market_value,
        "unrealized_pl": unrealized_pl,
        "last_update": datetime.now(pytz.UTC).isoformat(),
        "first_opened": first_opened  # Track when position was first opened
    }
    self.save()
```

**File:** `runner.py`

In main loop, add holding period check before rebalancing (around line 1117):
```python
# Check if we need to sell excess (over-allocated)
if existing_pos and allocation_diff < -10:
    # NEW: Check holding period before selling
    pos_obj = portfolio.get_position(sym)
    if pos_obj and "first_opened" in pos_obj:
        try:
            opened_time = dt.datetime.fromisoformat(pos_obj["first_opened"])
            if opened_time.tzinfo is None:
                opened_time = pytz.UTC.localize(opened_time)
            held_hours = (dt.datetime.now(pytz.UTC) - opened_time).total_seconds() / 3600
            
            if held_hours < config.MIN_HOLDING_PERIOD_HOURS:
                log_info(f"  HOLD: Won't rebalance {sym} (held {held_hours:.1f}h < {config.MIN_HOLDING_PERIOD_HOURS}h min)")
                continue  # Skip rebalancing this position
        except Exception as e:
            pass  # If timestamp parsing fails, proceed with rebalance
    
    # Check if this stock is still in optimal portfolio
    if sym in stock_allocations and stock_allocations[sym] > 10:
        # Stock is still in optimal portfolio, keep the excess
        log_info(f"  HOLD: Over-allocated by ${-allocation_diff:.2f} but still profitable - keeping")
    else:
        # Stock is no longer optimal, sell it
        log_info(f"  REBALANCE: Over-allocated by ${-allocation_diff:.2f} and not optimal - selling")
        ok, msg = sell_flow(client, sym)
        if ok:
            portfolio.remove_position(sym)
            log_info(f"  OK {msg}")
        else:
            log_info(f"  -- {msg}")
        continue
```

---

### FIX 3.2 - Unify Safety Checks

**File:** `runner.py`

Replace enforce_safety function (line 524) with comprehensive version:
```python
def enforce_all_safety_checks(client, symbol: str = None, portfolio = None) -> Tuple[bool, str]:
    """
    Check all safety limits. Returns (should_shutdown, reason).
    If should_shutdown=True, bot should immediately liquidate and exit.
    """
    try:
        account = client.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        
        if last_equity <= 0:
            return (False, "")  # Can't calculate on first run
        
        # Check 1: Daily loss percentage
        daily_pnl_pct = ((equity - last_equity) / last_equity) * 100
        if daily_pnl_pct < -config.MAX_DAILY_LOSS_PERCENT:
            return (True, f"Daily loss % limit: {daily_pnl_pct:.2f}% < -{config.MAX_DAILY_LOSS_PERCENT}%")
        
        # Check 2: Daily loss absolute USD
        daily_pnl_usd = equity - last_equity
        if daily_pnl_usd < -config.DAILY_LOSS_LIMIT_USD:
            return (True, f"Daily loss $ limit: ${daily_pnl_usd:.2f} < -${config.DAILY_LOSS_LIMIT_USD}")
        
        return (False, "")  # All checks passed
        
    except Exception as e:
        log_warn(f"Safety check error: {e}")
        return (False, "")  # Don't shutdown on check failure

# Keep old name for compatibility (just call new function)
def enforce_safety(client, symbol: str):
    should_stop, reason = enforce_all_safety_checks(client, symbol)
    if should_stop:
        log_warn(reason)
        sell_flow(client, symbol)
        sys.exit(1)
```

---

### FIX 3.3 - Add Profitability Freeze Detection

**File:** `runner.py`

In main(), add state tracking at start (around line 992):
```python
log_info(f"{'='*70}\n")

iteration = 0
consecutive_no_trade_cycles = 0  # Track cycles with no trading activity
MAX_NO_TRADE_CYCLES = 20  # Alert after 20 idle cycles

try:
```

In multi-stock loop, track trades (around line 1136):
```python
# Trade each stock
log_info(f"\nTrading {len(stocks_to_evaluate)} stocks:")
trades_this_cycle = 0  # Count trades this iteration

for i, sym in enumerate(stocks_to_evaluate, 1):
    # ... existing trading logic ...
    
    if action == "buy":
        # ... existing code ...
        ok, msg = buy_flow(...)
        if ok:
            trades_this_cycle += 1
        log_info(f"  {'OK' if ok else '--'} {msg}")
    elif action == "sell":
        ok, msg = sell_flow(client, sym)
        if ok:
            trades_this_cycle += 1
        log_info(f"  {'OK' if ok else '--'} {msg}")

# After trading loop, check for freeze
if trades_this_cycle == 0:
    consecutive_no_trade_cycles += 1
    if consecutive_no_trade_cycles >= MAX_NO_TRADE_CYCLES:
        hours_idle = (consecutive_no_trade_cycles * interval_seconds) / 3600
        log_warn(f"⚠️  NO TRADES for {hours_idle:.1f}h ({consecutive_no_trade_cycles} cycles)")
        log_warn(f"⚠️  Market may be unfavorable. Consider:")
        log_warn(f"     - Checking if market is trending")
        log_warn(f"     - Adjusting parameters")
        log_warn(f"     - Pausing bot until conditions improve")
        consecutive_no_trade_cycles = 0  # Reset
else:
    consecutive_no_trade_cycles = 0  # Reset on any trade
```

---

### FIX 3.4 - Add Unit Tests

**Create new file:** `test_runner.py`

```python
#!/usr/bin/env python3
"""
Unit tests for Paper Trading Bot
Run with: pytest test_runner.py -v
Install: pip install pytest
"""

import pytest
from runner import (
    sma, decide_action, compute_confidence, pct_stddev,
    compute_order_qty_from_remaining, adjust_runtime_params
)
import config

def test_sma_basic():
    """Test simple moving average calculation"""
    closes = [100, 102, 104, 106, 108]
    assert sma(closes, 3) == 106.0
    assert sma(closes, 5) == 104.0

def test_sma_insufficient_data():
    """Test SMA with insufficient data"""
    closes = [100, 102]
    assert sma(closes, 5) == 102

def test_decide_action_buy():
    """Test buy signal when short MA > long MA"""
    closes = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122]
    action = decide_action(closes, short_w=3, long_w=9)
    assert action == "buy"

def test_decide_action_sell():
    """Test sell signal when short MA < long MA"""
    closes = [120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98]
    action = decide_action(closes, short_w=3, long_w=9)
    assert action == "sell"

def test_decide_action_hold():
    """Test hold signal when MAs are close"""
    closes = [100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100, 101]
    action = decide_action(closes, short_w=3, long_w=9)
    assert action == "hold"

def test_compute_confidence():
    """Test confidence calculation"""
    closes = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122,
              124, 126, 128, 130, 132, 134, 136, 138, 140]
    conf = compute_confidence(closes)
    assert conf > 0.05

def test_pct_stddev_low_volatility():
    """Test percentage standard deviation with low volatility"""
    closes = [100, 101, 100, 101, 100]
    vol = pct_stddev(closes)
    assert 0 < vol < 0.01

def test_pct_stddev_high_volatility():
    """Test percentage standard deviation with high volatility"""
    closes = [100, 120, 80, 110, 90]
    vol = pct_stddev(closes)
    assert vol > 0.1

def test_compute_order_qty():
    """Test order quantity calculation"""
    qty = compute_order_qty_from_remaining(100.0, 1000.0, 0.5)
    assert qty == 5.0
    
    qty = compute_order_qty_from_remaining(100.0, 550.0, 1.0)
    assert qty == 5.5

def test_adjust_runtime_params_basic():
    """Test runtime parameter adjustment"""
    tp, sl, frac = adjust_runtime_params(
        confidence=0.05, base_tp=3.0, base_sl=1.0, base_frac=0.5
    )
    assert tp >= 3.0
    assert sl >= 1.0
    assert frac >= 0.5

def test_adjust_runtime_params_bounds():
    """Test params stay within bounds"""
    tp, sl, frac = adjust_runtime_params(
        confidence=10.0, base_tp=20.0, base_sl=15.0, base_frac=0.95
    )
    assert tp <= config.MAX_TAKE_PROFIT_PERCENT
    assert sl <= config.MAX_STOP_LOSS_PERCENT
    assert frac <= config.MAX_TRADE_SIZE_FRAC

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Install pytest:**
```bash
pip install pytest
```

---

## PHASE 4: LOW SEVERITY & POLISH

### FIX 4.1 - Market Hours Fallback

**File:** `runner.py` (line 629)

```python
def in_market_hours(client) -> bool:
    try:
        clock = client.get_clock()
        return clock.is_open
    except Exception as e:
        # Don't guess - if API fails, conservatively assume closed
        log_warn(f"Could not check market hours: {e}. Assuming CLOSED for safety.")
        return False
```

---

### FIX 4.2 - Add Volume Filter

**File:** `config.py` (add after line 117, before validate function):
```python
# Stock filtering
MIN_AVG_VOLUME: int = int(os.getenv("MIN_AVG_VOLUME", "1000000"))  # 1M shares/day minimum
```

**File:** `stock_scanner.py` (in score_stock, after line 176):
```python
def score_stock(symbol: str, interval_seconds: int, cap_per_stock: float, bars: int = 200, verbose: bool = False) -> Optional[Dict]:
    """Score a stock based on profitability potential."""
    try:
        # Add volume filter to avoid illiquid stocks
        import yfinance as yf
        ticker_obj = yf.Ticker(symbol)
        try:
            info = ticker_obj.info
            avg_volume = info.get('averageVolume', 0)
            if avg_volume < config.MIN_AVG_VOLUME:
                if verbose:
                    print(f"  {symbol}: Low volume ({avg_volume:,} < {config.MIN_AVG_VOLUME:,})")
                return None
        except:
            pass  # If volume check fails, continue anyway
        
        client = make_client(allow_missing=False, go_live=False)
        closes = fetch_closes(client, symbol, interval_seconds, bars)
        # ... rest of existing code ...
```

---

### FIX 4.3 - Conservative Preset System

**File:** `config.py` (add at very end, after validate function):

```python
# Preset configurations
def apply_conservative_preset():
    """Apply safe defaults for new users"""
    global TRADE_SIZE_FRAC_OF_CAP, RISKY_MODE_ENABLED, TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT
    TRADE_SIZE_FRAC_OF_CAP = 0.30  # 30% per position
    RISKY_MODE_ENABLED = False
    TAKE_PROFIT_PERCENT = 2.0
    STOP_LOSS_PERCENT = 1.0
    print("✅ Applied CONSERVATIVE preset (recommended for beginners)")

def apply_aggressive_preset():
    """Apply maximum profit settings (high risk!)"""
    global TRADE_SIZE_FRAC_OF_CAP, RISKY_MODE_ENABLED, TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT
    TRADE_SIZE_FRAC_OF_CAP = 0.75
    RISKY_MODE_ENABLED = True
    TAKE_PROFIT_PERCENT = 3.0
    STOP_LOSS_PERCENT = 1.5
    print("⚠️  Applied AGGRESSIVE preset (high risk, experienced users only)")

# Auto-apply based on environment variable
PRESET_MODE = os.getenv("PRESET_MODE", "").upper()
if PRESET_MODE == "CONSERVATIVE":
    apply_conservative_preset()
elif PRESET_MODE == "AGGRESSIVE":
    apply_aggressive_preset()
```

---

### FIX 4.4 - Add Risk Disclosure

**File:** `README.md` (add at very top, line 1):

```markdown
# Paper Trading Bot - Intelligent Multi-Stock Trading System

⚠️ **CRITICAL RISK DISCLOSURE** ⚠️

**READ THIS BEFORE USING THE BOT**

- **This software is for EDUCATIONAL and PAPER TRADING purposes**
- **Trading involves substantial risk of loss**
- **Past performance does NOT guarantee future results**
- **Paper trading uses instant fills and perfect liquidity that DO NOT exist in live markets**
- **Expect 20-50% performance degradation when moving from paper to live trading**
- **Never trade with money you cannot afford to lose completely**
- **This software is provided AS-IS with NO warranties or guarantees**
- **Authors are NOT responsible for any trading losses**

**Realistic Return Expectations:**
- S&P 500 historical average: ~10% per YEAR
- Professional hedge funds: 15-25% per YEAR in good years
- High-frequency trading bots: Often lose money or break even after costs
- **This bot's backtests are NOT predictive of live performance**
- **Realistic target: 5-20% annual return at best, with significant risk**
- **Most retail algo traders lose money - approach with extreme caution**

**Minimum Requirements Before Live Trading:**
1. Paper trade for 3-6 months minimum
2. Understand all risks and have emergency stop procedures
3. Start with absolute minimum capital ($100-500 max)
4. Monitor closely for first month
5. Expect losses and have stop-loss discipline

---

**Automated stock trading with smart capital allocation and portfolio rebalancing**

## 🎯 What This Bot Does
```

---

### FIX 4.5 - Add Compounding Calculator to Optimizer

**File:** `optimizer.py` (add before final return in main(), around line 302):

```python
# Show optimal config
print(f"\n{'='*70}")
print(f"OPTIMAL CONFIGURATION")
print(f"{'='*70}")
print(f"Symbol: {best_symbol}")
print(f"Interval: {best_interval}s ({best_interval/3600:.4f}h)")
print(f"Capital: ${best_cap:.2f}")
print(f"Expected Daily Return: ${best_return:.2f}")

# NEW: Compounding projections
if best_return > 0:
    print(f"\n{'='*70}")
    print(f"COMPOUNDING PROJECTIONS (THEORETICAL)")
    print(f"{'='*70}")
    
    daily_return_pct = (best_return / best_cap) * 100
    
    print(f"Starting capital: ${best_cap:.2f}")
    print(f"Daily return: ${best_return:.2f} ({daily_return_pct:.3f}%/day)")
    print(f"\nProjected balance after:")
    
    for months in [1, 3, 6, 12, 24, 60]:
        trading_days = months * 20
        final = best_cap * ((1 + daily_return_pct/100) ** trading_days)
        gain = final - best_cap
        gain_pct = (gain / best_cap) * 100
        print(f"  {months:2d} months ({trading_days:3d} days): ${final:>12,.2f}  (+{gain_pct:>6.1f}%)")
    
    print(f"\n⚠️  WARNING: These are THEORETICAL backtested projections.")
    print(f"⚠️  Real trading performance will be 20-50% LOWER due to:")
    print(f"     • Slippage (0.05-0.2% per trade)")
    print(f"     • Partial fills and rejected orders")
    print(f"     • Market regime changes")
    print(f"     • Losing streaks and drawdowns")
    print(f"     • Competition and market efficiency")
    print(f"\n💡 Realistic expectation: 5-20% per YEAR, not per day")

print(f"\n{'='*70}")
```

---

## VALIDATION & TESTING

After implementing all fixes, run these checks in order:

### 1. Config Validation
```bash
python -c "import config; result = config.validate_risk_config(); print('Config OK' if result is None else result)"
```

### 2. Import Check
```bash
python -c "import runner; print('Runner import: OK')"
python -c "import stock_scanner; print('Scanner import: OK')"
python -c "import portfolio_manager; print('Portfolio import: OK')"
python -c "import optimizer; print('Optimizer import: OK')"
```

### 3. Unit Tests
```bash
pip install pytest
pytest test_runner.py -v
```

### 4. Dry Run Test
```bash
# Test single stock mode (let run for 2-3 iterations, then Ctrl+C)
python runner.py -t 0.25 -s AAPL -m 100 --allow-missing-keys

# Check output for:
# - No errors or crashes
# - Idempotency messages
# - Data freshness validation
# - Slippage simulation logs
```

### 5. Configuration File Audit
Check that all new variables exist:
- [ ] `MAX_DAILY_LOSS_PERCENT` in config.py
- [ ] `SIMULATE_SLIPPAGE_ENABLED` in config.py
- [ ] `SLIPPAGE_PERCENT` in config.py
- [ ] `MIN_HOLDING_PERIOD_HOURS` in config.py
- [ ] `REBALANCE_THRESHOLD_PERCENT` in config.py
- [ ] `MIN_AVG_VOLUME` in config.py
- [ ] `validate_risk_config()` function exists
- [ ] Preset functions exist

### 6. Fractional Share Test
```bash
# Test with capital that results in fractional shares
python runner.py -t 0.25 -s AAPL -m 55 --allow-missing-keys
# Verify buy and sell handle fractional quantities
```

### 7. Rate Limit Test
```bash
# Scan many stocks to trigger rate limiting
python scan_best_stocks.py -n 20 -v
# Should see "Polygon rate limit: sleeping X.Xs" messages
```

---

## SUCCESS CRITERIA CHECKLIST

Mark each as complete after verifying:

**Phase 1 (Critical):**
- [ ] Bot starts without AttributeError (MAX_DAILY_LOSS_PERCENT exists)
- [ ] Can buy fractional shares (qty stays float)
- [ ] Can sell fractional shares (TIF = 'day')
- [ ] Orders have client_order_id (idempotency protection)
- [ ] Orders wait for fill confirmation

**Phase 2 (High Priority):**
- [ ] Polygon rate limiter activates (see log messages)
- [ ] Data timestamp validated and logged
- [ ] Simulation uses train/test split (see _train_return in output)
- [ ] Slippage simulated in paper mode (see adjusted prices)
- [ ] Risk config validated on startup (see warnings if misconfigured)

**Phase 3 (Medium Priority):**
- [ ] Positions track first_opened timestamp
- [ ] Rebalancing respects holding period
- [ ] Safety checks unified (all limits checked)
- [ ] Freeze detection alerts after 20 idle cycles
- [ ] Unit tests pass (`pytest test_runner.py`)

**Phase 4 (Polish):**
- [ ] Market hours fallback doesn't guess timezone
- [ ] Volume filter rejects low-volume stocks
- [ ] Conservative preset available (PRESET_MODE=CONSERVATIVE)
- [ ] README has risk disclosure at top
- [ ] Optimizer shows compounding calculator

**Overall:**
- [ ] All imports work
- [ ] Config validation passes
- [ ] Dry run completes without errors
- [ ] Logs show new features working (rate limiting, slippage, etc.)
- [ ] Code is well-commented
- [ ] Git committed after each phase

---

## FINAL NOTES

### Implementation Strategy
1. **Work sequentially** - Complete Phase 1 fully before moving to Phase 2
2. **Test after each fix** - Don't accumulate untested changes
3. **Use git** - Commit after each phase with message like "Phase 1: Critical bugs fixed"
4. **Read existing code** - Understand context before changing
5. **Preserve logic** - Only fix bugs, don't refactor unnecessarily

### Common Pitfalls
- Don't skip the idempotency global variable declaration
- Don't forget to import `uuid` and `random`
- Remember to update BOTH fractional and whole-share order paths
- Apply fill reconciliation to both buy_flow and sell_flow
- Test with paper trading API only (don't accidentally go live)

### After All Fixes
- Paper trade for 3-6 months minimum
- Monitor performance vs. expectations
- Start live trading with minimum capital ($100-500)
- Expect 20-50% worse performance than paper
- Have emergency stop procedures ready
- Never risk more than you can afford to lose

### Getting Help
If any fix is unclear:
1. Read the surrounding code context
2. Check function signatures and existing patterns
3. Run the dry run test after each fix to catch errors early
4. Review the detailed TODO file (BUGS_AND_FIXES_TODO.md) for more context

---

## SUMMARY

This prompt provides **complete step-by-step fixes for 20 bugs** across 4 priority phases:
- **Phase 1:** 5 critical bugs (will crash)
- **Phase 2:** 5 high severity bugs (silent failures)
- **Phase 3:** 5 medium severity bugs (reliability)
- **Phase 4:** 5 low severity bugs (polish)

**Time estimate:** 3-6 hours for complete implementation and testing

**Result:** Production-ready paper trading bot with proper error handling, rate limiting, slippage simulation, and safety checks.

**Remember:** This bot is for PAPER TRADING. Live trading requires 3-6 months of successful paper trading first, starting with minimum capital.

Good luck! 🚀

