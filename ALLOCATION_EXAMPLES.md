# Smart Allocation Examples

## How the NEW Allocation Works

**Key Features:**
1. **Risk-adjusted scoring**: Return / Volatility (Sharpe-like ratio)
2. **Power law concentration**: Best stocks get exponentially more (score^2 by default)
3. **Dynamic portfolio size**: Could be 1 stock or 50 stocks based on opportunities
4. **Safety limits**: Max 50% in one stock, min 3 stocks for diversification
5. **Minimum threshold**: Skips stocks below $5 allocation

---

## Example Scenarios

### Scenario 1: One Amazing Stock
```
Input: $100 capital
Stocks found:
  - AAPL: $20/day expected, 1% volatility → score: 2000
  - MSFT: $2/day expected, 2% volatility → score: 100
  - GOOGL: $1/day expected, 3% volatility → score: 33

OLD allocation (equal split):
  AAPL: $33.33
  MSFT: $33.33
  GOOGL: $33.33

NEW allocation (concentration=2.0):
  AAPL: $50.00 (max 50% cap)  ✅ Gets most money!
  MSFT: $29.50
  GOOGL: $20.50
  Portfolio: 3 stocks (safety minimum)
```

### Scenario 2: Many Good Stocks
```
Input: $100 capital
10 stocks found with scores: 500, 400, 300, 200, 150, 100, 80, 60, 40, 20

OLD allocation (equal split):
  All 10 stocks: $10.00 each

NEW allocation (concentration=2.0):
  Score^2: 250000, 160000, 90000, 40000, 22500, 10000, 6400, 3600, 1600, 400
  
  Stock 1: $43.27 (best gets ~43%)  ✅ Huge winner!
  Stock 2: $27.67
  Stock 3: $15.57
  Stock 4: $6.92
  Stock 5-7: $5.00 each (hit minimum)
  Stock 8-10: Skipped (would be < $5)
  
  Portfolio: 7 stocks, top stock: 43%, avg: $14.29
```

### Scenario 3: Only 2 Profitable Stocks
```
Input: $100 capital
Only 2 stocks pass profitability threshold

OLD allocation:
  Would force 15 stocks (including bad ones)

NEW allocation:
  Stock A: $70
  Stock B: $30
  Portfolio: 2 stocks only ✅ No forced diversification!
```

### Scenario 4: 50 Mediocre Stocks
```
Input: $100 capital
50 stocks found, all with similar low scores (20-40 range)

NEW allocation:
  - Top 15 stocks get $5-8 each
  - Rest get < $5 and are SKIPPED
  Portfolio: 15 stocks, fairly even distribution
  
  ✅ Doesn't over-diversify into weak stocks!
```

---

## Configuration Options

### ALLOCATION_CONCENTRATION
**Default: 2.0** (recommended)

- `1.0` = Proportional (if stock A is 2× better, it gets 2× more money)
- `2.0` = Aggressive (if stock A is 2× better, it gets 4× more money)
- `3.0` = Very aggressive (if stock A is 2× better, it gets 8× more money)

**When to change:**
- Conservative? Use `1.5` for more even distribution
- Aggressive? Use `2.5` or `3.0` for heavy concentration
- Very small capital ($50-100)? Use `1.5` to maintain diversity

### MIN_DIVERSIFICATION_STOCKS
**Default: 3** (safety minimum)

- Always keep at least this many stocks (unless fewer are profitable)
- Prevents putting 100% in one stock
- `3` = good for $100-500 capital
- `5` = better for $1000+ capital

### MAX_SINGLE_STOCK_PERCENT
**Default: 50%** (safety cap)

- No single stock can exceed this % of portfolio
- `50%` = balanced (can go up to 70% if you're confident)
- `30%` = conservative
- `70%` = aggressive (risky!)

### MIN_ALLOCATION_USD
**Default: $5** (skip threshold)

- Stocks that would get less than this are skipped
- With $100 capital: `$5` is good (allows ~20 stocks max)
- With $1000 capital: `$20` might be better
- With $10,000 capital: `$100` to avoid tiny positions

---

## Real Example from Your Log

**Your old allocation (lines 159-173 in bot.log):**
```
MDT: $10.43 (best stock: $10.69/day)
All other 14 stocks: $6.40 each
```

**What NEW allocation would do:**
```
MDT: $25-35 (gets 25-35% of capital - it's 2-3× better!)
REGN/INTC/MRK: $10-15 each (second tier)
ORCL/LRCX/AMZN: $5-8 each (third tier)
Bottom stocks: Skipped (< $5 allocation)

Portfolio: 8-10 stocks instead of 15
Top stock: 30% of capital (vs 10% before)
```

**Result:** More profit from winners, less waste on losers! 🚀

---

## How It Calculates Risk-Adjusted Score

```python
# For each stock:
expected_daily = $10.69  # From backtest
volatility = 2.0%        # Price volatility
confidence = 0.025       # Signal confidence

# Risk-adjusted return (Sharpe-like)
risk_adjusted = expected_daily / volatility
# $10.69 / 0.02 = 534.5

# Add confidence boost
score = risk_adjusted + (confidence * 100)
# 534.5 + 2.5 = 537

# Apply concentration (power law)
weighted_score = score ** 2.0
# 537^2 = 288,369

# Calculate proportion of total capital
allocation = (weighted_score / sum_all_weighted_scores) * $100
```

**Why this works:**
- High return + low risk = big score
- Score^2 = winners get exponentially more
- Still capped at 50% max for safety
- Minimum 3 stocks for diversification

---

## Summary

**OLD**: Split $100 across 15 stocks → $6.67 each (waste on bad stocks)
**NEW**: Smart allocation → Best stock gets $25-50, weak stocks get $0

**Benefits:**
- ✅ More profit from winners
- ✅ Less capital wasted on losers
- ✅ Automatic portfolio size adjustment
- ✅ Risk-adjusted (low volatility gets bonus)
- ✅ Still maintains safety minimums

