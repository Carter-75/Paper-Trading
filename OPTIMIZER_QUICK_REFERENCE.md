# Optimizer Quick Reference

## How to Read Optimizer Output

### The TWO Values You Need

The optimizer output shows these critical values clearly:

```
>>> YOU SET THESE 2 VALUES:
  1. Time Interval: 14400s (4.0000h)        <-- USE THIS
  2. Total Capital: $198699.39              <-- USE THIS
```

### How to Run the Bot

Copy the command from the optimizer output:

```powershell
$BotDir = 'C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading'

# EXACT command (replace with your values):
& "$BotDir\botctl.ps1" start python -u runner.py -t 4.0000 -m 198699.39
```

**Format:**
- `-t` = Time interval in **hours** (not seconds!)
- `-m` = Max capital in **dollars** (with decimals)

### Quick Conversions

The optimizer shows seconds, but `-t` flag uses **hours**:

| Seconds | Hours | Command Flag |
|---------|-------|--------------|
| 900s    | 0.25h | `-t 0.25`    |
| 1800s   | 0.50h | `-t 0.5`     |
| 3600s   | 1.00h | `-t 1.0`     |
| 7200s   | 2.00h | `-t 2.0`     |
| 14400s  | 4.00h | `-t 4.0`     |
| 21600s  | 6.00h | `-t 6.0`     |

### Example Commands

```powershell
# Conservative (small capital, frequent trades)
& "$BotDir\botctl.ps1" start python -u runner.py -t 0.25 -m 100

# Balanced (medium capital, hourly trades)
& "$BotDir\botctl.ps1" start python -u runner.py -t 1.0 -m 1000

# Aggressive (large capital, 4-hour intervals)
& "$BotDir\botctl.ps1" start python -u runner.py -t 4.0 -m 10000
```

### Understanding the Output

**Live Trading (REALISTIC):** This is your expected daily profit
```
Live Trading (REALISTIC):  $479.85/day  <<< USE THIS
```

**Expected Returns:** Conservative estimates with all real-world costs included
```
1 month  ( 20 days): +$7,422 (+3.7%)
3 months ( 60 days): +$20,557 (+10.3%)
```

**Consistency Score:** How reliable the strategy is
```
Consistency: 1.00 (VERY HIGH confidence)  <-- Want this HIGH
```

### Common Issues

❌ **Wrong:** `-t 14400` (using seconds instead of hours)
✅ **Correct:** `-t 4.0` (using hours)

❌ **Wrong:** `-m 198,699.39` (using commas)
✅ **Correct:** `-m 198699.39` (no commas)

❌ **Wrong:** Using "Paper Trading" value ($6710/day)
✅ **Correct:** Using "Live Trading" value ($479/day)

### What If Bot Isn't Trading?

Check bot.log for these issues:

1. **Low Confidence:** `conf=0.0001` means weak signals
   - Solution: Run optimizer with different interval
   - Try `-t 1.0` or `-t 2.0` instead

2. **All Stocks Negative:** Market might be in downtrend
   - Solution: Wait for better market conditions
   - Or reduce capital and try shorter intervals

3. **Volume Filters:** `skipped (low volume: 576,904)`
   - Solution: This is correct - avoiding illiquid stocks
   - Bot needs 1M+ daily volume for safety

### Need Help?

Run optimizer with verbose flag to see detailed analysis:
```powershell
python optimizer.py -s AAPL -v
```

This shows WHY a stock is or isn't profitable.

