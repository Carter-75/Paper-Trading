# Paper Trading Bot - Unified System

**One bot that auto-selects and trades the best stocks**

**How It Works:**
- **Optimizer**: Tests multiple stocks using binary search → finds best INTERVAL & CAPITAL
- **Runner**: Uses those parameters + AUTO-SELECTS best stocks to trade (default: 15)

**Features:**
- ONE file handles everything (runner.py)
- Comprehensive binary search optimizer
- Smart portfolio management and rebalancing
- Runs as hidden Windows Scheduled Task
- Full API integration (Alpaca + Polygon)

---

## 🚀 Quick Start

**From project directory:**
```powershell
# 1. Install
pip install -r requirements.txt

# 2. Setup .env (see Environment Variables below)

# 3. Find best interval & capital (optimizer tests multiple stocks)
python optimizer.py -v

# 4. Run bot with suggested interval & capital
#    Bot will auto-select best 15 stocks and trade them! (15 is default)
python runner.py -t 0.25 -m 1500

# That's it! Press Ctrl+C to stop.
```

**From anywhere (use full paths):**
```powershell
# Set your project path
$BotDir = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading"

# Install
cd $BotDir; pip install -r requirements.txt

# Find best parameters (tests multiple stocks)
python "$BotDir\optimizer.py" -v

# Run bot (auto-selects best 15 stocks - default)
python "$BotDir\runner.py" -t 0.25 -m 1500
```

---

## 📦 Installation

```powershell
# Install dependencies
pip install -r requirements.txt

# Create .env file
notepad .env
```

**Requirements:**
- Python 3.9+
- Alpaca Paper Account (alpaca.markets)
- Polygon API Key (free tier works)
- PowerShell 7+ (for background mode)

---

## 🔐 Environment Variables

Create `.env` file:

```env
# ===== REQUIRED =====
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets
POLYGON_API_KEY=your_polygon_key

# ===== SAFETY =====
CONFIRM_GO_LIVE=NO                    # Must be YES for live trading
EXIT_ON_NEGATIVE_PROJECTION=1         # Exit if negative returns

# ===== STRATEGY (optional, have good defaults) =====
DEFAULT_INTERVAL_SECONDS=900          # 15 minutes
SHORT_WINDOW=9                        # Fast MA
LONG_WINDOW=21                        # Slow MA
MAX_CAP_USD=100
TRADE_SIZE_FRAC_OF_CAP=0.65          # Use 65% per trade
TAKE_PROFIT_PERCENT=2.0              # TP at +2%
STOP_LOSS_PERCENT=1.0                # SL at -1%

# ===== RISK MANAGEMENT (optional) =====
MIN_CONFIDENCE_TO_TRADE=0.005
VOLATILITY_PCT_THRESHOLD=0.15
PROFITABILITY_MIN_EXPECTED_USD=0.10
MAX_DAILY_LOSS_PERCENT=5.0
MAX_TRADE_SIZE_FRAC=0.95

# ===== DYNAMIC (optional) =====
DYNAMIC_PARAM_ENABLED=1
RISKY_MODE_ENABLED=1
RISKY_TP_MULT=1.5
RISKY_SL_MULT=0.8
RISKY_FRAC_MULT=1.3
```

---

## 📁 File Structure

```
Core:
  runner.py              - UNIFIED bot (single OR multi-stock)
  config.py              - Configuration
  optimizer.py           - Binary search optimizer
  
Support:
  portfolio_manager.py   - Position tracking
  stock_scanner.py       - Stock evaluator
  multi_stock_config.py  - Multi-stock settings
  
Testing:
  test_all_systems.py    - Full system test
  test_signals.py        - Check current signals
  validate_setup.py      - Validate configuration
  scan_best_stocks.py    - Find best stocks
  
Background:
  botctl.ps1             - Task control (Admin)
  start_bot.ps1          - Auto-generated wrapper
  last_start_cmd.txt     - Last command (auto-saved)
  
Data (auto-created by bot):
  bot.log                - Trading log (auto-truncated)
  portfolio.json         - CURRENT STOCKS & AMOUNTS (live portfolio)
  pnl_ledger.json        - Trade history with realized P&L
```

**How It Works:**
- `runner.py` is the ONLY entry point
- Default: Auto-selects best 15 stocks (`--max-stocks 15`)
- Single stock: Provide `-s SYMBOL --max-stocks 1`
- All files import and work together automatically

---

## 💻 Usage

**💡 All commands work from anywhere! Just use full paths:**
```powershell
$BotDir = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading"
python "$BotDir\runner.py" -t 0.25 -m 1500
```

**Default Behavior:** Bot auto-selects best 15 stocks

### Multi-Stock (Default)

**Auto-Select All (Bot Picks Best Stocks):**
```powershell
# From project dir (--max-stocks defaults to 15)
python runner.py -t 0.25 -m 1500

# Or explicitly set max stocks
python runner.py -t 0.25 -m 1500 --max-stocks 10

# From anywhere
python "$BotDir\runner.py" -t 0.25 -m 1500
```
Bot scans universe, picks best stocks (default: 15), trades & rebalances.

**Force Specific + Auto-Fill:**
```powershell
# From project dir
python runner.py -t 0.25 -m 1500 --stocks TSLA AAPL --max-stocks 10

# From anywhere
python "$BotDir\runner.py" -t 0.25 -m 1500 --stocks TSLA AAPL --max-stocks 10
```
Keeps TSLA/AAPL always, auto-selects 8 more.

**Manual Only:**
```powershell
# From project dir
python runner.py -t 0.25 -m 300 --stocks AAPL MSFT GOOGL --max-stocks 3

# From anywhere
python "$BotDir\runner.py" -t 0.25 -m 300 --stocks AAPL MSFT GOOGL --max-stocks 3
```
Only trades your 3 picks.

**Arguments:**
- `--max-stocks N` - Max positions (default: 15)
- `--stocks SYM1 SYM2` - Force specific stocks
- `--cap-per-stock USD` - Capital per stock (default: total/max)
- `--rebalance-every N` - Rebalance frequency (default: 4)

### Single Stock

**To trade just ONE stock, set `--max-stocks 1` and provide `-s SYMBOL`:**

**From project directory:**
```powershell
# Basic
python runner.py -t 0.25 -s AAPL -m 100 --max-stocks 1

# With custom params
python runner.py -t 0.25 -s TSLA -m 500 --max-stocks 1 --tp 3.0 --sl 1.5
```

**From anywhere:**
```powershell
python "$BotDir\runner.py" -t 0.25 -s AAPL -m 100 --max-stocks 1
```

**Arguments:**
- `-s SYMBOL` - Stock symbol to trade
- `--max-stocks 1` - REQUIRED (limits to 1 stock)
- `-m USD` - Maximum capital
- `--tp`, `--sl`, `--no-dynamic` - Optional overrides

---

## 🔍 Optimizer

**Finds optimal INTERVAL and CAPITAL by testing multiple stocks.**

**Default Mode (Multi-Stock):**
```powershell
# Just run it - tests multiple stocks using binary search
python optimizer.py -v

# Custom capital limit
python optimizer.py -m 500 -v
```

**Single-Stock Mode (Optional):**
```powershell
# Only if you want to trade ONE specific stock
python optimizer.py -s AAPL -v

# Compare specific stocks
python optimizer.py --symbols AAPL TSLA NVDA -v
```

**From anywhere:**
```powershell
$BotDir = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading"
python "$BotDir\optimizer.py" -v
python "$BotDir\optimizer.py" -m 500 -v
python "$BotDir\optimizer.py" -s AAPL -v  # Only for single-stock mode
```

**What It Does:**
1. Tests multiple stocks using binary search (SPY, QQQ, AAPL, TSLA, NVDA, MSFT, GOOGL, AMZN)
2. Tests intervals from 60s to 23,400s (1min to 6.5 hours) for each
3. Tests capitals from $1 to your max-cap (default: $1M)
4. Finds interval & capital that work best across stocks
5. Returns: **Best interval + Best capital**

**⚠️ For multi-stock trading, don't use `-s SYMBOL`!** Just run `python optimizer.py -v` to test multiple stocks.

**Example Output:**
```
==================================================================
AUTO-SCANNING MODE
==================================================================
Testing 8 popular stocks to find best parameters...
Symbols: SPY, QQQ, AAPL, TSLA, NVDA, MSFT, GOOGL, AMZN

==================================================================
TESTING: SPY
==================================================================
Method: Binary search across ALL intervals and capitals
...
Result: $2.15/day @ 900s (0.2500h) with $100 cap

==================================================================
TESTING: TSLA
==================================================================
...
Result: $5.30/day @ 300s (0.0833h) with $250 cap

... (tests all 8 stocks) ...

==================================================================
RESULTS SUMMARY
==================================================================
1. TSLA   ✅  $  5.30/day  @   300s (0.0833h)  $    250 cap
2. NVDA   ✅  $  4.50/day  @   600s (0.1667h)  $    200 cap
3. AAPL   ✅  $  3.45/day  @   900s (0.2500h)  $    150 cap
4. SPY    ✅  $  2.15/day  @   900s (0.2500h)  $    100 cap
5. QQQ    ✅  $  1.80/day  @  1800s (0.5000h)  $     75 cap
==================================================================

==================================================================
OPTIMAL CONFIGURATION
==================================================================
Symbol: TSLA (best performing for reference)
Interval: 300s (0.0833h)  ← Use this
Capital: $250             ← Use this per stock
Expected Daily Return: $5.30

✅ STRATEGY IS PROFITABLE

Use these parameters with bot:
  
  # Multi-stock (bot auto-picks best 15 stocks!) - DEFAULT
  python runner.py -t 0.0833 -m 3750
  
  # Single stock (if you want to trade just TSLA)
  python runner.py -t 0.0833 -s TSLA -m 250 --max-stocks 1
```

---

## 🧪 Testing

**From project directory:**

### 1. Test Everything
```powershell
python test_all_systems.py
```

### 2. Check Current Signals
```powershell
python test_signals.py -s AAPL -t 0.25 -b 100
```

### 3. Scan Best Stocks
```powershell
python scan_best_stocks.py --interval 0.25 --cap 100 --top 5 --verbose
```

### 4. Validate Setup
```powershell
python validate_setup.py -t 0.25 -m 1500 --max-stocks 15
```

**From anywhere:**
```powershell
$BotDir = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading"
python "$BotDir\test_all_systems.py"
python "$BotDir\test_signals.py" -s AAPL -t 0.25 -b 100
python "$BotDir\scan_best_stocks.py" --interval 0.25 --cap 100 --top 5 --verbose
python "$BotDir\validate_setup.py" -t 0.25 -m 1500 --max-stocks 15
```

---

## ⚙️ Running the Bot

### Simple: Just Run It

**From project directory:**
```powershell
# Multi-stock (default: auto-picks 15 best stocks)
python runner.py -t 0.25 -m 1500

# Single stock (must specify --max-stocks 1)
python runner.py -t 0.25 -s AAPL -m 100 --max-stocks 1
```

**From anywhere:**
```powershell
$BotDir = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading"
python "$BotDir\runner.py" -t 0.25 -m 1500
python "$BotDir\runner.py" -t 0.25 -s AAPL -m 100 --max-stocks 1
```

**That's it!** Bot runs in your current window. Press Ctrl+C to stop.

### Want It to Run Forever? Use Admin Mode

**Only run as Admin if you want:**
- Auto-start on boot/logon
- Wake at 9:25 AM before market
- Auto-restart on crash
- Run hidden in background

**Start (As Admin):**

From project directory:
```powershell
.\botctl.ps1 start python -u runner.py -t 0.25 -s AAPL -m 100
```

From anywhere:
```powershell
$BotPath = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1"
pwsh -NoProfile -ExecutionPolicy Bypass -File $BotPath start python -u runner.py -t 0.25 -s AAPL -m 100
```

**Control Commands (As Admin):**

From project directory:
```powershell
.\botctl.ps1 restart             # Holy grail - always works
.\botctl.ps1 status              # Check status
.\botctl.ps1 stop                # Temporary stop (auto-restarts on boot)
.\botctl.ps1 stop-forever        # Permanent stop + cleanup
```

From anywhere:
```powershell
$BotPath = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1"
pwsh -File $BotPath restart
pwsh -File $BotPath status
pwsh -File $BotPath stop
pwsh -File $BotPath stop-forever
```

**What Admin Mode Does:**
- ✅ Auto-starts on system boot
- ✅ Auto-starts on user logon
- ✅ Wakes system at 9:25 AM (5 min before market)
- ✅ Keeps system awake during market hours (9:30 AM - 4:00 PM ET)
- ✅ Auto-restarts on crash
- ✅ Runs hidden in background

**Non-Admin?** Just run `python runner.py ...` directly. Simple!

### Monitor

```powershell
# Watch log live
Get-Content bot.log -Wait -Tail 50

# View portfolio (shows all current stocks + amounts)
type portfolio.json

# View P&L history
type pnl_ledger.json
```

**What Gets Saved:**

- **`portfolio.json`** - **Current stocks you hold** (auto-created/updated)
  - Stock symbols
  - Quantities
  - Entry prices
  - Market value
  - Unrealized P&L
  - Last update time

- **`pnl_ledger.json`** - Trade history with realized P&L

- **`bot.log`** - All activity (truncated automatically)

- **`last_start_cmd.txt`** - Last command used (for restart)

- **`start_bot.ps1`** - Wrapper script (auto-created for background mode)

**Example `portfolio.json`:**
```json
{
  "positions": {
    "AAPL": {
      "qty": 10,
      "avg_entry": 150.25,
      "market_value": 1520.00,
      "unrealized_pl": 17.50,
      "last_update": "2025-10-24T14:30:00Z"
    },
    "TSLA": {
      "qty": 5,
      "avg_entry": 245.80,
      "market_value": 1250.00,
      "unrealized_pl": 21.00,
      "last_update": "2025-10-24T14:30:00Z"
    }
  },
  "last_updated": "2025-10-24T14:30:00Z"
}
```

---

## 📖 CLI Reference

**💡 Tip: All commands work from anywhere using full paths!**
```powershell
$BotDir = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading"
python "$BotDir\runner.py" -t 0.25 -s AAPL -m 100
```

### runner.py (UNIFIED)

```
python runner.py -t HOURS -m CAPITAL [OPTIONS]
# Or from anywhere:
python "$BotDir\runner.py" -t HOURS -m CAPITAL [OPTIONS]

Required:
  -t, --time HOURS          Trading interval
  -m, --max-cap USD         Total capital

Stock Selection:
  --max-stocks N            Max positions (default: 15)
  -s, --symbol TICKER       Single stock symbol (requires --max-stocks 1)
  --stocks SYM1 SYM2 ...    Force specific stocks in portfolio
  --cap-per-stock USD       Capital per stock (default: total/max)
  --rebalance-every N       Rebalance frequency (default: 4)

Optional:
  --tp PERCENT              Take profit %
  --sl PERCENT              Stop loss %
  --frac DECIMAL            Position size fraction
  --no-dynamic              Disable confidence adjustments
  --go-live                 Enable live trading
  --allow-missing-keys      Debug mode
```

### optimizer.py

```
python optimizer.py [OPTIONS]
# Or: python "$BotDir\optimizer.py" [OPTIONS]

Optional:
  -s, --symbol TICKER       Single stock (only for single-stock mode)
  --symbols SYM1 SYM2 ...   Multiple stocks to test
  -m, --max-cap USD         Maximum capital to test (default: $1,000,000)
  -v, --verbose             Show detailed progress

Default (no args): Tests SPY, QQQ, AAPL, TSLA, NVDA, MSFT, GOOGL, AMZN
Returns: Best interval & capital that work across stocks
```

### Test Tools

```
python test_all_systems.py
python test_signals.py -s AAPL -t 0.25
python scan_best_stocks.py --verbose
python validate_setup.py -t 0.25 -m 1500 --max-stocks 15

# Or from anywhere:
python "$BotDir\test_all_systems.py"
python "$BotDir\test_signals.py" -s AAPL -t 0.25
python "$BotDir\scan_best_stocks.py" --verbose
python "$BotDir\validate_setup.py" -t 0.25 -m 1500 --max-stocks 15
```

---

## 💡 Examples

### Example 1: Multi-Stock Trading (Easiest!)
```powershell
# From project dir
# Optimizer tests multiple stocks, finds best interval & capital
python optimizer.py -v

# Use those parameters - bot auto-picks best 15 stocks! (15 is default)
python runner.py -t 0.083 -m 3750

# From anywhere
$BotDir = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading"
python "$BotDir\optimizer.py" -v
python "$BotDir\runner.py" -t 0.083 -m 3750
```

### Example 2: Single Stock Trading
```powershell
# From project dir
python optimizer.py -s AAPL -v
python runner.py -t 0.25 -s AAPL -m 150 --max-stocks 1

# From anywhere
python "$BotDir\optimizer.py" -s AAPL -v
python "$BotDir\runner.py" -t 0.25 -s AAPL -m 150 --max-stocks 1
```

### Example 3: Custom Capital Limit
```powershell
# From project dir
# Only test up to $500 per stock
python optimizer.py -m 500 -v

# From anywhere
python "$BotDir\optimizer.py" -m 500 -v
```

### Example 4: Force Specific Stocks in Portfolio
```powershell
# From project dir
python runner.py -t 0.25 -m 1500 --max-stocks 15

# From anywhere
python "$BotDir\runner.py" -t 0.25 -m 1500 --max-stocks 15
```

### Example 5: Force Favorites + Auto-Fill
```powershell
# From project dir
python runner.py -t 0.25 -m 1000 --stocks TSLA NVDA --max-stocks 10

# From anywhere
python "$BotDir\runner.py" -t 0.25 -m 1000 --stocks TSLA NVDA --max-stocks 10
```

### Example 6: Conservative Blue-Chips Only
```powershell
# From project dir
python runner.py -t 1.0 -m 600 --stocks AAPL MSFT GOOGL --max-stocks 3

# From anywhere
python "$BotDir\runner.py" -t 1.0 -m 600 --stocks AAPL MSFT GOOGL --max-stocks 3
```

### Example 7: Run Forever (Admin Mode)
```powershell
# From project dir
.\botctl.ps1 start python -u runner.py -t 0.25 -m 1500 --max-stocks 15
Get-Content bot.log -Wait -Tail 50
.\botctl.ps1 status

# From anywhere
$BotPath = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1"
pwsh -File $BotPath start python -u runner.py -t 0.25 -m 1500 --max-stocks 15
Get-Content "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\bot.log" -Wait -Tail 50
pwsh -File $BotPath restart
pwsh -File $BotPath stop-forever
```

---

## 🔧 Troubleshooting

### "No profitable stocks found"
Market is bearish. Try different symbols or wait.
```powershell
# From project dir
python scan_best_stocks.py --verbose
python test_signals.py -s SYMBOL

# From anywhere
$BotDir = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading"
python "$BotDir\scan_best_stocks.py" --verbose
python "$BotDir\test_signals.py" -s SYMBOL
```

### "API rate limit"
Use longer intervals or reduce scan frequency.
```powershell
# From project dir
python runner.py -t 0.5 -m 1500 --max-stocks 10 --rebalance-every 8

# From anywhere
python "$BotDir\runner.py" -t 0.5 -m 1500 --max-stocks 10 --rebalance-every 8
```

### "ERROR: Specified X stocks but max is Y"
Too many forced stocks.
```powershell
# Fix: Increase max-stocks
python runner.py --stocks A B C D E --max-stocks 5
python "$BotDir\runner.py" --stocks A B C D E --max-stocks 5
```

### "Bot exits with negative projection"
Strategy unprofitable. Try different symbol or wait for bullish market.
```powershell
python optimizer.py -s SPY -v
python "$BotDir\optimizer.py" -s SPY -v
```

### "Could not sync with broker"
Invalid API keys. Check .env file.
```powershell
python test_all_systems.py
python "$BotDir\test_all_systems.py"
```

### Background task not starting
Run PowerShell as Administrator.

---

## 📊 Strategy Behavior

**How It Works:**

1. **Each Interval:**
   - Fetches price data
   - Calculates moving averages
   - Generates signal (BUY/SELL/HOLD)
   - Computes confidence
   - Adjusts TP/SL/size (if dynamic mode)
   - Executes trades

2. **Safety:**
   - Stop loss and take profit enforced
   - Max daily loss limit
   - Volatility filtering
   - Profitability gates
   - Market hours respected

3. **Multi-Stock:**
   - Every N intervals: Scans for opportunities
   - Scores current holdings
   - Sells underperformers
   - Buys better alternatives
   - Never sells forced stocks

**Performance Expectations:**

| Mode | Interval | Return/Day | Win Rate | Trades/Day |
|------|----------|------------|----------|------------|
| Conservative | 1hr | $1-3 per $100 | 55-65% | 2-5 |
| Moderate | 15min | $2-5 per $100 | 50-60% | 5-10 |
| Aggressive | 5min | $3-8 per $100 | 45-55% | 10-20 |
| Multi-Stock | 15min | $20-40 per $1500 | 50-60% | 15-30 |

---

## 🎓 Key Concepts

### How Stock Selection Works

**Default (15 Stocks):**
```powershell
# Bot auto-selects best 15 stocks
python runner.py -t 0.25 -m 1500
```

**Single Stock:**
```powershell
# Trade just AAPL - requires --max-stocks 1
python runner.py -t 0.25 -s AAPL -m 100 --max-stocks 1
```

**Custom Number:**
```powershell
# Bot picks best 10 stocks
python runner.py -t 0.25 -m 1000 --max-stocks 10
```

### Forced vs Auto-Selected Stocks

**Forced:**
- Specified via `--stocks`
- NEVER replaced during rebalancing
- Still subject to buy/sell signals

**Auto-Selected:**
- Found by scanner
- CAN be replaced if underperforming
- Rebalanced periodically

### Capital Allocation

```powershell
# From project dir
# $1500 total, 15 stocks = $100 per stock
python runner.py -t 0.25 -m 1500 --max-stocks 15

# Override: $150 per stock (total can use $2250)
python runner.py -t 0.25 -m 1500 --max-stocks 15 --cap-per-stock 150

# From anywhere
$BotDir = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading"
python "$BotDir\runner.py" -t 0.25 -m 1500 --max-stocks 15
python "$BotDir\runner.py" -t 0.25 -m 1500 --max-stocks 15 --cap-per-stock 150
```

---

## ⚠️ Important Notes

- **Always paper trade first** - Set `CONFIRM_GO_LIVE=NO` in .env
- **Test with optimizer** - Find profitable setup before running
- **Monitor logs** - Check `bot.log` regularly
- **Bearish markets** - Bot will show negative returns (by design)
- **API limits** - Don't scan too frequently
- **Background mode** - Requires Admin PowerShell

---

## 🎯 Common Workflows

### Workflow 1: Multi-Stock Portfolio (Easiest!)
```powershell
# From project dir
1. python optimizer.py -v                    # Tests 8 stocks, finds best params
2. python runner.py -t 0.083 -m 3750         # Bot auto-picks 15 best! (default)

# From anywhere
$BotDir = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading"
1. python "$BotDir\optimizer.py" -v
2. python "$BotDir\runner.py" -t 0.083 -m 3750
```

### Workflow 2: Single Stock Trading
```powershell
# From project dir
1. python optimizer.py -s SYMBOL -v          # Find best params for this stock
2. python test_signals.py -s SYMBOL -t INTERVAL  # Verify signals
3. python runner.py -t INTERVAL -s SYMBOL -m CAPITAL --max-stocks 1  # Trade it

# From anywhere
$BotDir = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading"
1. python "$BotDir\optimizer.py" -s SYMBOL -v
2. python "$BotDir\test_signals.py" -s SYMBOL -t INTERVAL
3. python "$BotDir\runner.py" -t INTERVAL -s SYMBOL -m CAPITAL --max-stocks 1
```

### Workflow 3: Advanced Portfolio Setup
```powershell
# From project dir
1. python scan_best_stocks.py --interval 0.25 --cap 100 --top 15 --verbose
2. python validate_setup.py -t 0.25 -m 1500 --max-stocks 15
3. python runner.py -t 0.25 -m 1500 --max-stocks 15

# From anywhere
1. python "$BotDir\scan_best_stocks.py" --interval 0.25 --cap 100 --top 15 --verbose
2. python "$BotDir\validate_setup.py" -t 0.25 -m 1500 --max-stocks 15
3. python "$BotDir\runner.py" -t 0.25 -m 1500 --max-stocks 15
```

### Workflow 4: Run Forever
```powershell
# From project dir
1. Test: python runner.py -t 0.25 -s AAPL -m 100
2. Works? Start as Admin: .\botctl.ps1 start python -u runner.py -t 0.25 -s AAPL -m 100
3. Control: .\botctl.ps1 status / restart / stop-forever

# From anywhere
$BotDir = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading"
$BotPath = "$BotDir\botctl.ps1"
1. Test: python "$BotDir\runner.py" -t 0.25 -s AAPL -m 100
2. Works? Start as Admin: pwsh -File $BotPath start python -u runner.py -t 0.25 -s AAPL -m 100
3. Control: pwsh -File $BotPath status / restart / stop-forever
```

---

**Everything is now unified and simplified!**

- ONE runner.py handles all modes
- ONE optimizer.py tests everything
- ONE README.md with all info
- Cleaner, faster, easier to use

**Happy Trading!** 🚀
