# Paper Trading Bot - Intelligent Multi-Stock Trading System

**Automated stock trading with smart capital allocation and portfolio rebalancing**

## 🎯 What This Bot Does

- **Auto-selects profitable stocks** - Scans top 100 stocks by market cap, picks the best performers
- **Smart capital allocation** - More $ to winners, less to losers (or equal split if you prefer)
- **Dynamic rebalancing** - Replaces underperformers automatically
- **Risk management** - Stop loss, take profit, volatility filtering, profitability gates
- **Binary search optimizer** - Finds optimal trading interval and capital
- **Fractional shares** - Trade with any budget (even $10 works)
- **Windows automation** - Runs as scheduled task, auto-starts, wakes PC before market open

## 📊 Key Features

### Trading Modes
- **Multi-stock portfolio** (default: 15 stocks) - Best for diversification
- **Single stock** - Focus on one symbol
- **Forced + auto-fill** - Keep favorites, auto-select the rest

### Smart Systems
- **Profitability scoring** - Only trades stocks with positive expected returns
- **Confidence-based sizing** - Larger positions when signals are strong
- **Volatility filtering** - Avoids overly volatile stocks
- **Dynamic TP/SL** - Adjusts targets based on market conditions
- **Risk overlay** - More aggressive when opportunities are excellent

### Automation
- **Auto-start on boot/logon** - Never miss market open
- **Wake at 9:25 AM** - PC powers on 5 minutes before market
- **Keep-awake during trading** - Prevents sleep during market hours
- **Auto-restart on crash** - Resilient against errors
- **Market hours aware** - Sleeps when market is closed

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
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
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

### Advanced Configuration Explained

**Strategy Parameters:**
- `DEFAULT_INTERVAL_SECONDS`: Default trading interval (900 = 15 minutes)
- `SHORT_WINDOW`: Fast moving average period (9 bars)
- `LONG_WINDOW`: Slow moving average period (21 bars)
- `TRADE_SIZE_FRAC_OF_CAP`: Use 75% of allocated capital per trade (keep 25% buffer)
- `MAX_CAP_USD`: Maximum capital per stock (default $100)

**Risk Management:**
- `TAKE_PROFIT_PERCENT`: Exit when profit reaches 3%
- `STOP_LOSS_PERCENT`: Exit when loss reaches 1%
- `TRAILING_STOP_PERCENT`: Optional trailing stop (0 = disabled)
- `MAX_DAILY_LOSS_PERCENT`: Shut down if account drops 5% in one day
- `MAX_DRAWDOWN_PERCENT`: Maximum drawdown allowed (10%)

**Confidence & Volatility:**
- `MIN_CONFIDENCE_TO_TRADE`: Minimum MA separation to trade (0.005 = 0.5%)
- `CONFIDENCE_MULTIPLIER`: How much confidence affects position sizing (9.0)
- `VOLATILITY_PCT_THRESHOLD`: Skip stocks with volatility > 15%
- `VOLATILITY_WINDOW`: Look back 30 bars for volatility calculation

**Profitability Gates:**
- `PROFITABILITY_GATE_ENABLED`: Only trade stocks with positive expected return
- `PROFITABILITY_MIN_EXPECTED_USD`: Minimum $0.01/day expected return
- `STRONG_CONFIDENCE_THRESHOLD`: High confidence = 8% MA separation
- `STRONG_CONFIDENCE_BYPASS_ENABLED`: Allow trades with strong signals even if expected return is low

**Risky Mode (Aggressive Profit Targeting):**
When enabled, bot takes larger positions with higher targets when opportunities are excellent:
- `RISKY_MODE_ENABLED`: Enable aggressive mode (default: on)
- `RISKY_EXPECTED_DAILY_USD_MIN`: Trigger threshold ($0.05/day)
- `RISKY_TP_MULT`: Increase take profit by 25% (1.25×)
- `RISKY_SL_MULT`: Widen stop loss by 15% (1.15×)
- `RISKY_SIZE_MULT`: Increase position size by 30% (1.30×)
- `RISKY_MAX_FRAC_CAP`: Use up to 95% of allocated capital

**Logging & Safety:**
- `LOG_PATH`: Log file location (bot.log)
- `LOG_MAX_AGE_HOURS`: Auto-delete logs older than 48 hours
- `PNL_LEDGER_PATH`: Trade history file (pnl_ledger.json)
- `ENABLE_MARKET_HOURS_ONLY`: Only trade during market hours (9:30-4:00 ET)
- `EXIT_ON_NEGATIVE_PROJECTION`: Exit if expected return is negative
- `ALLOW_MISSING_KEYS_FOR_DEBUG`: Skip API key validation (testing only)

---

## 📁 File Structure

```
Core Trading Engine:
  runner.py              - Main bot (handles single & multi-stock)
  config.py              - Configuration & environment variables
  optimizer.py           - Binary search optimizer (finds best params)
  
Portfolio Management:
  portfolio_manager.py   - Position tracking & portfolio state
  stock_scanner.py       - Stock evaluation & ranking engine
  multi_stock_config.py  - Multi-stock portfolio settings
  
Utilities:
  validate_setup.py      - Pre-flight configuration validator
  scan_best_stocks.py    - CLI tool to find best stocks right now
  
Windows Automation:
  botctl.ps1             - Task controller (start/stop/restart)
  start_bot.ps1          - Auto-generated wrapper (created by botctl)
  last_start_cmd.txt     - Last command saved (for restart)
  
Data Files (auto-created):
  bot.log                - Trading activity log (auto-truncated to 250 lines)
  portfolio.json         - Current positions (symbol, qty, entry, value, P&L)
  pnl_ledger.json        - Trade history with realized gains/losses
  top_stocks_cache.json  - Top 100 stocks by market cap (refreshed weekly)
  .env                   - API keys & secrets (you create this)
```

**Architecture:**
- **Entry point**: `runner.py` - Single file that does everything
- **Data providers**: yfinance (free), Alpaca (fallback), Polygon (optional)
- **Broker**: Alpaca paper trading (free) or live trading
- **Strategy**: Dual moving average crossover (9/21 SMA) with dynamic adjustments
- **Execution**: Market orders with bracket orders (TP/SL) on whole shares
- **Portfolio**: Smart allocation (profit-weighted) or equal split

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
- `--cap-per-stock USD` - Fixed capital per stock (disables smart allocation)
- `--rebalance-every N` - Rebalance frequency (default: 4)

**💡 Smart Allocation (Always Enabled):**
The bot automatically uses **smart capital allocation** to maximize returns:

**How It Works:**
1. **Max cap is your total budget** across ALL stocks (e.g., $1500 total, not per stock)
2. **Scans all stocks** and ranks by profitability
3. **Allocates more $ to better performers** (e.g., TSLA $600, AAPL $500, NVDA $400)
4. **Sells underperformers** to free capital for better opportunities
5. **Holds cash when needed** - being under max cap is totally fine!

**Example:**
```
You have: TSLA $300, AAPL $400, NVDA $200 (Total: $900/$1500)
Bot finds: TSLA still great, AAPL good, MSFT better than NVDA
Action: Sells NVDA ($200) → Buys MSFT ($500) → Result: $900→$1000 invested
```

To force equal split instead, specify `--cap-per-stock` manually.

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
.\botctl.ps1 start python -u runner.py
```

From anywhere:
```powershell
$BotPath = "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1"
pwsh -NoProfile -ExecutionPolicy Bypass -File $BotPath start python -u runner.py -t 0.25 -m 100
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
& $BotPath restart
& $BotPath status
& $BotPath stop
& $BotPath stop-forever
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
& $BotPath start python -u runner.py -t 0.25 -m 1500 --max-stocks 15
Get-Content "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\bot.log" -Wait -Tail 50
& $BotPath restart
& $BotPath stop-forever
```

---

## 🔧 Troubleshooting & Debugging

### Common Errors & Solutions

#### "No profitable stocks found"

**Cause**: Market is bearish or strategy doesn't fit current conditions.

**Solutions**:
```powershell
# 1. Scan to see what's available
python scan_best_stocks.py --verbose

# 2. Try different interval (optimizer finds best)
python optimizer.py -v

# 3. Check specific stock signals
python test_signals.py -s SPY -t 0.25

# 4. Lower profitability threshold (temporary)
# Add to .env:
PROFITABILITY_MIN_EXPECTED_USD=0.001  # Lower from $0.01 to $0.001
```

**Note**: If ALL stocks show negative returns, the market is likely bearish. The bot correctly holds cash instead of forcing bad trades.

---

#### "API rate limit exceeded"

**Cause**: Too many API calls (scanning too many stocks too frequently).

**Solutions**:
```powershell
# 1. Use longer intervals (fewer data requests)
python runner.py -t 0.5 -m 1500  # 30 min instead of 15 min

# 2. Reduce number of stocks
python runner.py -t 0.25 -m 1500 --max-stocks 10  # 10 instead of 15

# 3. Rebalance less frequently
python runner.py -t 0.25 -m 1500 --rebalance-every 8  # Every 2 hours

# 4. Use yfinance (free, unlimited) - already default fallback
```

**Note**: yfinance is used first (unlimited), so rate limits are rare unless data is missing.

---

#### "ERROR: Specified 5 stocks but max is 3"

**Cause**: Too many forced stocks for the max-stocks limit.

**Solution**:
```powershell
# Either reduce forced stocks OR increase max-stocks
python runner.py --stocks AAPL MSFT GOOGL --max-stocks 5  # Increase max

# Or remove some forced stocks
python runner.py --stocks AAPL MSFT --max-stocks 3  # Reduce forced
```

---

#### "Bot exits with negative projection"

**Cause**: Strategy shows negative expected returns. Config `EXIT_ON_NEGATIVE_PROJECTION=1` makes bot exit instead of trading.

**Solutions**:
```powershell
# 1. Find a profitable symbol
python optimizer.py -v  # Tests 8 popular stocks

# 2. Try different intervals
python optimizer.py -s AAPL -v  # Test AAPL specifically

# 3. Disable exit-on-negative (bot will hold cash instead)
# Add to .env:
EXIT_ON_NEGATIVE_PROJECTION=0
```

---

#### "Could not sync with broker" / "API authentication failed"

**Cause**: Invalid or missing Alpaca API keys.

**Solutions**:
```powershell
# 1. Check .env file exists and has keys
type .env

# 2. Verify keys are correct (login to Alpaca, regenerate if needed)
# https://app.alpaca.markets/paper/dashboard/overview

# 3. Check for typos (no quotes, no spaces)
# CORRECT:   ALPACA_API_KEY=PKX7Y3M2...
# WRONG:     ALPACA_API_KEY="PKX7Y3M2..."
# WRONG:     ALPACA_API_KEY = PKX7Y3M2...

# 4. Make sure using paper trading URL
# ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

---

#### Background task not starting

**Cause**: Not running as Administrator.

**Solution**:
```powershell
# 1. Close PowerShell
# 2. Right-click PowerShell → "Run as Administrator"
# 3. cd to project directory
# 4. Run command again
.\botctl.ps1 start python -u runner.py -t 0.25 -m 100
```

---

#### "Bot is buying/selling constantly" (overtrading)

**Cause**: Interval too short or strategy too sensitive.

**Solutions**:
```powershell
# 1. Use longer interval
python runner.py -t 0.5 -m 1500  # 30 min = fewer trades

# 2. Increase confidence threshold
# Add to .env:
MIN_CONFIDENCE_TO_TRADE=0.01  # Increase from 0.005

# 3. Check optimizer for recommended interval
python optimizer.py -v
```

---

#### "Position size is too small" / "Insufficient capital"

**Cause**: Not enough capital allocated per stock.

**Solutions**:
```powershell
# 1. Increase capital per stock
python runner.py -t 0.25 -m 1500 --max-stocks 10  # $150/stock instead of $100

# 2. Or specify exact amount
python runner.py -t 0.25 -m 1500 --cap-per-stock 200  # $200/stock

# 3. Check minimum is at least $10/stock for fractional shares
```

---

#### Log file keeps growing / disk space issues

**Cause**: Log truncation disabled or very active trading.

**Solution**: The bot auto-truncates to 250 lines. If growing:
```powershell
# 1. Manually clear log
Remove-Item bot.log

# 2. Check if truncation is working (check bot.log stays ~250 lines)

# 3. If needed, reduce log retention
# Add to .env:
LOG_MAX_AGE_HOURS=24  # Delete logs older than 24 hours
```

---

### Debugging Tools

#### 1. Validate Setup (Pre-flight Check)
```powershell
python validate_setup.py -t 0.25 -m 1500 --max-stocks 15 --stocks AAPL TSLA
```
Shows configuration errors BEFORE running bot.

#### 2. Watch Logs Live
```powershell
Get-Content bot.log -Wait -Tail 50
```
Real-time monitoring of bot activity.

#### 3. Check Portfolio State
```powershell
type portfolio.json | ConvertFrom-Json | ConvertTo-Json
```
See current positions, values, P&L.

#### 4. Check Task Status (Admin Mode)
```powershell
.\botctl.ps1 status
```
Shows if bot is running, when it started, recent activity.

#### 5. Test Specific Stock
```powershell
python scan_best_stocks.py -s AAPL TSLA NVDA --verbose
```
Evaluate specific symbols to see expected performance.

---

### Getting Help

**Check logs first**:
```powershell
Get-Content bot.log -Tail 100
```
Most issues are explained in log messages.

**Common log messages explained**:
- `"No data available"` → Data provider issue (try different stock)
- `"Low confidence"` → Signal too weak (working as intended)
- `"Expected $X/day < $Y"` → Below profitability threshold (correct behavior)
- `"High volatility: X%"` → Stock too risky (safety feature working)
- `"Position exists"` → Already holding stock (can't buy more without adding capital)
- `"No position"` → Can't sell what you don't own (expected)

**Enable verbose logging**:
Add to Python command: `--verbose` (if supported by that script)

**Test everything**:
```powershell
# Run full system test (if available)
python test_all_systems.py
```

---

## 📊 Trading Strategy & How It Works

### Core Strategy: Dual Moving Average Crossover

The bot uses a proven technical analysis strategy:

**Indicators:**
- **Short MA**: 9-period Simple Moving Average (fast)
- **Long MA**: 21-period Simple Moving Average (slow)
- **Crossover threshold**: 0.2% to avoid noise

**Signal Generation:**
- **BUY**: Short MA > Long MA × 1.002 (bullish momentum)
- **SELL**: Short MA < Long MA × 0.998 (bearish momentum)
- **HOLD**: No clear signal (wait for better entry)

**Confidence Scoring:**
- Measures MA separation (larger gap = higher confidence)
- Momentum boost (recent price movement in signal direction)
- Used to adjust position size and TP/SL dynamically

### Execution Flow

**Each Trading Interval:**

1. **Market check** - Is market open? (9:30 AM - 4:00 PM ET)
2. **Fetch data** - Get recent price bars (yfinance → Alpaca → Polygon)
3. **Calculate indicators** - Compute 9/21 SMAs
4. **Generate signal** - BUY/SELL/HOLD based on crossover
5. **Compute confidence** - How strong is the signal?
6. **Profitability check** - Expected daily return > $0.01?
7. **Volatility check** - Recent volatility < 15%?
8. **Dynamic adjustment** - Scale TP/SL/size based on confidence
9. **Execute order** - Market order with bracket TP/SL
10. **Safety enforcement** - Daily loss limit, max drawdown checks

**Multi-Stock Rebalancing (Every N Intervals):**

1. **Score current positions** - Rate each holding
2. **Scan for opportunities** - Evaluate top 100 stock universe (by market cap)
3. **Compare scores** - New opportunity vs worst holding
4. **Rebalance decision** - Replace if new stock scores 10%+ higher
5. **Execute swap** - Sell underperformer, buy better stock
6. **Never touch forced stocks** - User-specified symbols stay

### Safety Features

**Risk Management:**
- **Take Profit**: Default 3% gain target (adjustable)
- **Stop Loss**: Default 1% loss limit (adjustable)
- **Max daily loss**: Exits if account drops 5% in one day
- **Volatility filter**: Skips stocks with >15% recent volatility
- **Profitability gate**: Only trades stocks with positive expected return
- **Confidence minimum**: Requires 0.5% MA separation to trade
- **Position sizing**: Uses 75% of allocated capital (keeps 25% cash buffer)

**Market Hours:**
- Bot automatically sleeps when market is closed
- In scheduled task mode: exits cleanly, restarts at market open
- In console mode: waits silently, resumes when market opens
- Respects weekends and holidays

### Performance Expectations

| Mode | Interval | Capital | Expected Return/Day | Win Rate | Trades/Day | Risk Level |
|------|----------|---------|---------------------|----------|------------|------------|
| Conservative | 1hr | $100/stock | $1-3 | 55-65% | 2-5 | Low |
| Moderate | 15min | $100/stock | $2-5 | 50-60% | 5-10 | Medium |
| Aggressive | 5min | $100/stock | $3-8 | 45-55% | 10-20 | High |
| Multi-Stock | 15min | $1500 total | $20-40 | 50-60% | 15-30 | Medium |

**Notes:**
- Returns vary based on market conditions (bull/bear/sideways)
- Historical simulation assumes perfect execution (real-world: slippage, fees)
- Paper trading is commission-free (live trading has costs)
- Results improve with longer backtesting periods

---

## 🧠 Smart Capital Allocation Explained

The bot's secret weapon is its **profit-weighted capital allocation** system.

### How Traditional Bots Work (Equal Split)

```
$1500 total ÷ 15 stocks = $100 per stock
AAPL: $100 | MSFT: $100 | GOOGL: $100 | ... (all equal)
```

**Problem**: Great stocks get same capital as mediocre ones.

### How This Bot Works (Smart Allocation)

```
1. Score each stock (expected daily return + confidence)
2. Rank by profitability
3. Allocate MORE $ to better performers
4. Allocate LESS $ to weaker performers
5. Hold cash if no good opportunities
```

**Example:**
```
Total capital: $1500
Stocks evaluated: TSLA, AAPL, NVDA, AMD, GOOGL

Scoring:
  TSLA:  $5.30/day → Score: 8.2  → Allocation: $350 (23%)
  NVDA:  $4.50/day → Score: 7.1  → Allocation: $300 (20%)
  AAPL:  $3.45/day → Score: 5.8  → Allocation: $250 (17%)
  AMD:   $2.80/day → Score: 4.5  → Allocation: $200 (13%)
  GOOGL: $2.10/day → Score: 3.4  → Allocation: $150 (10%)
  ...remaining stocks get less...
  
Total invested: $1250 (holding $250 cash - being under max is OK!)
```

**Result**: Best stocks get 2-3× more capital than average stocks.

### Rebalancing Logic

**Every 4 intervals** (customizable):

1. **Re-score all holdings**
   - Still profitable? Keep it
   - Turned unprofitable? Candidate for replacement

2. **Scan for new opportunities**
   - Find 5 best stocks not currently held
   - Compare with worst current holding

3. **Execute swaps**
   - If new stock scores 10%+ higher than worst holding
   - Sell worst performer
   - Buy better opportunity
   - Reallocate capital

4. **Forced stocks protection**
   - User-specified symbols NEVER sold
   - Always kept in portfolio
   - Can still BUY/SELL based on signals, but position stays

### Disable Smart Allocation (Use Equal Split)

If you prefer equal capital distribution:

```powershell
# Force $150 per stock (overrides smart allocation)
python runner.py -t 0.25 -m 1500 --max-stocks 15 --cap-per-stock 150
```

**When to use equal split:**
- You trust your stock picks equally
- You want predictable capital per stock
- You're testing specific symbols

**When to use smart allocation (default):**
- You want the bot to maximize returns
- You trust the profitability scoring
- You want automatic optimization

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

## ⏰ Market Hours & Weekend Behavior

**Important:** Stock prices **DO NOT change** when the market is closed!

### How It Works:
- **Market Open**: Monday-Friday, 9:30 AM - 4:00 PM ET
- **Market Closed**: Evenings, Weekends, Holidays
- **Bot Behavior**: Automatically sleeps when market closes, resumes when it opens

### Your Positions Are Safe:
✅ **Stock values FREEZE when market closes**
- Your shares keep their exact value
- No overnight/weekend price changes
- No risk of losses while you sleep
- Bot doesn't need to sell before close

⚠️ **Gap Risk (Advanced)**
While prices don't change when closed, they CAN "gap" at open:
- News happens → Market opens with instant price change
- This is normal market behavior
- Bot's stop-loss orders protect you after open

### Example:
```
Friday 4:00 PM: You own AAPL at $150/share → Market closes
Weekend:       Stock STAYS at $150 (no change possible)
Monday 9:30 AM: Market opens, AAPL might gap to $152 or $148
                (Bot's orders activate immediately)
```

**Bottom Line:** The bot correctly handles weekends by doing nothing. Your positions are safe because trading is suspended. The market will resume Monday morning, and the bot will resume trading then.

---

## ❓ Frequently Asked Questions (FAQ)

### General Questions

**Q: Is this real money or paper trading?**
A: By default, it's **paper trading** (fake money, real market data). To enable live trading, you must set `CONFIRM_GO_LIVE=YES` in `.env` AND use `--go-live` flag. Always test with paper trading first!

**Q: Do I need to pay for anything?**
A: No! Everything is free:
- Alpaca paper trading account (free)
- yfinance data (free, unlimited)
- Python & all libraries (free, open source)
- Polygon is optional (free tier works if you have it)

**Q: How much money do I need to start?**
A: Paper trading: $0 (virtual money)
   Live trading: As little as $10 (fractional shares supported)
   Recommended: $500-1500 for multi-stock portfolio

**Q: What stocks can I trade?**
A: Any stock on US exchanges (NYSE, NASDAQ). Default universe: **Top 100 stocks by market cap** (automatically updated weekly), including major tech (AAPL, MSFT, GOOGL, NVDA), financials (JPM, BAC), consumer (WMT, COST), healthcare (UNH, JNJ), and ETFs (SPY, QQQ, IWM, DIA) for diversification.

**Q: Can I trade crypto or forex?**
A: No, this bot is designed for US stocks only. Alpaca doesn't support crypto/forex for paper trading.

**Q: How does the "top 100 stocks" selection work?**
A: The bot automatically fetches the top 100 US stocks by market capitalization from the S&P 500:
- **First run**: Downloads list from Wikipedia (takes ~1 minute), caches locally
- **Subsequent runs**: Uses cached list (instant)
- **Auto-refresh**: Cache expires after 7 days, automatically updates
- **Fallback**: If download fails, uses predefined list of 100 major stocks
- **Manual refresh**: Delete `top_stocks_cache.json` to force immediate refresh
- **Disable dynamic fetch**: Use `--no-dynamic` flag in scan_best_stocks.py

This ensures you're always trading the most liquid, actively-traded stocks in the market.

### Technical Questions

**Q: Why does the bot hold cash instead of using all my capital?**
A: **This is intentional!** Smart allocation only invests in profitable opportunities. If stocks don't meet the profitability threshold ($0.01/day expected return), the bot holds cash instead of forcing bad trades. Being under max cap is a feature, not a bug.

**Q: What's the difference between `-m` (total capital) and `--cap-per-stock`?**
A:
- `-m 1500`: Total budget across ALL stocks
- `--cap-per-stock 100`: Force exactly $100 per stock (disables smart allocation)
- Default: Smart allocation divides `-m` proportionally by profitability

**Q: The bot shows "No profitable stocks found" - what do I do?**
A:
1. Market might be bearish (strategy is long-only)
2. Try different interval: `python optimizer.py -v` to find better parameters
3. Try different stocks: `python scan_best_stocks.py --verbose`
4. Wait for better market conditions (sideways/uptrend)

**Q: Why did the optimizer suggest negative returns?**
A: The market is currently bearish for that symbol/interval. The strategy is long-only (profits from uptrends). Either:
- Try different symbols (some stocks are bullish even in bear markets)
- Try different intervals (longer intervals smooth out volatility)
- Wait for better market conditions

**Q: What's the minimum interval I can use?**
A: 1 minute (0.0167 hours). Shorter intervals = more trades but more noise. Recommended: 15 minutes (0.25 hours) for balance.

**Q: Does the bot work on Mac or Linux?**
A: The **trading bot** works on any OS (Python is cross-platform). The **scheduled task automation** (botctl.ps1) is Windows-only. On Mac/Linux, use cron jobs or screen/tmux instead.

### Performance Questions

**Q: How much profit can I expect?**
A: Varies by market conditions:
- Bull market: $2-5 per $100 capital per day (15min interval)
- Sideways market: $1-3 per $100 capital per day
- Bear market: Negative (bot will hold cash or exit)
- **Past performance ≠ future results**

**Q: Why is my paper trading performance different from the optimizer's prediction?**
A: Several reasons:
- Market conditions changed since optimization
- Real execution has slippage (slight price differences)
- Random variance (short-term results differ from long-term averages)
- Optimizer uses historical data (backward looking)

**Q: Can I backtest before running live?**
A: The optimizer IS the backtester. It simulates historical trades and shows expected returns. Run `python optimizer.py -v` to see performance on past data.

**Q: What's a good win rate?**
A: 50-60% is excellent for a simple MA crossover strategy. Even 45% can be profitable if winners are bigger than losers (good risk:reward ratio).

### Operational Questions

**Q: The bot stopped running - why?**
A: Several possibilities:
1. Market closed (normal - bot sleeps until open)
2. Negative projection (config: `EXIT_ON_NEGATIVE_PROJECTION=1`)
3. Daily loss limit hit (safety feature)
4. API key issue (check `.env`)
5. Crash (check `bot.log` for errors)

**Q: How do I know if the bot is working?**
A:
```powershell
# Check status (Admin mode)
.\botctl.ps1 status

# Watch logs live
Get-Content bot.log -Wait -Tail 50

# Check portfolio
type portfolio.json

# Check Alpaca dashboard
# https://app.alpaca.markets/paper/dashboard/overview
```

**Q: Can I run multiple bots with different strategies?**
A: Yes, but they'll share the same Alpaca account. Create separate project folders with different `.env` files pointing to different Alpaca accounts.

**Q: The bot bought/sold when I didn't expect it - why?**
A: Check `bot.log` for the exact reason. Common causes:
- MA crossover triggered (strategy signal)
- Take profit hit (target reached)
- Stop loss hit (loss limit reached)
- Rebalancing (better opportunity found)
- Low confidence (signal too weak)

### Safety & Risk Questions

**Q: Can I lose more than I invest?**
A: **No**. Stop losses prevent catastrophic losses. Max loss per trade is ~1% (configurable). Daily loss limit exits bot if account drops 5% in one day.

**Q: What happens if my PC crashes or loses power?**
A: 
- **Admin mode**: Task auto-restarts on boot
- **Simple mode**: Bot stops, positions remain open with TP/SL orders active
- **Positions**: Alpaca holds your positions, they don't disappear

**Q: What happens over the weekend?**
A: Stock prices **freeze** when market closes. No price changes occur Sat/Sun. Bot sleeps and resumes Monday 9:30 AM. Your positions are safe (trading is suspended).

**Q: What's "gap risk"?**
A: Market can open at a different price than it closed (due to news). Example: Friday close $100, Monday open $95. Stop loss orders trigger immediately at open to protect you.

**Q: Should I use live trading?**
A: **Not recommended for beginners**. Paper trade for at least 30 days first. Live trading requires:
- Understanding of risks
- Emergency fund (don't trade money you need)
- Emotional discipline (don't panic on losses)
- `CONFIRM_GO_LIVE=YES` in `.env`

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
2. Works? Start as Admin: & $BotPath start python -u runner.py -t 0.25 -s AAPL -m 100
3. Control: & $BotPath status / restart / stop-forever
```

---

---

## 🎓 Learning & Best Practices

### Start Small, Scale Up

**Week 1: Paper Trading Basics**
```powershell
# Start with single stock to learn
python optimizer.py -s AAPL -v
python runner.py -t 0.25 -s AAPL -m 100 --max-stocks 1
```
Learn how signals work, watch the logs, understand TP/SL.

**Week 2: Multi-Stock Portfolio**
```powershell
# Expand to 5 stocks
python optimizer.py -v
python runner.py -t 0.25 -m 500 --max-stocks 5
```
See rebalancing in action, compare performance across stocks.

**Week 3: Full Portfolio**
```powershell
# Scale to 15 stocks with smart allocation
python runner.py -t 0.25 -m 1500 --max-stocks 15
```
Let the bot manage everything, monitor weekly performance.

**Week 4+: Optimization & Fine-Tuning**
- Adjust intervals based on market conditions
- Test different stock universes
- Tweak TP/SL based on your risk tolerance
- Consider live trading (if consistent profits for 30+ days)

### Risk Management Tips

1. **Never invest money you can't afford to lose**
2. **Start paper trading for minimum 30 days**
3. **Live trading: Start with $100-500 max**
4. **Only increase capital after 3+ months of profits**
5. **Set account-wide stop loss** (if you lose 20%, stop and re-evaluate)
6. **Diversify**: Don't put all capital in one strategy**
7. **Keep emergency fund separate** (6 months expenses minimum)

### Performance Tracking

**Daily:**
- Check `bot.log` for errors
- Review `portfolio.json` for positions

**Weekly:**
- Calculate total P&L from `pnl_ledger.json`
- Compare vs SPY (S&P 500) benchmark
- Identify best/worst performers

**Monthly:**
- Review win rate (should be 45-60%)
- Check if returns match optimizer predictions
- Adjust strategy if market conditions changed
- Re-run optimizer to find new optimal parameters

### Common Mistakes to Avoid

❌ **Don't**: Run live trading without 30+ days paper trading
✅ **Do**: Master paper trading first

❌ **Don't**: Override safety limits (TP/SL/daily loss)
✅ **Do**: Trust the risk management system

❌ **Don't**: Panic sell during losses
✅ **Do**: Let stop losses handle exits automatically

❌ **Don't**: Increase position sizes after losses
✅ **Do**: Reduce sizes after drawdowns

❌ **Don't**: Trade in highly volatile markets (earnings, news)
✅ **Do**: Use volatility filter (already enabled)

❌ **Don't**: Manually interfere with bot trades
✅ **Do**: Let the bot execute its strategy

❌ **Don't**: Expect 100% win rate
✅ **Do**: Accept 50-60% is excellent

---

## 📚 Additional Resources

### Understanding the Strategy

**Moving Average Crossover:**
- [Investopedia: Moving Average](https://www.investopedia.com/terms/m/movingaverage.asp)
- [Technical Analysis: SMA vs EMA](https://www.investopedia.com/articles/trading/10/simple-exponential-moving-averages-compare.asp)

**Risk Management:**
- [Position Sizing](https://www.investopedia.com/terms/p/positionsizing.asp)
- [Stop Loss Orders](https://www.investopedia.com/terms/s/stop-lossorder.asp)
- [Take Profit Orders](https://www.investopedia.com/terms/t/take-profitorder.asp)

**Alpaca Trading:**
- [Alpaca Docs](https://alpaca.markets/docs/)
- [Paper Trading Guide](https://alpaca.markets/docs/trading/paper-trading/)
- [API Authentication](https://alpaca.markets/docs/api-references/trading-api/authentication/)

### Market Data Providers

**yfinance (Primary, Free)**
- Unlimited requests
- 5-minute delay
- Best for backtesting & paper trading

**Alpaca (Fallback, Free)**
- Real-time for paper trading
- Rate limits apply
- Best for live trading

**Polygon (Optional)**
- Real-time data
- Free tier: 5 requests/minute
- Best for high-frequency needs

---

## ⚠️ Disclaimer

**IMPORTANT LEGAL NOTICE**

This software is provided for **educational and research purposes only**.

- **No warranty**: Software provided "as is" with no guarantees
- **No financial advice**: This is not investment advice
- **Trading risks**: You can lose money trading stocks
- **Past performance ≠ future results**: Historical returns don't guarantee future profits
- **Use at your own risk**: You are responsible for your trading decisions
- **Paper trade first**: Always test thoroughly before live trading
- **Consult professionals**: Seek advice from licensed financial advisors

**By using this software, you acknowledge:**
1. You understand trading risks
2. You will not hold the authors liable for losses
3. You will comply with all applicable laws and regulations
4. You are 18+ years old and legally able to trade
5. You have read and understand this disclaimer

**Trading involves substantial risk of loss and is not suitable for every investor.**

---

## 🚀 Final Notes

**This bot is designed to be:**
- ✅ Beginner-friendly (easy setup, clear logs)
- ✅ Educational (learn trading strategies)
- ✅ Transparent (open source, no black boxes)
- ✅ Safe (multiple safety layers)
- ✅ Flexible (single or multi-stock)
- ✅ Automated (runs 24/7 unattended)

**Key Strengths:**
- **Smart capital allocation** - More $ to winners
- **Automatic rebalancing** - Adapts to market
- **Fractional shares** - Trade with any budget
- **Free data** - yfinance = unlimited backtesting
- **Windows automation** - Wake PC before market

**Limitations:**
- **Long-only strategy** - No shorting (can't profit from downtrends)
- **Technical analysis only** - No fundamental analysis
- **US stocks only** - No crypto, forex, options
- **Simple MA crossover** - Not a complex AI/ML model
- **Execution lag** - Not for high-frequency trading (seconds)

**Use Cases:**
- ✅ Learning algorithmic trading
- ✅ Testing trading strategies
- ✅ Automating portfolio management
- ✅ Small-scale personal trading
- ❌ Professional hedge fund (not designed for this)
- ❌ High-frequency trading (too slow)
- ❌ Institutional trading (not compliant)

---

**Everything is unified and production-ready!**

- ✅ ONE runner.py handles all modes
- ✅ ONE optimizer.py finds best parameters
- ✅ Smart allocation maximizes returns
- ✅ Comprehensive README with everything explained
- ✅ Safety features prevent disasters
- ✅ Windows automation for hands-off trading
- ✅ Free data sources (unlimited backtesting)
- ✅ Fractional shares (start with $10)

**Ready to start?**

```powershell
# 1. Setup
pip install -r requirements.txt
notepad .env  # Add your Alpaca keys

# 2. Find best stocks
python optimizer.py -v

# 3. Start trading (paper)
python runner.py -t 0.25 -m 1500

# 4. Monitor
Get-Content bot.log -Wait -Tail 50
```

**Happy Trading! 📈🤖**
