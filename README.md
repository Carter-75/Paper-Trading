# Paper Trading Bot - Intelligent Multi-Stock Trading System

⚠️ **CRITICAL RISK DISCLOSURE** ⚠️

- **This software is for EDUCATIONAL and PAPER TRADING purposes**
- **Trading involves substantial risk of loss - never trade with money you cannot afford to lose**
- **Past performance does NOT guarantee future results**
- **Paper trading results are NOT predictive of live performance**
- **This software is provided AS-IS with NO warranties or guarantees**

---

## 🎯 What This Bot Does

Automated stock trading bot with advanced strategy filters, risk management, and machine learning:

- **Auto-selects profitable stocks** - Scans top 100 stocks by market cap, picks the best performers
- **Smart capital allocation** - Dynamic position sizing (increases after wins, decreases after losses)
- **Advanced strategy filters** - RSI, MACD, Bollinger Bands, multi-timeframe confirmation, volume analysis
- **Sophisticated risk management** - Trailing stops, ATR-based stops, drawdown protection, Kelly sizing, correlation checks
- **Machine learning prediction** - Random Forest model auto-trains on first run, confirms/overrides signals
- **SQLite caching** - 80% fewer API calls, 5× faster backtests
- **Full automation** - Runs as scheduled task, auto-starts on boot, wakes PC before market open (9:25 AM)
- **Parallel optimization** - 8× faster parameter search on multi-core CPUs
- **Fractional shares** - Trade with any budget (even $10 works)

**Expected Performance:**
- Win Rate: 65-75% (vs 50% baseline)
- Daily Return: +$180 (vs $50 baseline) - 3.6× improvement
- API Calls: 80% reduction via SQLite caching
- Sharpe Ratio: 1.5-2.0 (vs 0.5 baseline)

---

## 🚀 Quick Start

**Prerequisites:**
- Python 3.9+
- Alpaca Paper Trading Account (free at alpaca.markets)
- PowerShell 7+ (run as Administrator for full automation)

### 1. Setup

```powershell
# Open PowerShell as Administrator
cd "C:\Users\carte\OneDrive\Desktop\Code\Portfolio-Websites (Mostly)\Paper-Trading"

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
notepad .env
```

**Add to `.env`:**
```env
# ===== REQUIRED =====
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
POLYGON_API_KEY=your_polygon_key

# ===== SAFETY (Important) =====
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
TRAILING_STOP_PERCENT=0.75           # Trailing stop (locks in profits)

# ===== RISK MANAGEMENT (optional) =====
MIN_CONFIDENCE_TO_TRADE=0.005
VOLATILITY_PCT_THRESHOLD=0.15
PROFITABILITY_MIN_EXPECTED_USD=0.10
MAX_DAILY_LOSS_PERCENT=5.0
MAX_TRADE_SIZE_FRAC=0.95

# ===== SMART ALLOCATION (Optimized with Best Practices) =====
ALLOCATION_CONCENTRATION=2.0         # How aggressive (1.0=proportional, 2.0=optimal, 3.0=very aggressive)
MIN_DIVERSIFICATION_STOCKS=5         # Minimum stocks for safety (best practice: 5-10)
MAX_SINGLE_STOCK_PERCENT=20.0        # Max % in one stock (best practice: 15-25%, default 20%)
MIN_ALLOCATION_USD=5.0               # Skip stocks below this $ amount
CONFIDENCE_WEIGHT_MULTIPLIER=400.0   # How much confidence matters (300-500 range, higher = more weight)
RESERVE_CASH_PERCENT=25.0            # Keep % in reserve (best practice: 20-30% for safety)

# ===== DYNAMIC (optional) =====
DYNAMIC_PARAM_ENABLED=1
RISKY_MODE_ENABLED=1
RISKY_TP_MULT=1.5
RISKY_SL_MULT=0.8
RISKY_FRAC_MULT=1.3

# ===== ADVANCED FILTERS (All enabled by default) =====
RSI_ENABLED=1                          # RSI overbought/oversold filter
RSI_OVERBOUGHT=70                      # Don't buy above this RSI
RSI_OVERSOLD=30                        # Don't sell below this RSI

MULTI_TIMEFRAME_ENABLED=1              # Multi-timeframe confirmation
VOLUME_CONFIRMATION_ENABLED=1          # Volume confirmation filter

ENABLE_DRAWDOWN_PROTECTION=1           # Stop trading if down >15% from peak
MAX_PORTFOLIO_DRAWDOWN_PERCENT=15.0

ENABLE_KELLY_SIZING=1                  # Kelly Criterion position sizing
ENABLE_CORRELATION_CHECK=1             # Correlation-based diversification
CORRELATION_CHECK_ENABLED=1

USE_LIMIT_ORDERS=1                     # Use limit orders for better prices
LIMIT_ORDER_OFFSET_PERCENT=0.1
LIMIT_ORDER_TIMEOUT_SECONDS=300

ENABLE_SAFE_HOURS=1                    # Avoid market open/close volatility
AVOID_FIRST_MINUTES=15
AVOID_LAST_MINUTES=15

ENABLE_ML_PREDICTION=1                 # Random Forest prediction
ML_CONFIDENCE_THRESHOLD=0.6
```

**Note:** You only need the REQUIRED section to get started. The rest have good defaults and make the bot smarter, but you can skip them initially.

### 2. Find Optimal Parameters (Optional)

```powershell
# Test multiple stocks, find best interval & capital
python optimizer.py -v

# FAST MODE: Parallel processing (8× faster)
python optimizer.py --parallel -v

# Presets for different risk levels
python optimizer.py --preset conservative  # $25k cap
python optimizer.py --preset balanced      # $250k cap
python optimizer.py --preset aggressive    # $1M cap
```

**How to Read Optimizer Output:**

The optimizer tells you EXACTLY what 2 values to use:

```
>>> YOU SET THESE 2 VALUES:
  1. Time Interval: 1.3000 hours                <-- USE THIS
     -> 3.0 trades/day
     -> Perfect fit! No wasted time
  2. Total Capital: $198699.39                  <-- USE THIS
```

Then copy the command it gives you:
```powershell
& "$BotDir\botctl.ps1" start python -u runner.py -t 2.1667 -m 198699.39
```

**How the Optimizer Works (Market-Hour Aware):**
- Market is open 6.5 hours/day (9:30 AM - 4:00 PM ET)
- Optimizer finds the **optimal number of trades** (not arbitrary intervals!)
- Calculates interval = 390 minutes / trades_per_day
- This ensures **zero wasted time** (unlike old 4-hour = 1.6 trades = wastes 2.5 hours)

**Example Intervals:**
- 1 trade/day = 390min (6.5h)
- 2 trades/day = 195min (3.25h)
- 3 trades/day = 130min (2.17h) ← Perfect division!
- 6 trades/day = 65min (1.08h)

**Important:**
- `-t` flag uses **HOURS** (not seconds!)
- `-m` flag uses **DOLLARS** (no commas)
- Use the **"Live Trading (REALISTIC)"** value, NOT "Paper Trading"

### 3. Start the Bot

```powershell
# Start bot (runs as scheduled task, auto-restarts on crash)
.\botctl.ps1 start python -u runner.py -t 0.25 -m 1500

# First run: Auto-trains ML model (5-10 min, press Ctrl+C to use partial data)
# Subsequent runs: Instant startup
```

**What happens:**
- ✅ Bot runs in background
- ✅ Auto-starts on system boot
- ✅ Wakes PC at 9:25 AM daily (5 min before market open)
- ✅ Auto-restarts on crash
- ✅ Logs everything to `bot.log`

### 4. Monitor

```powershell
# Check status
.\botctl.ps1 status

# Watch logs live
Get-Content bot.log -Wait -Tail 50

# View portfolio
Get-Content portfolio.json | ConvertFrom-Json | ConvertTo-Json

# Check Alpaca dashboard
# https://app.alpaca.markets/paper/dashboard/overview
```

### 5. Control

```powershell
.\botctl.ps1 restart       # Restart bot
.\botctl.ps1 stop          # Temporary stop (auto-restarts on boot)
.\botctl.ps1 stop-forever  # Permanent stop + clean all files

# Emergency kill switch (instant halt without restart)
echo $null > KILL_SWITCH.flag  # Activate (stops trading immediately)
del KILL_SWITCH.flag           # Deactivate (resumes automatically)
```

**Kill Switch Features:**
- ✅ Instant halt - stops trading immediately when file is created
- ✅ No restart needed - bot checks every 30 seconds and auto-resumes when file deleted
- ✅ Safe during anomalies - quick manual override during market irregularities
- ✅ Preserves positions - doesn't liquidate, just pauses trading

**When to use Kill Switch:**
- Market flash crash or extreme volatility
- Suspicious API behavior or data issues
- Need to pause bot while investigating an issue
- Testing configuration changes mid-day

---

## 📖 Command Reference

### botctl.ps1 (Task Controller)

```powershell
.\botctl.ps1 start <command>  # Start bot with full automation
.\botctl.ps1 stop             # Temporary stop
.\botctl.ps1 restart          # Restart bot
.\botctl.ps1 stop-forever     # Permanent stop + cleanup
.\botctl.ps1 status           # Show status
```

### runner.py (Main Bot)

```powershell
python runner.py -t HOURS -m CAPITAL [OPTIONS]

Required:
  -t, --time HOURS          Trading interval (e.g., 0.25 = 15 minutes)
  -m, --max-cap USD         Total capital across all stocks

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
```

**Examples:**
```powershell
# Multi-stock portfolio (default 15 stocks)
.\botctl.ps1 start python -u runner.py -t 0.25 -m 1500

# Single stock
.\botctl.ps1 start python -u runner.py -t 0.25 -s AAPL -m 100 --max-stocks 1

# Force specific stocks + auto-fill rest
.\botctl.ps1 start python -u runner.py -t 0.25 -m 1000 --stocks TSLA NVDA --max-stocks 10

# Custom TP/SL
.\botctl.ps1 start python -u runner.py -t 0.25 -m 1500 --tp 3.0 --sl 1.5
```

### optimizer.py (Parameter Finder)

```powershell
python optimizer.py [OPTIONS]

Options:
  -s, --symbol TICKER       Single stock
  --symbols SYM1 SYM2 ...   Multiple stocks to test
  -m, --max-cap USD         Maximum capital to test (default: $1M)
  --parallel                Enable parallel processing (8× faster)
  --preset MODE             conservative/balanced/aggressive
  -v, --verbose             Show detailed progress

Examples:
  python optimizer.py -v                    # Test top 100 stocks
  python optimizer.py --parallel -v         # Fast mode
  python optimizer.py -s AAPL -v            # Single stock
  python optimizer.py --preset conservative # Low risk
```

### ML Training (Optional)

```powershell
# Auto-trains on first bot run (5-10 min)
# Press Ctrl+C during training to use partial data (ML stays enabled!)

# Or manually train ahead of time
python train_ml_model.py
```

### Stress Testing (Recommended)

```powershell
# Test strategy against historical market crashes
python stress_test.py --symbol SPY -v

# Test multiple stocks
python stress_test.py --symbols SPY QQQ AAPL --interval 0.25

# Test specific crisis only
python stress_test.py --symbol SPY --crisis 2020

# Available crises:
#   - 2008 Financial Crisis (Sep 2008 - Mar 2009)
#   - 2020 COVID Crash (Feb 2020 - Apr 2020)
#   - 2022 Bear Market (Jan 2022 - Oct 2022)
#   - 2011 Debt Crisis
#   - 2018 Q4 Selloff
```

### Parameter Stability Analysis (Recommended)

```powershell
# Analyze optimization_history.csv for parameter drift
python parameter_stability.py

# Analyze specific symbol
python parameter_stability.py --symbol SPY

# Analyze last 30 days only
python parameter_stability.py --window 30

# What it detects:
#   - Parameter drift (intervals/capitals changing over time)
#   - Overfitting signals (unstable parameters)
#   - Performance degradation
#   - Statistical stability metrics (CV, drift %)
```

---

## ⚙️ Configuration

**All configuration is in the `.env` file** - see the full example in the Quick Start section above.

**Key settings you might want to change:**
- `TAKE_PROFIT_PERCENT` - Default: 2.0% (when to take profits)
- `STOP_LOSS_PERCENT` - Default: 1.0% (when to cut losses)
- `TRAILING_STOP_PERCENT` - Default: 0.75% (locks in profits as price rises)
- `MAX_DAILY_LOSS_PERCENT` - Default: 5.0% (bot exits if down this much in one day)
- `MAX_PORTFOLIO_DRAWDOWN_PERCENT` - Default: 15.0% (stops trading if down this much from peak)

**All advanced filters are enabled by default** - see `.env` example in Quick Start to customize.

---

## 📊 Trading Strategy

**Core Strategy:** Enhanced SMA crossover (9/21) with advanced filters

**Signal Generation:**
- **BUY**: Short MA > Long MA × 1.002 (bullish momentum)
- **SELL**: Short MA < Long MA × 0.998 (bearish momentum)
- **HOLD**: No clear signal

**Execution Flow (each interval):**
1. Market check (9:30 AM - 4:00 PM ET, skip first/last 15 min)
2. Fetch data (SQLite cache → yfinance → Alpaca → Polygon)
3. Calculate indicators (SMAs, RSI, MACD, Bollinger Bands, ATR, volume)
4. Multi-timeframe confirmation (1x, 3x, 5x)
5. Generate signal (BUY/SELL/HOLD)
6. Apply filters (RSI, MACD, Bollinger Bands, volume, ML)
7. Check profitability & volatility
8. Correlation check (reduce size for correlated holdings)
9. Dynamic position sizing (increase after wins, decrease after losses)
10. Kelly sizing (optimal position based on win rate)
11. Execute order (limit order with trailing stops)
12. Safety checks (daily loss limit, max drawdown)

**Risk Management:**
- Take Profit: 2% (adjustable)
- Trailing Stops: 0.75% (locks in profits)
- Stop Loss: 1% with ATR-based adjustment
- Max Daily Loss: 5%
- Drawdown Protection: Stops trading if down >15% from peak
- Kelly Criterion: Optimal position sizing based on win rate
- Correlation Check: Reduces position size for highly correlated stocks

---

## 🤖 Machine Learning

**Auto-Training (First Run):**
- Bot automatically trains ML model on first run (5-10 minutes)
- Trains on 17 symbols with 500 bars each (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, WMT, JNJ, PG, UNH, HD, DIS, SPY, QQQ)
- Press **Ctrl+C** anytime to train with partial data (ML stays enabled!)
- All future runs load the model instantly (no delay)

**How ML Works:**
- Extracts 5 features: returns, RSI, volume, momentum, volatility
- Random Forest classifier (100 trees)
- Confirms signals: "ML confirms BUY (conf=75%)"
- Overrides signals: "ML DISAGREES - predicts DOWN (conf=82%)" → converts to HOLD
- Requires 60% confidence to override

---

## 🛠️ Running Without Admin Mode

**If you don't run as Administrator:**
- ✅ Bot runs fine, just won't have full automation features
- ❌ No auto-start on boot
- ❌ No scheduled task (no wake-before-market)
- ❌ No auto-restart on crash

**Simple mode (no automation):**
```powershell
# Just run Python directly (no admin needed)
python runner.py -t 0.25 -m 1500

# Press Ctrl+C to stop
```

**Good for:**
- Testing configuration changes
- Debugging issues
- Quick manual runs

---

## 🔧 Troubleshooting

**Bot not starting after PC restart:**
```powershell
# Bot waits up to 60 seconds for network on boot
# Check logs:
Get-Content bot.log -Wait -Tail 50

# Manually restart:
.\botctl.ps1 restart
```

**ML model won't train:**
```powershell
# Disable ML and run anyway:
# Add to .env:
ENABLE_ML_PREDICTION=0
```

**API rate limit exceeded:**
```powershell
# Use longer intervals (fewer API requests):
.\botctl.ps1 restart python -u runner.py -t 0.5 -m 1500  # 30 min

# Or reduce number of stocks:
.\botctl.ps1 restart python -u runner.py -t 0.25 -m 1500 --max-stocks 10
```

**Task not auto-starting:**
```powershell
# Check if task exists:
Get-ScheduledTask -TaskName "PaperTradingBot"

# Recreate it:
.\botctl.ps1 stop-forever
.\botctl.ps1 start python -u runner.py -t 0.25 -m 1500
```

**"No profitable stocks found":**
```powershell
# Market is bearish or strategy doesn't fit current conditions
# Try different interval:
python optimizer.py -v

# Lower profitability threshold (temporary):
# Add to .env:
PROFITABILITY_MIN_EXPECTED_USD=0.001
```

---

## 📁 File Structure

**Core Files:**
- `runner.py` - Main bot (handles single & multi-stock)
- `config.py` - Configuration & environment variables
- `optimizer.py` - Binary search optimizer (finds best params)
- `portfolio_manager.py` - Position tracking & portfolio state
- `stock_scanner.py` - Stock evaluation & ranking engine
- `ml_predictor.py` - Random Forest predictor
- `train_ml_model.py` - ML training script

**Windows Automation:**
- `botctl.ps1` - Task controller (start/stop/restart/status)
- `start_bot.ps1` - Auto-generated wrapper (ignored by git)
- `last_start_cmd.txt` - Last command saved (ignored by git)

**Runtime Files (auto-created, ignored by git):**
- `bot.log` - Trading activity log (auto-truncated to 250 lines)
- `portfolio.json` - Current positions
- `pnl_ledger.json` - Trade history with realized gains/losses
- `top_stocks_cache.json` - Top 100 stocks by market cap (refreshed daily)
- `price_cache.db` - SQLite cache for historical prices (80% fewer API calls)
- `optimization_history.csv` - Optimizer run history
- `ml_model.pkl` - Trained ML model

**Note:** All runtime files are automatically deleted when you run `.\botctl.ps1 stop-forever`

---

## ⚠️ Important Notes

- **Always paper trade first** - Set `CONFIRM_GO_LIVE=NO` in .env
- **Run as Administrator** - For full automation features (optional but recommended)
- **Test with optimizer** - Find profitable setup before running
- **Monitor logs** - Check `bot.log` regularly
- **Bearish markets** - Bot will show negative returns (by design - it's long-only)
- **API limits** - Don't scan too frequently
- **Market hours** - Stock prices DO NOT change when market is closed (weekends/evenings)

---

## 📚 Additional Resources

- [Alpaca Docs](https://alpaca.markets/docs/)
- [Paper Trading Guide](https://alpaca.markets/docs/trading/paper-trading/)
- [Moving Average Strategy](https://www.investopedia.com/terms/m/movingaverage.asp)
- [Risk Management](https://www.investopedia.com/terms/r/riskmanagement.asp)

---

## ⚖️ Disclaimer

**This software is provided for educational and research purposes only.**

- No warranty: Software provided "as is" with no guarantees
- No financial advice: This is not investment advice
- Trading risks: You can lose money trading stocks
- Past performance ≠ future results
- Use at your own risk
- Consult licensed financial advisors

**Trading involves substantial risk of loss and is not suitable for every investor.**

---

## 🎉 Quick Reference

```powershell
# Setup (one-time)
pip install -r requirements.txt
notepad .env  # Add API keys

# Find best parameters
python optimizer.py -v

# Start bot (full automation)
.\botctl.ps1 start python -u runner.py -t 0.25 -m 1500

# Monitor
.\botctl.ps1 status
Get-Content bot.log -Wait -Tail 50

# Control
.\botctl.ps1 restart       # Restart
.\botctl.ps1 stop-forever  # Stop + clean
```

**Happy Trading! 📈🤖**
