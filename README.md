# Smart Paper Trading Bot 🤖📈

A fully autonomous, "Smart" trading bot that acts like a hedge fund analyst. It reasons about the market using Machine Learning, Market Regime detection, and Technical Analysis to make dynamic, probability-weighted decisions. It operates via the **Alpaca Trade API**.

## 🧠 The "Brain" (Decision Engine)
The bot synthesizes multiple data points to generate `TradeSignal` objects:
- **ML Prediction**: Uses a Random Forest model (`ml_predictor.py`) to predict price direction based on historical patterns.
- **Market Regime**: Detects if the market is Trending (Bull/Bear), Sideways, or Volatile (`strategies/regime_detection.py`).
- **Technical Analysis**: Crossover logic, RSI, ATR, and Volume confirmation (`strategies/decision_engine.py`).

**Reasoning Example:**
> "Buying AAPL because Regime is Trending Up, ML confirms High Confidence (0.85), and Volume is spiking."

## 💰 The "Banker" (Allocation Engine)
It protects capital using institutional-grade risk management (`risk/allocation_engine.py`):
- **Dynamic Sizing**: Dynamically sizes positions based on signal confidence.
- **Portfolio Constraints**: Enforces Max Drawdown, Max Daily Loss, exposure caps, and position limits (e.g. strict TSLA allocation floors).
- **Kill Switches**: Tracks a High Water Mark (HWM) and halts trading if equity drops below configured thresholds.

## 🛠️ Tech Stack
- **Language**: Python 3.11+
- **APIs**: Alpaca Trade API (primary), yfinance (fallback data)
- **Data & ML**: `pandas`, `numpy`, `scikit-learn` (RandomForestClassifier)
- **Configuration**: `pydantic` and `pydantic-settings` (type-safe environment variables)
- **Dashboard UI**: Flask, HTML, Vanilla JS/CSS
- **Storage**: SQLite (`price_cache.db`), JSON (`portfolio.json`, `trade_history.json`, `dashboard_state.json`)
- **Supervision**: PowerShell (`botctl.ps1` via Windows Task Scheduler)

## 📂 Directory Structure
- `runner.py`: The central orchestrator loop (analyzing, trading, sleeping).
- `dashboard.py`: A lightweight Flask server serving the live web UI.
- `portfolio_manager.py`: Tracks held positions and transaction history.
- `ml_predictor.py`: Machine learning model training and inference.
- `stock_scanner.py`: Dynamically scans and ranks S&P 500 stocks.
- `strategies/`: Core trading logic (`decision_engine.py`, `regime_detection.py`, `news_sentinel.py`).
- `risk/`: Money management (`allocation_engine.py`).
- `execution/`: Order routing and Alpaca integration (`orders.py`).
- `utils/`: Locking and helpers (`file_lock.py`, `process_lock.py`, `market_schedule.py`).
- `templates/`: HTML templates for the dashboard (`dashboard.html`).

## ⚙️ Environment Variables
Configuration is handled via `.env` and typed in `config_validated.py`. Key variables include:

### API Keys
- `ALPACA_API_KEY`: Your Alpaca public key.
- `ALPACA_SECRET_KEY`: Your Alpaca secret key.
- `POLYGON_API_KEY`: (Optional) Polygon data key.
- `DISCORD_WEBHOOK_URL`: (Optional) For trade notifications. <!-- TODO: verify webhook usage -->

### Trading Settings
- `CONFIRM_GO_LIVE`: Must be set to `YES` to execute real trades. Otherwise, the bot simulates execution.
- `TRADE_SIZE_FRAC_OF_CAP`: Fraction of capital to risk per trade.
- `RISKY_MODE_ENABLED`: If true, relaxes some risk constraints.
- `MAX_POSITIONS`: Maximum number of concurrent stock positions.

### Dashboard Settings
- `DASHBOARD_HOST`: Host for the Flask server (default: `127.0.0.1`)
- `DASHBOARD_PORT`: Port for the Flask server (default: `5000`)
- `DASHBOARD_PASSWORD`: Password to secure the web UI.

## 🚀 Installation & Running

### 1. Configure
Create a `.env` file with your credentials (see `.env.example` if available, or create manually):
```env
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
CONFIRM_GO_LIVE=NO
DASHBOARD_PASSWORD=secure_password
```

### 2. Run the Bot and Dashboard Manually
**Bot:**
```bash
python runner.py
```
**Dashboard:**
```bash
python dashboard.py
```

### 3. Run as a Windows Service (Using `botctl.ps1`)
Use the PowerShell orchestrator to manage the bot via Windows Task Scheduler. Run these in an **Administrator** PowerShell:
```powershell
# Start the bot (creates scheduled tasks to run 24/7)
.\botctl.ps1 start python -u runner.py

# Check status of the bot and dashboard
.\botctl.ps1 status

# Stop tasks
.\botctl.ps1 stop

# Remove tasks entirely
.\botctl.ps1 remove
```

## 🌐 API Endpoints (Dashboard)
The Flask dashboard (`dashboard.py`) runs locally and exposes the following endpoints (protected by `DASHBOARD_PASSWORD`):
- `GET /`: Serves the HTML dashboard frontend.
- `GET /api/state`: Returns live portfolio state, bot metrics, and cached symbols. Uses cache-busting headers.
- `GET /api/history`: Returns the localized trading history (`trade_history.json`).
- `GET /api/logs`: Returns the last 100 lines of exactly `bot.log`. 

## 📝 Scripts & Utilities
- `optimizer.py`: Runs backward-looking simulations to find optimal intervals and capital configurations for specific symbols.
- `train_ml_model.py`: Manually forces a full retraining of the Random Forest model on 17 diverse tickers.
- `test_all.py`: Comprehensive test suite (`pytest test_all.py`).
- `validate_setup.py`: Utility to verify environment setup and API connectivity.
- `stress_test.py`: Runs thousands of simulated trades to verify the stability of the allocation engine.

## ⚠️ Disclaimer
This bot is designed for paper trading and educational purposes. If `CONFIRM_GO_LIVE=YES` is used, it will execute real orders. **Use at your own financial risk.**
