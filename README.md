# Smart Paper Trading Bot 🤖📈

A fully autonomous, "Smart" trading bot that acts like a hedge fund analyst. It doesn't just look at one indicator; it **reasons** about the market using Machine Learning, Market Regime detection, and Technical Analysis to make dynamic, probability-weighted decisions.

## 🧠 The "Brain" (Decision Engine)
The bot acts like it knows "everything" by synthesizing multiple data points:
- **ML Prediction**: Uses a Random Forest model to predict price direction based on historical patterns.
- **Market Regime**: Detects if the market is Trending (Bull/Bear), Sideways, or Volatile.
- **Technical Analysis**: Crossover logic, RSI, and Volume confirmation.

**Reasoning Example:**
> "Buying AAPL because Regime is Trending Up, ML confirms High Confidence (0.85), and Volume is spiking."

## 💰 The "Banker" (Allocation Engine)
It protects your capital using institutional-grade risk management:
- **Kelly Criterion**: Dynamically sizes positions based on win probability.
- **Volatility Scaling**: Automatically reduces position size in high-volatility environments.
- **Risk Limits**: Enforces Max Drawdown, Max Daily Loss, and Position Limits.

## 🚀 Quick Start

### 1. Configure
Create a `.env` file with your Alpaca API keys:
```env
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 2. Run
```bash
python runner.py
```
That's it! The bot will:
- Auto-train its ML model (if needed).
- Connect to the market.
- Start its "Always On" loop (analyzing, trading, sleeping).

## Key Features
- **Always On**: Runs continuously, sleeping intelligently between intervals.
- **Resilient**: Robust retry logic for API calls; never crashes on a single error.
- **Smart Data**: Uses SQLite caching to limit API usage and speed up analysis.
- **Dynamic**: Doesn't use fixed sizes; adjusts every trade based on **Confidence**.

## Architecture
- `strategies/decision_engine.py`: The reasoning core.
- `risk/allocation_engine.py`: The money manager.
- `execution/orders.py`: The safe execution handler.
- `runner.py`: The central orchestrator.

## Disclaimer
Paper trading only. Use at your own risk.
