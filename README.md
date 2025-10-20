## Paper Trading Bot (Alpaca + Polygon)

Simple local paper-trading bot that:
- Allocates $100 to `TSLA` on first run
- Every hour: pulls hourly bars from Polygon, runs a simple EMA crossover, and either buys more (notional) or sells the full position
- Sends market orders via Alpaca Paper API

### Requirements
- Python 3.9+
- Alpaca account with paper trading enabled
- Polygon API key (free tier works for aggregates)

### Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file (not committed) with your keys, or set env vars:
   ```env
   ALPACA_API_KEY=your_key
   ALPACA_API_SECRET=your_secret
   POLYGON_API_KEY=your_polygon_key
   ```
   `config.py` will auto-load `.env` if present. You can tweak strategy/runtime params in `config.py`.

### Usage
- Run with defaults:
  ```bash
  python runner.py
  ```
- Change polling interval (in hours):
  ```bash
  python runner.py -t 0.5   # every 30 seconds
  ```
- Change symbol and per-symbol max cap:
  ```bash
  python runner.py -s TSLA -m 1000
  ```
- Live trading (requires `CONFIRM_GO_LIVE=YES`):
  ```bash
  python runner.py --go-live
  ```

### How it works
- Each loop it fetches bars from Alpaca, computes a short/long SMA crossover, and decides buy/sell/hold.
- Buys size dynamically based on confidence, volatility and remaining cap; sells all or partial per settings.
- Orders are `market` (`time_in_force=day`).

### Notes
- This is for paper trading only by default (`ALPACA_BASE_URL` points to the paper API). Switching to live trading is out of scope here.
- Keys are read from environment variables (or `.env` via python-dotenv). Use `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. Do not commit your real keys.


