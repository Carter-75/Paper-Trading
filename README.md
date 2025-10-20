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
2. Open `config.py` and fill in:
   - `ALPACA_API_KEY`
   - `ALPACA_API_SECRET`
   - `POLYGON_API_KEY`
   You can also tweak strategy/runtime parameters there.

### Usage
- Run continuously with default runtime cap (24h):
  ```bash
  python runner.py
  ```
- Run with no time limit (override flag `-0`):
  ```bash
  python runner.py -0
  ```
- Run exactly one decision cycle and exit (useful for testing):
  ```bash
  python runner.py --once
  ```
- Change maximum runtime hours (ignored if `-0` is set):
  ```bash
  python runner.py --hours 8
  ```

### How it works
- On startup, if no `TSLA` position exists, it buys `$INITIAL_NOTIONAL_USD` (default $100).
- Each hour it fetches hourly closes from Polygon for the last `HOURS_BACK_FOR_TREND` hours.
- It computes a short and long EMA; if short > long it buys a dynamically sized notional (fraction of remaining cap), else it sells per signal strength.
- Orders are `market` and `time_in_force=day`.

### Notes
- This is for paper trading only by default (`ALPACA_BASE_URL` points to the paper API). Switching to live trading is out of scope here.
- Keys are stored locally in `config.py` since this is a local-only setup.


