## Paper Trading Bot (Alpaca + Polygon)

- SMA crossover strategy with dynamic sizing and safety rules
- Alpaca data primary with automatic Polygon fallback
- Runs continuously as a hidden Windows Scheduled Task via `botctl.ps1`

### Requirements
- Python 3.9+
- Alpaca account (paper by default)
- Polygon API key (fallback + projections)

### Setup
1) Install dependencies:
```powershell
pip install -r requirements.txt
```
2) Create `.env` in project root:
```env
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
POLYGON_API_KEY=your_polygon_key
# Optional
CONFIRM_GO_LIVE=YES              # required for --go-live
ENABLE_MARKET_HOURS_ONLY=1       # 1/0 (default 1)
ALLOW_MISSING_KEYS_FOR_DEBUG=0   # 1/0 (default 0)
```

### Run directly (foreground)
Key CLI flags (see `runner.py`):
- `-t, --time <hours>`: polling interval in hours. Examples: `2` (2h), `.25` (15m), `.0065` (~23s).
- `-s, --symbol <TICKER>`: symbol to trade. Default from `config.DEFAULT_TICKER`.
- `-m, --max-cap <USD>`: per-symbol maximum capital exposure. Bot won’t buy above this.
- `--tp <percent>`: base take-profit percent; may be adjusted dynamically.
- `--sl <percent>`: base stop-loss percent; may be adjusted dynamically.
- `--frac <0..1>`: base fraction of remaining cap to buy per signal; can be tuned by confidence/volatility.
- `--fixed-usd <USD>`: use fixed dollar buys instead of fraction (0 disables).
- `--no-dynamic`: disable dynamic sizing/TP/SL; use provided base values only.
- `--go-live`: enable live trading (requires `CONFIRM_GO_LIVE=YES` in `.env`).
- `--allow-missing-keys`: bypass key requirement (debug/dev only).

Examples:
```powershell
python runner.py -t 2 -m 1000
python runner.py -t .25 -s TSLA -m 100
```

### Run as hidden task (recommended)
Install/start (Admin):
```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" start python -u runner.py -t .0065 -m 1000
```
Controls (Admin):
```powershell
# Pause (disable triggers + stop current run)
pwsh -NoProfile -ExecutionPolicy Bypass -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" pause
# Resume (enable triggers + start)
pwsh -NoProfile -ExecutionPolicy Bypass -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" resume
# Restart (clean: end task, kill strays, recreate)
schtasks /End /TN PaperTradingBot 2>$null; Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'runner.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; Get-CimInstance Win32_Process -Filter "Name='pwsh.exe'" | Where-Object { $_.CommandLine -match 'start_bot.ps1' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; pwsh -NoProfile -ExecutionPolicy Bypass -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" restart python -u runner.py -t .0065 -m 1000
# Stop (keeps task; starts again at boot/logon)
pwsh -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" stop
# Stop forever (delete the scheduled task, then remove only regenerable files)
pwsh -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" stop-forever; Remove-Item "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\bot.log","C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\last_start_cmd.txt","C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\start_bot.ps1" -Force -ErrorAction SilentlyContinue
# Status
pwsh -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" status
```
Controller command reference (`botctl.ps1`):
- `start <python ...>`: creates/updates the scheduled task and runs it immediately. First arg must be `python`. Example: `start python -u runner.py -t 2 -m 1000`.
- `pause`: stops the running task instance and disables triggers (won’t auto-start).
- `resume`: enables triggers and starts the task now.
- `restart`: stops then starts the existing task once (triggers unchanged).
- `stop`: stops current run; task remains enabled and will start at boot/logon/daily.
- `stop-forever`: stops and deletes the scheduled task.
- `status`: shows next run time, last run time, and last result.
Logs:
```powershell
Get-Content "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\bot.log" -Wait
```

Print once (full path):
```powershell
Get-Content "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\bot.log"
```

### Behavior
- Backfills enough bars at startup (no long warm-up wait)
- Polygon fallback if Alpaca data fails; logs “Using Polygon fallback …”
- Market-hours guard with pre-open wake (5 minutes before open)
- Safety: max drawdown, max position age, daily loss limit

### Paper vs Live
Paper (default):
- `config.ALPACA_BASE_URL` points to paper API
- Use paper keys in `.env`

Live (safe ramp-up):
1) In `.env`: set live `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` and `CONFIRM_GO_LIVE=YES`
2) Optional in `config.py`: `ALPACA_BASE_URL = "https://api.alpaca.markets"`
3) Validate during market hours conservatively:
```powershell
python runner.py --go-live -t .0065 -m 50
```
(optional clock check):
```powershell
python -c "import config as cfg; from alpaca_trade_api import REST; c=REST(cfg.ALPACA_API_KEY, cfg.ALPACA_SECRET_KEY, base_url=cfg.ALPACA_BASE_URL); print(bool(getattr(c.get_clock(),'is_open', False)))"
```
4) Continuous live (hidden task):
```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" start python -u runner.py --go-live -t .0065 -m 50
```
5) Observe, then scale `-m` and sizing gradually

### Troubleshooting
- Prefer DNS 1.1.1.1/8.8.8.8; verify with `Resolve-DnsName` / `Test-NetConnection`
- Force Polygon (test only): set `FORCE_POLYGON_FALLBACK=1`
- Ensure `.env` is at project root so the scheduled task (SYSTEM) can read it
