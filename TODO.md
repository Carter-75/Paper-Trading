# Live Trading Cutover Checklist (Safe Ramp-Up)

Use this list when switching from paper to live on Alpaca. Keep this file as your source of truth; check items off as you complete them.

## 1) Configure Live Environment
- [ ] Update `config.py`:
  - [ ] Set `ALPACA_BASE_URL = "https://api.alpaca.markets"`
  - [ ] Set `ALPACA_API_KEY` and `ALPACA_API_SECRET` to your LIVE keys
- [ ] In Alpaca dashboard:
  - [ ] Enable Live Trading and confirm account status is active
  - [ ] Deposit funds and ensure cash is available (settled)

## 2) Reduce Initial Risk
- [ ] Lower cap (`-m`) for first session (e.g., `$50`)
  - Explanation: Smaller cap proves live flow without maxing exposure.
- [ ] Keep dynamic sizing conservative
  - [ ] Set `TRADE_SIZE_FRAC_OF_CAP` lower (e.g., `0.1`)
  - [ ] Optionally set `FIXED_TRADE_USD` (e.g., `10`) to force small buys
- [ ] Keep protections on
  - [ ] `TRAILING_STOP_PERCENT` > 0 (e.g., 3.0)
  - [ ] `MAX_DRAWDOWN_PERCENT` > 0 (e.g., 6.0)

## 3) One-Shot Live Validation (during market open)
- [ ] Run a single decision to verify live connectivity and orders:
  - PowerShell (foreground):
    ```powershell
    python runner.py
    ```
  - Confirm: no missing keys, no rejections, and expected “Decision: …” output.
  - Optional: verify Alpaca market clock (True during 9:30–16:00 ET):
    ```powershell
    python -c "import config as cfg; from alpaca_trade_api import REST; c=REST(cfg.ALPACA_API_KEY, cfg.ALPACA_API_SECRET, base_url=cfg.ALPACA_BASE_URL); print(bool(getattr(c.get_clock(),'is_open', False)))"
    ```

## 4) Start Continuous Live (conservative flags)
- [ ] Start via controller with conservative flags (example: every ~23s, cap $50):
  ```powershell
  pwsh -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" start python -u runner.py -t .0065 -m 50
  ```
- [ ] Tail logs to monitor decisions:
  ```powershell
  Get-Content "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\bot.log" -Wait
  ```
- [ ] Alternative (avoid policy issues):
  ```powershell
  pwsh -NoProfile -ExecutionPolicy Bypass -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" start python -u runner.py -t .0065 -m 50
  ```

## 5) Observe and Scale Up Gradually
- [ ] After a stable session (no rejections, fills as expected), raise:
  - [ ] Cap (`-m`) in small steps (e.g., +$25–$50)
  - [ ] Per-buy (`-p`) to match risk tolerance
- [ ] Continue monitoring logs and Alpaca dashboard (positions/orders) while scaling.

## 6) Operational Controls
- [ ] Update flags quickly (without editing files):
  ```powershell
  pwsh -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" start python -u runner.py -t .0065 -m 75
  ```
- [ ] Start (hidden; run as Administrator for wake/boot triggers):
  ```powershell
  pwsh -NoProfile -ExecutionPolicy Bypass -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" start python -u runner.py -t .0065
  ```
- [ ] Stop (will auto-start again on next logon/boot unless removed):
  ```powershell
  pwsh -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" stop
  ```
- [ ] Restart (shows Access is denied if prior run was SYSTEM; harmless):
  ```powershell
  pwsh -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" restart
  ```
- [ ] Status:
  ```powershell
  pwsh -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" status
  ```
- [ ] Remove permanently (won’t come back):
  ```powershell
  pwsh -File "C:\Users\carte\OneDrive\Desktop\Code\Paper-Trading\botctl.ps1" remove
  ```
- [ ] Verify wake/boot triggers (daily 9:25 and WakeToRun):
  ```powershell
  Get-ScheduledTask -TaskName PaperTradingBot | Select-Object -ExpandProperty Triggers
  ```

## 7) Rollback to Paper (if needed)
- [ ] In `config.py`, revert to paper:
  - [ ] `ALPACA_BASE_URL = "https://paper-api.alpaca.markets"`
  - [ ] Use paper keys
- [ ] Restart controller with paper-safe flags as above.


