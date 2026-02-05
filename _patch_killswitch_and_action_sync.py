from pathlib import Path
import re, time

ts = time.strftime("%Y%m%d_%H%M%S")

# --- config_validated.py: clarify max_cap_usd meaning (doc only) ---
cv = Path('config_validated.py')
cv_bak = cv.with_suffix(cv.suffix + f'.bak_killswitch_{ts}')
cv_bak.write_text(cv.read_text(encoding='utf-8'), encoding='utf-8')
cv_txt = cv.read_text(encoding='utf-8')
cv_txt = cv_txt.replace(
    'max_cap_usd: float = Field(default=100.0, ge=1.0, description="Maximum capital per symbol in USD")',
    'max_cap_usd: float = Field(default=100.0, ge=0.0, description="KILL SWITCH: if account equity drops below this USD, stop bot (0 disables)")'
)
cv.write_text(cv_txt, encoding='utf-8')

# --- runner.py: add kill-switch + enforce desired==executed as FINAL action ---
rp = Path('runner.py')
rp_bak = rp.with_suffix(rp.suffix + f'.bak_killswitch_{ts}')
rp_bak.write_text(rp.read_text(encoding='utf-8'), encoding='utf-8')
rtxt = rp.read_text(encoding='utf-8')

# 1) Kill switch: after virtual_account_size override (so it uses final equity value used for trading)
if 'KILL SWITCH TRIGGERED' not in rtxt:
    # place it right after the virtual_account_size block where equity is set
    rtxt = re.sub(
        r"(if self\.config\.virtual_account_size:\s*\n\s*log_info\(f\"Using Virtual Account Size: \$\{self\.config\.virtual_account_size:[^\n]+\n\s*equity = self\.config\.virtual_account_size\s*\n)",
        r"\1\n        # Kill switch: stop bot if equity drops below configured floor (max_cap_usd).\n        try:\n            floor = float(getattr(self.config, 'max_cap_usd', 0.0) or 0.0)\n            if floor > 0 and equity < floor:\n                log_error(f\"KILL SWITCH TRIGGERED: equity ${equity:.2f} < floor ${floor:.2f}. Exiting.\")\n                try:\n                    self._dump_dashboard_state(equity, 'KILL_SWITCH', None)\n                except Exception:\n                    pass\n                raise SystemExit(2)\n        except SystemExit:\n            raise\n        except Exception:\n            pass\n",
        rtxt,
        count=1,
    )

# 2) Make desired_action/executed_action always the SAME FINAL action.
# We already track _desired_action/_executed_action; now we set both from signal.action AFTER we know what we actually did.

# Ensure we set executed_action to 'hold' when order not placed
rtxt = rtxt.replace(
    'log_warn(f"ORDER NOT PLACED for {symbol} (see earlier warnings).")',
    'log_warn(f"ORDER NOT PLACED for {symbol} (see earlier warnings).")\n                            try:\n                                # If broker rejected / we failed to place order, final action is HOLD\n                                signal.action = "hold"\n                                self._blocked_reason = "order_not_placed"\n                            except Exception:\n                                pass'
)

# Ensure we set both desired/executed from final signal.action before dumping dashboard state
if '_final_action_synced' not in rtxt:
    rtxt = rtxt.replace(
        'self._dump_dashboard_state(equity, symbol, signal, next_updates=self.schedule)',
        'try:\n                self._desired_action = str(getattr(signal, "action", "")).lower()\n                self._executed_action = self._desired_action\n            except Exception:\n                pass\n            self._dump_dashboard_state(equity, symbol, signal, next_updates=self.schedule)\n            self._final_action_synced = True'
    )

rp.write_text(rtxt, encoding='utf-8')

print('patched: kill switch + desired_action==executed_action as final action')
