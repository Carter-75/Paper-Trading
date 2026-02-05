from pathlib import Path
import re, time

p = Path('runner.py')
orig = p.read_text(encoding='utf-8')
ts = time.strftime('%Y%m%d_%H%M%S')
p.with_suffix(p.suffix + f'.bak_desired_exec_fix_{ts}').write_text(orig, encoding='utf-8')

txt = orig

# 1) Remove the "sync desired=executed before dump" patch we added earlier.
# It was inserted as a block starting with "try:\n                self._desired_action" ... and sets _final_action_synced.
txt = re.sub(r"\n\s*try:\n\s*self\._desired_action = str\(getattr\(signal, \\\"action\\\", \\\"\\\"\)\)\.lower\(\)\n\s*self\._executed_action = self\._desired_action\n\s*except Exception:\n\s*pass\n\s*self\._dump_dashboard_state\(equity, symbol, signal, next_updates=self\.schedule\)\n\s*self\._final_action_synced = True\s*", "\n            self._dump_dashboard_state(equity, symbol, signal, next_updates=self.schedule)", txt, count=1)
# Also remove stray _final_action_synced flag if present elsewhere
if '_final_action_synced' in txt:
    txt = txt.replace('self._final_action_synced = True', '')

# 2) Ensure desired/executed semantics:
# desired_action = final pre-exec decision (after allocation/fees/risk gating)
# executed_action = what actually happened (same as desired unless broker/order failure)

# We'll patch inside the existing block (the one we previously installed) that computes allocation and can_trade.
# We set desired/executed appropriately.

# Set desired_action after can_trade is computed: when not can_trade -> desired=hold, executed=hold
# when can_trade -> desired=signal.action (buy/sell), executed initially = desired

txt = txt.replace(
    'if not can_trade:\n                        try:\n                            self._executed_action = "hold"\n                            self._blocked_reason = getattr(allocation, "reason", "")\n                        except Exception:\n                            pass',
    'if not can_trade:\n                        try:\n                            # Final pre-exec decision is HOLD (blocked by fees/risk/constraints)\n                            self._desired_action = "hold"\n                            self._executed_action = "hold"\n                            self._blocked_reason = getattr(allocation, "reason", "")\n                        except Exception:\n                            pass'
)

# In the tradeable branch, right before logging SIGNAL, set desired/executed
needle = 'log_info(f"SIGNAL {symbol}: {signal.action.upper()} (Conf: {signal.confidence:.2f})")'
if needle in txt:
    txt = txt.replace(
        needle,
        needle + "\n                        try:\n                            self._desired_action = str(getattr(signal, 'action', '')).lower()\n                            self._executed_action = self._desired_action\n                            self._blocked_reason = ''\n                        except Exception:\n                            pass",
        1,
    )

# 3) If order not placed, executed becomes HOLD but desired remains BUY/SELL
# Change our earlier patch that set signal.action='hold' on order not placed.
# Replace that block with executed_action hold only.
txt = txt.replace(
    'log_warn(f"ORDER NOT PLACED for {symbol} (see earlier warnings).")\n                            try:\n                                # If broker rejected / we failed to place order, final action is HOLD\n                                signal.action = "hold"\n                                self._blocked_reason = "order_not_placed"\n                            except Exception:\n                                pass',
    'log_warn(f"ORDER NOT PLACED for {symbol} (see earlier warnings).")\n                            try:\n                                self._executed_action = "hold"\n                                self._blocked_reason = "order_not_placed"\n                            except Exception:\n                                pass'
)

p.write_text(txt, encoding='utf-8')
print('patched runner.py: desired_action vs executed_action semantics restored (desired!=executed on broker failure)')
