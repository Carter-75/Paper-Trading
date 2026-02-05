from pathlib import Path
import re, time

p = Path('runner.py')
orig = p.read_text(encoding='utf-8')
ts = time.strftime("%Y%m%d_%H%M%S")
bak = p.with_suffix(p.suffix + f".bak_desiredexec_{ts}")
bak.write_text(orig, encoding='utf-8')

txt = orig

# Inject tracking right after we compute allocation in the patched block
# We look for the line: allocation = self.allocation_engine.calculate_allocation(signal, current_price, equity)
needle = "allocation = self.allocation_engine.calculate_allocation(signal, current_price, equity)"
idx = txt.find(needle)
if idx < 0:
    raise SystemExit('PATCH FAILED: allocation line not found')

# Only patch once
if 'self._desired_action' not in txt:
    txt = txt.replace(
        needle,
        needle + "\n\n                    # Track desired vs executed action for dashboard/UI\n                    try:\n                        self._desired_action = str(getattr(signal, 'action', '')).lower()\n                    except Exception:\n                        self._desired_action = ''\n                    self._executed_action = ''\n                    self._blocked_reason = ''\n"
    , 1)

# After can_trade calculation, we set executed/blocked
# Find 'if not can_trade:' inside the block
if 'if not can_trade:' not in txt:
    raise SystemExit('PATCH FAILED: can_trade block not found')

# Patch the first occurrence after our replacement block
txt = txt.replace(
    'if not can_trade:',
    'if not can_trade:\n                        try:\n                            self._executed_action = "hold"\n                            self._blocked_reason = getattr(allocation, "reason", "")\n                        except Exception:\n                            pass',
    1
)

# Patch the tradeable branch to set executed action
txt = txt.replace(
    'log_info(f"SIGNAL {symbol}: {signal.action.upper()} (Conf: {signal.confidence:.2f})")',
    'log_info(f"SIGNAL {symbol}: {signal.action.upper()} (Conf: {signal.confidence:.2f})")\n                        try:\n                            self._executed_action = str(getattr(signal, "action", "")).lower()\n                        except Exception:\n                            pass',
    1
)

p.write_text(txt, encoding='utf-8')
print('patched runner.py: track desired/executed/blocked_reason for dashboard')
