from pathlib import Path
import time

ts=time.strftime('%Y%m%d_%H%M%S')

# patch runner.py dashboard_state
rp=Path('runner.py')
orig=rp.read_text(encoding='utf-8')
rp.with_suffix(rp.suffix+f'.bak_dashlimit_{ts}').write_text(orig, encoding='utf-8')

needle='"virtual_account_size": self.config.virtual_account_size,'
if needle in orig and 'kill_switch_floor_usd' not in orig:
    orig=orig.replace(needle, needle+'\n                "kill_switch_floor_usd": float(getattr(self.config, "max_cap_usd", 0.0) or 0.0),')

rp.write_text(orig, encoding='utf-8')
print('patched runner.py: add kill_switch_floor_usd to dashboard_state')

# patch dashboard.html label
hp=Path('templates/dashboard.html')
h=hp.read_text(encoding='utf-8')
hp.with_suffix(hp.suffix+f'.bak_{ts}').write_text(h, encoding='utf-8')

old="document.getElementById('max-capital').textContent = 'Limit: $' + (data.virtual_account_size || \"---\");"
new="document.getElementById('max-capital').textContent = 'Kill floor: $' + ((data.kill_switch_floor_usd !== undefined) ? Number(data.kill_switch_floor_usd).toFixed(2) : '---');"

if old in h:
    h=h.replace(old,new)
else:
    print('WARN: did not find expected Limit line; manual edit may be needed')

hp.write_text(h, encoding='utf-8')
print('patched dashboard.html: show kill floor instead of virtual_account_size')
