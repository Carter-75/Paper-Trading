from pathlib import Path
import time
p=Path('runner.py')
orig=p.read_text(encoding='utf-8')
ts=time.strftime('%Y%m%d_%H%M%S')
p.with_suffix(p.suffix+f'.bak_state_stamp_{ts}').write_text(orig, encoding='utf-8')

if '"runner_patch":' in orig:
    print('runner_patch already in dashboard_state')
    raise SystemExit(0)

needle='"timestamp": datetime.datetime.now().isoformat(),'
if needle not in orig:
    raise SystemExit('PATCH FAILED: timestamp line not found')

stamp=f'RUNNER_PATCH_{ts}'
new=needle + f"\n                \"runner_patch\": \"{stamp}\","

p.write_text(orig.replace(needle,new,1), encoding='utf-8')
print('patched runner.py: dashboard_state includes runner_patch=',stamp)
