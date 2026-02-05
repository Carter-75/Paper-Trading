from pathlib import Path
import time

ts=time.strftime('%Y%m%d_%H%M%S')

p=Path('config_validated.py')
orig=p.read_text(encoding='utf-8')
p.with_suffix(p.suffix+f'.bak_no_maxcap_alias_{ts}').write_text(orig, encoding='utf-8')

# Remove MAX_CAP_USD alias, keep only KILL_SWITCH_EQUITY_FLOOR_USD
orig = orig.replace(
    'validation_alias=AliasChoices("KILL_SWITCH_EQUITY_FLOOR_USD", "MAX_CAP_USD"),',
    'validation_alias="KILL_SWITCH_EQUITY_FLOOR_USD",'
)

p.write_text(orig, encoding='utf-8')
print('patched config_validated.py: removed MAX_CAP_USD alias; only KILL_SWITCH_EQUITY_FLOOR_USD supported')
