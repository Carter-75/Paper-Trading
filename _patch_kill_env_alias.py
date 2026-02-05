from pathlib import Path
import time

ts=time.strftime("%Y%m%d_%H%M%S")

p=Path('config_validated.py')
orig=p.read_text(encoding='utf-8')
p.with_suffix(p.suffix+f'.bak_alias_{ts}').write_text(orig, encoding='utf-8')

txt=orig

# Ensure AliasChoices import
if 'AliasChoices' not in txt:
    txt = txt.replace('from pydantic import Field, field_validator, model_validator',
                      'from pydantic import Field, field_validator, model_validator, AliasChoices')

# Replace the max_cap_usd Field line to use a proper env alias
old = 'max_cap_usd: float = Field(default=100.0, ge=0.0, description="KILL SWITCH: if account equity drops below this USD, stop bot (0 disables)")'
new = 'max_cap_usd: float = Field(default=100.0, ge=0.0, validation_alias=AliasChoices("KILL_SWITCH_EQUITY_FLOOR_USD", "MAX_CAP_USD"), description="KILL SWITCH: if account equity drops below this USD, stop bot (0 disables)")'
if old in txt:
    txt = txt.replace(old, new)
else:
    # fallback: patch by locating the field start
    raise SystemExit('PATCH FAILED: expected max_cap_usd line not found')

p.write_text(txt, encoding='utf-8')
print('patched config_validated.py: kill switch uses env KILL_SWITCH_EQUITY_FLOOR_USD (MAX_CAP_USD still accepted)')

# Update .env: replace MAX_CAP_USD with KILL_SWITCH_EQUITY_FLOOR_USD
envp=Path('.env')
env=envp.read_text(encoding='utf-8')
envp.with_suffix(envp.suffix+f'.bak_killenv_{ts}').write_text(env, encoding='utf-8')
lines=[]
for line in env.splitlines():
    if line.strip().lower().startswith('max_cap_usd='):
        continue
    lines.append(line)
# Append kill switch floor at end (keep it explicit)
lines.append('KILL_SWITCH_EQUITY_FLOOR_USD=100')
envp.write_text('\n'.join(lines)+"\n", encoding='utf-8')
print('updated .env: removed MAX_CAP_USD, added KILL_SWITCH_EQUITY_FLOOR_USD=100')
