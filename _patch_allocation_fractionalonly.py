from pathlib import Path
import re, time

p = Path('risk/allocation_engine.py')
orig = p.read_text(encoding='utf-8')
ts = time.strftime("%Y%m%d_%H%M%S")
bak = p.with_suffix(p.suffix + f".bak_fractionalonly_{ts}")
bak.write_text(orig, encoding='utf-8')

txt = orig

# Make max_cap_usd optional cap (only apply if set)
txt = re.sub(
    r"base_alloc\s*=\s*min\(\s*\n\s*total_equity\s*\*\s*self\.config\.trade_size_frac_of_cap,\s*\n\s*self\.config\.max_cap_usd\s*\n\s*\)",
    "base_alloc = (total_equity * self.config.trade_size_frac_of_cap)\n        try:\n            if getattr(self.config, 'max_cap_usd', None):\n                base_alloc = min(base_alloc, float(self.config.max_cap_usd))\n        except Exception:\n            pass",
    txt,
    count=1,
)

# Replace final quantity logic with notional-only
# Find the "Final Quantity" section and replace through the final AllocationResult(...) block
pat = re.compile(r"# Final Quantity[\s\S]*?return AllocationResult\([\s\S]*?\)\s*\n\s*\n", re.MULTILINE)
if not pat.search(txt):
    raise SystemExit('PATCH FAILED: could not find Final Quantity block')

txt = pat.sub(
"""# Final Allocation (FRACTIONAL / NOTIONAL ONLY)
        min_notional = float(getattr(self.config, 'min_notional_usd', 1.0))
        if alloc_value < min_notional:
            return AllocationResult(signal.symbol, 0, 0.0, "Notional below minimum", False)

        return AllocationResult(
            symbol=signal.symbol,
            target_quantity=0,
            target_value=0.0,
            target_notional=float(alloc_value),
            reason=f"NOTIONAL ${alloc_value:.2f} | Conf:{signal.confidence:.2f}, Kelly:{self.config.enable_kelly_sizing}, Vol:{signal.regime}",
            is_allowed=True,
            limit_price=None
        )

""", txt, count=1)

p.write_text(txt, encoding='utf-8')
print('patched allocation_engine.py: fractional-only notional buys + optional max_cap_usd cap')
