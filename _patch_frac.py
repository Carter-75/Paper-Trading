from pathlib import Path
import re, time

def backup_file(p: Path, tag: str):
    b = p.with_suffix(p.suffix + f".{tag}_{time.strftime('%Y%m%d_%H%M%S')}.bak")
    b.write_text(p.read_text(encoding='utf-8'), encoding='utf-8')
    return b

# ---- allocation_engine.py (fractional via notional) ----
path = Path('risk/allocation_engine.py')
backup_file(path, 'frac')
orig = path.read_text(encoding='utf-8')
txt = orig

# Add target_notional field to dataclass if missing
if 'target_notional' not in txt:
    txt = re.sub(r'(target_value:\s*float\s*\r?\n\s*)(reason:)', r'\1target_notional: float\n    \2', txt, count=1)

# Ensure final AllocationResult includes target_notional=0.0
if 'target_notional=0.0' not in txt:
    txt = re.sub(r'(target_value\s*=\s*qty\s*\*\s*current_price,\s*\r?\n\s*)(reason=)', r'\1target_notional=0.0,\n            \2', txt, count=1)

# Replace qty<1 behavior
pattern = re.compile(r"\n\s*if qty < 1:\s*\n\s*return AllocationResult\(signal\.symbol, 0, 0\.0, \"Calculated qty < 1\", False\)\s*\n", re.MULTILINE)
if not pattern.search(txt):
    raise SystemExit('PATCH FAILED: qty<1 block not found (already modified?)')

txt = pattern.sub("""
        if qty < 1:
            # Fractional support: if we can't afford 1 share, try a notional market buy instead
            min_notional = float(getattr(self.config, 'min_notional_usd', 1.0))
            if alloc_value >= min_notional:
                return AllocationResult(
                    symbol=signal.symbol,
                    target_quantity=0,
                    target_value=0.0,
                    target_notional=float(alloc_value),
                    reason=f\"NOTIONAL ${alloc_value:.2f} | Conf:{signal.confidence:.2f}, Kelly:{self.config.enable_kelly_sizing}, Vol:{signal.regime}\",
                    is_allowed=True,
                    limit_price=None
                )
            return AllocationResult(signal.symbol, 0, 0.0, \"Calculated qty < 1\", False)
""", txt, count=1)

path.write_text(txt, encoding='utf-8')
print('patched allocation_engine.py: notional fractional support')

# ---- orders.py (submit notional market order) ----
path = Path('execution/orders.py')
backup_file(path, 'frac')
orig = path.read_text(encoding='utf-8')
txt = orig

# Inject notional extraction
txt = re.sub(r"qty = allocation\.target_quantity\s*\r?\n\s*limit_price = allocation\.limit_price\s*\r?\n",
             "qty = allocation.target_quantity\n        notional = float(getattr(allocation, 'target_notional', 0.0) or 0.0)\n        limit_price = allocation.limit_price\n",
             txt, count=1)

# Fix guard to allow notional-only allocations
txt = re.sub(r"if not allocation or not allocation\.is_allowed or allocation\.target_quantity <= 0:\s*\r?\n\s*return False",
             "if (not allocation) or (not allocation.is_allowed) or ((allocation.target_quantity <= 0) and (float(getattr(allocation,'target_notional',0.0) or 0.0) <= 0.0)):\n            return False",
             txt, count=1)

# Improve intent log
txt = txt.replace(
    'log_info(f"EXECUTING BUY {symbol}: {qty} shares @ ~${limit_price if limit_price else \'MKT\'}")',
    'log_info(f"EXECUTING BUY {symbol}: {qty} shares / notional=${notional:.2f} @ ~${limit_price if limit_price else \'MKT\'}")'
)

# Replace submit logic in execute_allocation
old_submit = """            if self.config.use_limit_orders and limit_price:
                self._submit_limit_order(symbol, qty, \"buy\", limit_price)
            else:
                self._submit_market_order(symbol, qty, \"buy\")
                
            return True
"""

new_submit = """            if notional > 0:
                # Fractional: notional market order (Alpaca supports notional for market orders)
                self._submit_notional_market_order(symbol, notional, \"buy\")
            elif self.config.use_limit_orders and limit_price:
                self._submit_limit_order(symbol, qty, \"buy\", limit_price)
            else:
                self._submit_market_order(symbol, qty, \"buy\")

            return True
"""

if old_submit not in txt:
    raise SystemExit('PATCH FAILED: orders.py submit block not found')

txt = txt.replace(old_submit, new_submit)

# Add helper if missing
if 'def _submit_notional_market_order' not in txt:
    marker = 'def _submit_market_order'
    idx = txt.find(marker)
    if idx < 0:
        raise SystemExit('PATCH FAILED: could not locate _submit_market_order')

    helper = """

    def _submit_notional_market_order(self, symbol: str, notional: float, side: str):
        \"\"\"Submit a notional (fractional) market order.\"\"\"
        if self._block_if_market_closed('NOTIONAL_MARKET_ORDER', symbol):
            return None

        notional = round(float(notional), 2)
        order = self.api.submit_order(
            symbol=symbol,
            notional=notional,
            side=side,
            type='market',
            time_in_force='day'
        )
        log_info(f\"  [ORDER SENT] {side.upper()} ${notional:.2f} {symbol} @ MARKET (ID: {order.id})\")
        return order
"""

    txt = txt[:idx] + helper + txt[idx:]

path.write_text(txt, encoding='utf-8')
print('patched orders.py: notional market orders supported')
