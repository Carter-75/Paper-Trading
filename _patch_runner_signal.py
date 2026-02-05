from pathlib import Path
import re, time

p = Path('runner.py')
orig = p.read_text(encoding='utf-8')
ts = time.strftime("%Y%m%d_%H%M%S")
bak = p.with_suffix(p.suffix + f".bak_signal_{ts}")
bak.write_text(orig, encoding='utf-8')

txt = orig

pattern = re.compile(
    r"""\n\s*if signal\.action != \"hold\":\s*\n\s*log_info\(f\"SIGNAL \{symbol\}: \{signal\.action\.upper\(\)\} \(Conf: \{signal\.confidence:[^\}]+\}\)\"\)\s*\n\s*allocation = self\.allocation_engine\.calculate_allocation\(signal, current_price, equity\)\s*\n\s*\n\s*# Always log allocation decision[\s\S]*?log_warn\(f\"ORDER NOT PLACED for \{symbol\} \(see earlier warnings\)\.\"\)\s*\n\s*\n""",
    re.MULTILINE,
)

if not pattern.search(txt):
    raise SystemExit('PATCH FAILED: could not locate existing signal/allocation block')

replacement = """
                if signal.action != "hold":
                    allocation = self.allocation_engine.calculate_allocation(signal, current_price, equity)

                    # Always log allocation decision; prevents misleading BUY logs
                    try:
                        notional = float(getattr(allocation, "target_notional", 0.0) or 0.0)
                        log_info(
                            f"ALLOCATION {symbol}: desired_action={signal.action.upper()} "
                            f"allowed={allocation.is_allowed} qty={allocation.target_quantity} notional=${notional:.2f} "
                            f"value=${allocation.target_value:.2f} reason={allocation.reason}"
                        )
                    except Exception:
                        notional = 0.0

                    can_trade = bool(
                        allocation.is_allowed
                        and ((allocation.target_quantity and allocation.target_quantity > 0) or (notional and notional > 0.0))
                    )

                    if not can_trade:
                        # Treat as HOLD so dashboard/logs don't claim we bought when fees/constraints blocked it
                        signal.action = "hold"
                    else:
                        log_info(f"SIGNAL {symbol}: {signal.action.upper()} (Conf: {signal.confidence:.2f})")
                        placed = self.order_executor.execute_allocation(allocation)
                        if placed:
                            # Update portfolio from Alpaca position if available (avoid assuming instant fills)
                            try:
                                pos = self.api.get_position(symbol)
                                qty = float(pos.qty)
                                avg_entry = float(getattr(pos, 'avg_entry_price', current_price))
                                mv = float(getattr(pos, 'market_value', allocation.target_value))
                                upl = float(getattr(pos, 'unrealized_pl', 0.0))
                                self.pm.update_position(symbol, qty, avg_entry, mv, upl, confidence=signal.confidence)
                            except Exception:
                                # If position isn't visible yet, record intended allocation
                                intended_value = allocation.target_value if allocation.target_value else notional
                                intended_qty = allocation.target_quantity if allocation.target_quantity else (notional / current_price if current_price else 0.0)
                                self.pm.update_position(symbol, intended_qty, current_price, intended_value, 0.0, confidence=signal.confidence)
                        else:
                            log_warn(f"ORDER NOT PLACED for {symbol} (see earlier warnings).")

"""

txt = pattern.sub(replacement, txt, count=1)

p.write_text(txt, encoding='utf-8')
print('patched runner.py: signal logged only when tradeable; supports notional allocations')
