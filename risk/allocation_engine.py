
import logging
from typing import Dict, Optional, Any
import math
from dataclasses import dataclass

# Import config and models
try:
    from config_validated import get_config
    from strategies.decision_engine import TradeSignal
except ImportError:
    import sys
    sys.path.append("..")
    from config_validated import get_config
    from strategies.decision_engine import TradeSignal

@dataclass
class AllocationResult:
    symbol: str
    target_quantity: int
    target_value: float
    target_notional: float
    reason: str
    is_allowed: bool
    limit_price: Optional[float] = None

class AllocationEngine:
    """
    The Banker of the trading bot.
    Decides HOW MUCH to buy/sell based on:
    1. Confidence (from DecisionEngine)
    2. Volatility (Risk)
    3. Kelly Criterion (Optimal Sizing)
    4. Portfolio Constraints (Max loss, Max exposure)
    """
    
    def __init__(self, portfolio_manager):
        self.config = get_config()
        self.pm = portfolio_manager
        self.logger = logging.getLogger("AllocationEngine")
        
        # Win Rate Statistics (for Kelly)
        # TODO: Load this from a persistent stats file
        self.win_rate = 0.55 # Conservative initial estimate
        self.avg_win_loss_ratio = 1.5
    
    def calculate_allocation(self, signal: TradeSignal, current_price: float, total_equity: float, max_exposure_cap: Optional[float] = None) -> AllocationResult:
        """
        Calculate the optimal position size.
        """
        if signal.action == "hold":
             return AllocationResult(signal.symbol, 0, 0.0, 0.0, "Hold signal", False)

        if signal.action == "sell":
            # Liquidation logic
            # For now, simplistic full sell. Smart logic could scale out.
            pos = self.pm.get_position(signal.symbol)
            qty = pos['qty'] if pos else 0
            return AllocationResult(signal.symbol, 0, 0.0, 0.0, "Sell Signal", True)

        # --- BUY ALLOCATION LOGIC ---
        
        # 1. Base Capital Allocation
        # Start with a fixed fraction of equity (e.g., 20%)
        # But clamp it to MAX_CAP_USD
        base_alloc = (total_equity * self.config.trade_size_frac_of_cap)
        try:
            if getattr(self.config, 'max_cap_usd', None):
                # If restricted mode is active (max_exposure_cap passed), ignore the fixed floor for sizing
                # The fixed floor is now a Hard Kill level, not a position sizer.
                # However, we keep this logic if no cap is passed to respect original intent if any.
                if max_exposure_cap is None:
                     base_alloc = min(base_alloc, float(self.config.max_cap_usd))
        except Exception:
            pass
        
        # 2. Confidence Scaling (Dynamic Sizing)
        # Scaled Logic:
        # Conf 0.5 -> 1.0x Base (Normal trade)
        # Conf 1.0 -> 2.0x Base (High conviction)
        # Conf 0.0 -> 0.0x Base (No trade)
        
        # We want "Base" to be the average trade size.
        # If signal.confidence is usually 0.5-0.8.
        # dynamic_factor = 1 + (signal.confidence - 0.5) * 2
        # Example: Conf 0.8 -> 1 + (0.3)*2 = 1.6x Base.
        
        dynamic_factor = 1.0 + (max(0.0, signal.confidence - 0.5) * 2.0)
        
        # User requested aggressive dynamic sizing.
        # If confidence is exceptionally high (>0.8), boost further?
        # Let's stick to the 2x cap for safety unless Config says otherwise.
        
        alloc_value = base_alloc * dynamic_factor
        
        # 3. Volatility Scaling (Risk Parity-lite)
        # If stock is very volatile, reduce size.
        if signal.regime == "high_volatility":
            alloc_value *= 0.6 # Reduce by 40% in high vol
            
        # 4. Kelly Criterion (Optional)
        if self.config.enable_kelly_sizing:
            kelly_pct = self._calculate_kelly_fraction()
            if self.config.kelly_use_half:
                kelly_pct *= 0.5 # Half-Kelly is industry standard for safety
            
            kelly_alloc = total_equity * kelly_pct
            # Blend Kelly with Base (don't let Kelly go too wild)
            alloc_value = min(alloc_value, kelly_alloc)
            
        # 5. Global Risk Constraints
        
        # 5.1 Max Exposure Check
        current_exposure = self.pm.get_total_market_value()
        if (current_exposure + alloc_value) > (total_equity * (self.config.max_exposure_pct / 100.0)):
             reduction = (total_equity * (self.config.max_exposure_pct / 100.0)) - current_exposure
             if reduction < 0: reduction = 0
             if reduction < 0: reduction = 0
             alloc_value = min(alloc_value, reduction)
             
        # 5.1.5 Restricted Mode Cap (Soft Kill Recovery)
        if max_exposure_cap is not None:
            # We must not exceed this total portfolio value
            current_total_val = self.pm.get_total_market_value()
            remaining_cap = max(0.0, max_exposure_cap - current_total_val)
            
            if alloc_value > remaining_cap:
                alloc_value = remaining_cap
                # If we are already over, alloc becomes 0
            
            if alloc_value <= 0:
                 return AllocationResult(signal.symbol, 0, 0.0, 0.0, f"Restricted Mode: Max Exposure Cap ${max_exposure_cap:.2f} reached", False)
             
        # 5.2 Max Loss per Trade (Stop Loss Risk)
        # Risk = Value * StopLoss%
        # We want Risk < Equity * MaxRisk%
        max_risk_dollars = total_equity * (self.config.max_loss_per_trade_pct / 100.0)
        
        # [MOD] Live mode optimization: Use ATR for risk if available, instead of estimating with stop_loss_percent
        sl_pct = self.config.stop_loss_percent
        if self.config.wants_live_mode() and getattr(signal, 'atr', 0) > 0 and current_price > 0:
            # implied_sl_pct = (2 * atr / price) * 100
            sl_pct = (2.0 * float(signal.atr) / current_price) * 100.0
            self.logger.info(f"Using ATR-based stop loss risk: {sl_pct:.2f}% for {signal.symbol}")

        implied_risk_dollars = alloc_value * (sl_pct / 100.0)
        
        if implied_risk_dollars > max_risk_dollars:
            # Scale down to fit risk budget
            scaler = max_risk_dollars / implied_risk_dollars
            alloc_value *= scaler
        
        # 6. Fee & Slippage Guard (Net Profit Check)
        if self.config.simulate_fees_enabled:
            # Estimate costs
            fee = self.config.fee_per_trade_usd
            slippage_cost = alloc_value * (self.config.slippage_percent / 100.0) if self.config.simulate_slippage_enabled else 0
            total_cost = fee + slippage_cost
            
            # [MOD] Live mode optimization: Use ATR as proxy for expected move instead of hardcoded 0.5%
            # This ensures we don't 'double check' ourselves into rejection when live stop-losses are also ATR-based.
            expected_move_pct = 0.005 # 0.5% fallback
            if self.config.wants_live_mode() and getattr(signal, 'atr', 0) > 0 and current_price > 0:
                expected_move_pct = float(signal.atr) / current_price
            
            expected_gross_profit = alloc_value * expected_move_pct
            
            if expected_gross_profit < (total_cost * 1.5):
                return AllocationResult(signal.symbol, 0, 0.0, 0.0, f"Fees too high ({total_cost:.2f} > profit {expected_gross_profit:.2f}, move {expected_move_pct:.2%})", False)

        # Final Allocation (FRACTIONAL / NOTIONAL ONLY)
        min_notional = float(getattr(self.config, 'min_notional_usd', 1.0))
        if alloc_value < min_notional:
            return AllocationResult(signal.symbol, 0, 0.0, 0.0, "Notional below minimum", False)

        return AllocationResult(
            symbol=signal.symbol,
            target_quantity=0,
            target_value=0.0,
            target_notional=float(alloc_value),
            reason=f"NOTIONAL ${alloc_value:.2f} | Conf:{signal.confidence:.2f}, Kelly:{self.config.enable_kelly_sizing}, Vol:{signal.regime}",
            is_allowed=True,
            limit_price=None
        )

    def _calculate_kelly_fraction(self) -> float:
        """
        Kelly % = W - (1-W)/R
        W = Win Probability
        R = Win/Loss Ratio
        """
        W = self.win_rate
        R = self.avg_win_loss_ratio
        if R == 0: return 0.0

        k = W - ((1 - W) / R)
        return max(0.0, k)  # Never return negative allocation

    # ------------------------------------------------------------------
    # TSLA 80% Floor Enforcement
    # ------------------------------------------------------------------

    TSLA_FLOOR_PCT: float = 0.80   # 80% of total portfolio value
    DUST_THRESHOLD_USD: float = 50.0  # Positions smaller than this are fully liquidated

    def enforce_tsla_floor(
        self,
        tsla_price: float,
        equity: float,
        order_executor,
        api,
    ) -> None:
        """
        Enforce the rule that TSLA must represent at least 80% of total
        portfolio value.  Called at the start of every trade cycle.

        Steps:
          1. Compute current TSLA value and target.
          2. If already at/above floor → return immediately.
          3. Calculate deficit (how much TSLA $ we need to buy).
          4. Cover the deficit with available cash first.
          5. If still short, sell non-TSLA positions (worst-performers
             first; locked positions last, overriding locks only if required).
          6. Submit the TSLA buy for the full deficit.
        """
        try:
            from utils.helpers import log_info, log_warn, log_error
        except ImportError:
            import logging
            log_info = log_warn = log_error = logging.getLogger("AllocationEngine").info

        if equity <= 0 or tsla_price <= 0:
            return

        # ── 1. Current TSLA value ────────────────────────────────────────
        tsla_pos = self.pm.get_position("TSLA")
        tsla_value = float((tsla_pos or {}).get("market_value", 0.0))

        target_tsla_value = equity * self.TSLA_FLOOR_PCT

        # ── 2. Floor already met ─────────────────────────────────────────
        if tsla_value >= target_tsla_value:
            return

        deficit = target_tsla_value - tsla_value
        log_warn(
            f"TSLA FLOOR: TSLA=${tsla_value:.2f} target=${target_tsla_value:.2f} "
            f"(equity=${equity:.2f}) — deficit=${deficit:.2f}"
        )

        # ── 3. Cover with available cash first ───────────────────────────
        total_invested = self.pm.get_total_market_value()
        available_cash = max(0.0, equity - total_invested)
        still_needed = max(0.0, deficit - available_cash)

        # ── 4. Sell non-TSLA positions to fund the remainder ─────────────
        if still_needed > 0:
            # Build sorted candidate list
            all_positions = self.pm.get_all_positions()
            candidates = []
            for sym, pos_data in all_positions.items():
                if sym == "TSLA":
                    continue
                mv = float(pos_data.get("market_value", 0.0))
                if mv <= 0:
                    continue
                cost_basis = mv - float(pos_data.get("unrealized_pl", 0.0))
                pnl_pct = (float(pos_data.get("unrealized_pl", 0.0)) / cost_basis
                           if cost_basis > 0 else 0.0)
                locked = bool(pos_data.get("is_locked", False))
                candidates.append({
                    "symbol": sym,
                    "market_value": mv,
                    "pnl_pct": pnl_pct,
                    "is_locked": locked,
                })

            # Sort: unlocked first, then worst PnL, then smallest MV
            candidates.sort(key=lambda c: (
                1 if c["is_locked"] else 0,   # unlocked first
                c["pnl_pct"],                  # worst performer first
                c["market_value"],             # smallest first on tie
            ))

            for cand in candidates:
                if still_needed <= 0:
                    break

                sym = cand["symbol"]
                mv = cand["market_value"]
                sell_amount = min(mv, still_needed)

                # Full liquidation if remainder would be dust
                remainder = mv - sell_amount
                full_sell = remainder < self.DUST_THRESHOLD_USD

                if full_sell:
                    log_info(
                        f"TSLA FLOOR: Liquidating {sym} (mv=${mv:.2f}) "
                        f"{'[LOCK OVERRIDE]' if cand['is_locked'] else ''}"
                    )
                    if order_executor.liquidate(sym, "TSLA 80% floor rebalance"):
                        self.pm.remove_position(sym)
                        still_needed -= mv
                else:
                    log_info(
                        f"TSLA FLOOR: Partial sell ${sell_amount:.2f} of {sym} "
                        f"{'[LOCK OVERRIDE]' if cand['is_locked'] else ''}"
                    )
                    try:
                        order_executor._submit_notional_market_order(sym, sell_amount, "sell")
                        still_needed -= sell_amount
                        # PM will be reconciled on next sync_portfolio_with_alpaca
                    except Exception as e:
                        log_error(f"TSLA FLOOR: Partial sell of {sym} failed: {e}")

            if still_needed > 0:
                log_warn(
                    f"TSLA FLOOR: Could not fully cover deficit — "
                    f"${still_needed:.2f} still unmet after selling candidates."
                )

        # ── 5. Buy TSLA for the full deficit ─────────────────────────────
        try:
            log_info(f"TSLA FLOOR: Buying ${deficit:.2f} TSLA @ ~${tsla_price:.2f}")
            order_executor._submit_notional_market_order("TSLA", deficit, "buy")
        except Exception as e:
            log_error(f"TSLA FLOOR: TSLA buy order failed: {e}")
