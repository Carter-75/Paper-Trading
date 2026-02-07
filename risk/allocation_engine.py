
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
             return AllocationResult(signal.symbol, 0, 0.0, "Hold signal", False)

        if signal.action == "sell":
            # Liquidation logic
            # For now, simplistic full sell. Smart logic could scale out.
            pos = self.pm.get_position(signal.symbol)
            qty = pos['qty'] if pos else 0
            return AllocationResult(signal.symbol, 0, 0.0, "Sell Signal", True)

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
        
        # 2. Confidence Scaling
        # If confidence is high (e.g. 0.9), use full base.
        # If confidence is low (e.g. 0.6), scale down.
        # Linear scaling: 0.5 conf -> 0.5 alloc
        confidence_mult = max(0.5, signal.confidence) # Floor at 0.5
        alloc_value = base_alloc * confidence_mult
        
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
                 return AllocationResult(signal.symbol, 0, 0.0, f"Restricted Mode: Max Exposure Cap ${max_exposure_cap:.2f} reached", False)
             
        # 5.2 Max Loss per Trade (Stop Loss Risk)
        # Risk = Value * StopLoss%
        # We want Risk < Equity * MaxRisk%
        max_risk_dollars = total_equity * (self.config.max_loss_per_trade_pct / 100.0)
        implied_risk_dollars = alloc_value * (self.config.stop_loss_percent / 100.0)
        
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
            
            # Simple heuristic: Expected profit should safely cover costs (e.g. 2x costs)
            # If expected movement is ~1% (typical daily vol), profit = value * 0.01
            # We assume a conservative 0.5% move for this check
            expected_gross_profit = alloc_value * 0.005 
            
            if expected_gross_profit < (total_cost * 1.5):
                return AllocationResult(signal.symbol, 0, 0.0, f"Fees too high ({total_cost:.2f} > profit {expected_gross_profit:.2f})", False)

        # Final Allocation (FRACTIONAL / NOTIONAL ONLY)
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
        return max(0.0, k) # Never return negative allocation
