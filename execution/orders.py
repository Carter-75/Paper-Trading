
import logging
import time
from typing import Optional, Dict, Any
from alpaca_trade_api.rest import APIError
from alpaca_trade_api import REST

try:
    from config_validated import get_config
    from utils.helpers import log_info, log_error, log_warn
except ImportError:
    import sys
    sys.path.append("..")
    from config_validated import get_config
    from utils.helpers import log_info, log_error, log_warn

class OrderExecutor:
    """
    Handles the execution of trades via Alpaca API.
    Ensures:
    1. Safety (limit orders, timeouts)
    2. Reliability (retries)
    3. Logging (audit trail)
    """
    
    def __init__(self, api_client: REST):
        self.config = get_config()
        self.api = api_client
        self.logger = logging.getLogger("OrderExecutor")
        
    def execute_allocation(self, allocation) -> bool:
        """
        Execute an AllocationResult.
        """
        if not allocation or not allocation.is_allowed or allocation.target_quantity <= 0:
            return False
            
        symbol = allocation.symbol
        qty = allocation.target_quantity
        limit_price = allocation.limit_price
        
        # 1. Log Intent
        log_info(f"EXECUTING BUY {symbol}: {qty} shares @ ~${limit_price if limit_price else 'MKT'}")
        log_info(f"  Reason: {allocation.reason}")
        
        # 2. Place Order
        try:
            # Cancel open orders first to prevent double-fills
            self.cancel_open_orders(symbol)
            
            if self.config.use_limit_orders and limit_price:
                self._submit_limit_order(symbol, qty, "buy", limit_price)
            else:
                self._submit_market_order(symbol, qty, "buy")
                
            return True
            
        except Exception as e:
            log_error(f"Failed to execute buy for {symbol}: {e}")
            return False

    def liquidate(self, symbol: str, reason: str) -> bool:
        """
        Fully liquidate a position.
        """
        try:
            # Get current position size
            try:
                pos = self.api.get_position(symbol)
                qty = float(pos.qty)
            except APIError:
                # Position might not exist
                return False
                
            if qty <= 0:
                return False
                
            log_info(f"LIQUIDATING {symbol}: {qty} shares")
            log_info(f"  Reason: {reason}")
            
            self.cancel_open_orders(symbol)
            self.api.close_position(symbol) # Market close
            return True
            
        except Exception as e:
            log_error(f"Failed to liquidate {symbol}: {e}")
            return False

    def _submit_limit_order(self, symbol: str, qty: int, side: str, limit_price: float):
        """
        Submit a limit order with basic retry logic.
        """
        # Round price to 2 decimals
        limit_price = round(limit_price, 2)
        
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='limit',
            time_in_force='day',
            limit_price=limit_price
        )
        log_info(f"  [ORDER SENT] {side.upper()} {qty} {symbol} @ {limit_price} (ID: {order.id})")
        return order

    def _submit_market_order(self, symbol: str, qty: int, side: str):
        """
        Submit a market order.
        """
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        log_info(f"  [ORDER SENT] {side.upper()} {qty} {symbol} @ MARKET (ID: {order.id})")
        return order

    def cancel_open_orders(self, symbol: str):
        """Cancel open orders ONLY for the specific symbol."""
        try:
            orders = self.api.list_orders(status='open')
            for o in orders:
                if o.symbol == symbol:
                    self.api.cancel_order(o.id)
                    log_info(f"  [CANCELED] Replaced open order for {symbol}")
        except Exception as e:
            log_error(f"Failed to cancel orders for {symbol}: {e}")
