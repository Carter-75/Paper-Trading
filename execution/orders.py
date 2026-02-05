
import logging
import time
import datetime
import pytz
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
        
    def _market_is_open(self) -> bool:
        """Best-effort market open check.

        Uses Alpaca clock when available; falls back to 9:30-16:00 ET weekdays.
        """
        try:
            clock = self.api.get_clock()
            return bool(getattr(clock, 'is_open', False))
        except Exception:
            pass

        try:
            tz = pytz.timezone('US/Eastern')
            now_et = datetime.datetime.now(tz)
            if now_et.weekday() >= 5:
                return False
            open_min = 9 * 60 + 30
            close_min = 16 * 60
            now_min = now_et.hour * 60 + now_et.minute
            return open_min <= now_min < close_min
        except Exception:
            return False

    def _block_if_market_closed(self, action: str, symbol: str) -> bool:
        """Return True if caller should skip because market is closed."""
        try:
            if getattr(self.config, 'enable_market_hours_only', True) and not self._market_is_open():
                log_warn(f"Market closed - skipping {action} for {symbol}.")
                return True
        except Exception:
            return False
        return False

    def execute_allocation(self, allocation) -> bool:
        """
        Execute an AllocationResult.
        """
        if (not allocation) or (not allocation.is_allowed) or ((allocation.target_quantity <= 0) and (float(getattr(allocation,'target_notional',0.0) or 0.0) <= 0.0)):
            return False
            
        symbol = allocation.symbol
        qty = allocation.target_quantity
        notional = float(getattr(allocation, 'target_notional', 0.0) or 0.0)
        limit_price = allocation.limit_price
        if self._block_if_market_closed('BUY', symbol):
            return False

        # 1. Log Intent
        log_info(f"EXECUTING BUY {symbol}: {qty} shares / notional=${notional:.2f} @ ~${limit_price if limit_price else 'MKT'}")
        log_info(f"  Reason: {allocation.reason}")
        
        # 2. Place Order
        try:
            # Cancel open orders first to prevent double-fills
            self.cancel_open_orders(symbol)
            
            if notional > 0:
                # Fractional: notional market order (Alpaca supports notional for market orders)
                self._submit_notional_market_order(symbol, notional, "buy")
            elif self.config.use_limit_orders and limit_price:
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
                
            if self._block_if_market_closed('LIQUIDATE', symbol):
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
        
        if self._block_if_market_closed('LIMIT_ORDER', symbol):
            return None

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

    

    def _submit_notional_market_order(self, symbol: str, notional: float, side: str):
        """Submit a notional (fractional) market order."""
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
        log_info(f"  [ORDER SENT] {side.upper()} ${notional:.2f} {symbol} @ MARKET (ID: {order.id})")
        return order
    def _submit_market_order(self, symbol: str, qty: int, side: str):
        """
        Submit a market order.
        """
        if self._block_if_market_closed('MARKET_ORDER', symbol):
            return None
    
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
