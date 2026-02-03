#!/usr/bin/env python3
"""
Portfolio Manager - Tracks and manages multiple stock positions.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pytz


class PortfolioManager:
    """Manages multi-stock portfolio with position tracking and rebalancing."""
    
    def __init__(self, portfolio_file: str = "portfolio.json"):
        self.portfolio_file = portfolio_file
        self.history_file = "trade_history.json"
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> {qty, avg_entry, market_value, unrealized_pl, last_update}
        self.history: List[Dict[str, Any]] = []
        self.load()
    
    def load(self):
        """Load portfolio from file."""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.positions = data.get("positions", {})
            except Exception as e:
                print(f"Warning: Could not load portfolio: {e}")
                try:
                    import traceback
                    print(traceback.format_exc())
                except Exception:
                    pass
                self.positions = {}
        else:
            self.positions = {}
            
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []
        else:
            self.history = []
    
    def save(self):
        """Save portfolio to file."""
        try:
            data = {
                "positions": self.positions,
                "last_updated": datetime.now(pytz.UTC).isoformat()
            }
            with open(self.portfolio_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                
            # Save history
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save portfolio: {e}")
            try:
                import traceback
                print(traceback.format_exc())
            except Exception:
                pass
    
    def update_position(self, symbol: str, qty: float, avg_entry: float, 
                       market_value: float, unrealized_pl: float, 
                       confidence: float = 0.0, expected_return: float = 0.0):
        """Update or add a position."""
        # Preserve first_opened if position already exists
        first_opened = self.positions.get(symbol, {}).get("first_opened")
        if first_opened is None:
            first_opened = datetime.now(pytz.UTC).isoformat()
        
        self.positions[symbol] = {
            "qty": qty,
            "avg_entry": avg_entry,
            "market_value": market_value,
            "unrealized_pl": unrealized_pl,
            "last_update": datetime.now(pytz.UTC).isoformat(),
            "first_opened": first_opened,
            "confidence": confidence,
            "expected_return": expected_return
        }
        self.save()
    
    def remove_position(self, symbol: str):
        """Remove a position (fully sold)."""
        if symbol in self.positions:
            del self.positions[symbol]
            del self.positions[symbol]
            self.save()

    def log_closed_trade(self, symbol: str, avg_entry: float, exit_price: float, qty: int, reason: str):
        """Log a closed trade to history."""
        pnl = (exit_price - avg_entry) * qty
        pnl_pct = (exit_price - avg_entry) / avg_entry if avg_entry > 0 else 0
        
        trade_record = {
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "symbol": symbol,
            "action": "SELL", # We mostly log sells for PnL
            "qty": qty,
            "entry_price": avg_entry,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason
        }
        
        self.history.append(trade_record)
        
        # Enforce Memory Limit (User Request: "Not too long")
        # Keep last 500 trades
        if len(self.history) > 500:
            self.history = self.history[-500:]
            
        self.save()

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all positions."""
        return self.positions.copy()
    
    def get_position_count(self) -> int:
        """Get number of positions held."""
        return len(self.positions)
    
    def get_total_market_value(self) -> float:
        """Get total portfolio value."""
        return sum(pos.get("market_value", 0.0) for pos in self.positions.values())
    
    def get_total_unrealized_pl(self) -> float:
        """Get total unrealized P&L."""
        return sum(pos.get("unrealized_pl", 0.0) for pos in self.positions.values())
    
    def get_worst_performer(self) -> Optional[Tuple[str, float]]:
        """Get worst performing stock by P&L percentage."""
        if not self.positions:
            return None
        
        worst_symbol = None
        worst_pct = float('inf')
        
        for symbol, pos in self.positions.items():
            cost_basis = pos.get("market_value", 0) - pos.get("unrealized_pl", 0)
            if cost_basis > 0:
                pnl_pct = pos.get("unrealized_pl", 0) / cost_basis
                if pnl_pct < worst_pct:
                    worst_pct = pnl_pct
                    worst_symbol = symbol
        
        return (worst_symbol, worst_pct) if worst_symbol else None
    
    def get_lowest_confidence_position(self) -> Optional[Dict[str, Any]]:
        """Get the position with the lowest confidence score."""
        if not self.positions:
            return None
            
        worst_pos = None
        min_conf = float('inf')
        worst_sym = None
        
        for sym, data in self.positions.items():
            conf = data.get("confidence", 0.5) # Default to 0.5 if missing
            if conf < min_conf:
                min_conf = conf
                worst_pos = data
                worst_sym = sym
                
        if worst_pos:
            # Return copy with symbol included
            result = worst_pos.copy()
            result['symbol'] = worst_sym
            return result
        return None
    
    def has_room_for_new_position(self, max_positions: int) -> bool:
        """Check if we can add a new position."""
        return self.get_position_count() < max_positions
    
    def get_symbols(self) -> List[str]:
        """Get list of symbols currently held."""
        return list(self.positions.keys())

