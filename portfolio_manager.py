#!/usr/bin/env python3
"""
Portfolio Manager - Tracks and manages multiple stock positions.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pytz


class PortfolioManager:
    """Manages multi-stock portfolio with position tracking and rebalancing."""
    
    def __init__(self, portfolio_file: str = "portfolio.json"):
        self.portfolio_file = portfolio_file
        self.positions = {}  # symbol -> {qty, avg_entry, market_value, unrealized_pl, last_update}
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
                self.positions = {}
        else:
            self.positions = {}
    
    def save(self):
        """Save portfolio to file."""
        try:
            data = {
                "positions": self.positions,
                "last_updated": datetime.now(pytz.UTC).isoformat()
            }
            with open(self.portfolio_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save portfolio: {e}")
    
    def update_position(self, symbol: str, qty: float, avg_entry: float, 
                       market_value: float, unrealized_pl: float):
        """Update or add a position."""
        self.positions[symbol] = {
            "qty": qty,
            "avg_entry": avg_entry,
            "market_value": market_value,
            "unrealized_pl": unrealized_pl,
            "last_update": datetime.now(pytz.UTC).isoformat()
        }
        self.save()
    
    def remove_position(self, symbol: str):
        """Remove a position (fully sold)."""
        if symbol in self.positions:
            del self.positions[symbol]
            self.save()
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict]:
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
    
    def has_room_for_new_position(self, max_positions: int) -> bool:
        """Check if we can add a new position."""
        return self.get_position_count() < max_positions
    
    def get_symbols(self) -> List[str]:
        """Get list of symbols currently held."""
        return list(self.positions.keys())

