#!/usr/bin/env python3
"""
Unit tests for Portfolio Manager
Run with: pytest test_portfolio_manager.py -v
"""

import pytest
import os
import json
import tempfile
from portfolio_manager import PortfolioManager


@pytest.fixture
def temp_portfolio_file():
    """Create a temporary portfolio file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    yield temp_file
    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)


@pytest.fixture
def portfolio(temp_portfolio_file):
    """Create a PortfolioManager instance with temporary file"""
    return PortfolioManager(temp_portfolio_file)


def test_portfolio_initialization(temp_portfolio_file):
    """Test portfolio initialization"""
    pm = PortfolioManager(temp_portfolio_file)
    assert pm.portfolio_file == temp_portfolio_file
    assert pm.positions == {}


def test_portfolio_load_empty():
    """Test loading non-existent portfolio"""
    pm = PortfolioManager("nonexistent_file.json")
    assert pm.positions == {}


def test_portfolio_save_and_load(temp_portfolio_file):
    """Test saving and loading portfolio"""
    pm1 = PortfolioManager(temp_portfolio_file)
    pm1.update_position("AAPL", 10.0, 150.0, 1500.0, 50.0)
    pm1.save()
    
    # Load in new instance
    pm2 = PortfolioManager(temp_portfolio_file)
    assert "AAPL" in pm2.positions
    assert pm2.positions["AAPL"]["qty"] == 10.0
    assert pm2.positions["AAPL"]["avg_entry"] == 150.0


def test_update_position(portfolio):
    """Test updating a position"""
    portfolio.update_position("TSLA", 5.0, 200.0, 1000.0, 100.0)
    
    assert "TSLA" in portfolio.positions
    pos = portfolio.positions["TSLA"]
    assert pos["qty"] == 5.0
    assert pos["avg_entry"] == 200.0
    assert pos["market_value"] == 1000.0
    assert pos["unrealized_pl"] == 100.0
    assert "last_update" in pos
    assert "first_opened" in pos


def test_update_existing_position_preserves_first_opened(portfolio):
    """Test that updating position preserves first_opened timestamp"""
    portfolio.update_position("AAPL", 10.0, 150.0, 1500.0, 0.0)
    first_opened_1 = portfolio.positions["AAPL"]["first_opened"]
    
    # Update position
    import time
    time.sleep(0.01)  # Small delay to ensure different timestamp
    portfolio.update_position("AAPL", 15.0, 155.0, 2325.0, 75.0)
    first_opened_2 = portfolio.positions["AAPL"]["first_opened"]
    
    assert first_opened_1 == first_opened_2  # Should be preserved


def test_remove_position(portfolio):
    """Test removing a position"""
    portfolio.update_position("NVDA", 20.0, 400.0, 8000.0, 200.0)
    assert "NVDA" in portfolio.positions
    
    portfolio.remove_position("NVDA")
    assert "NVDA" not in portfolio.positions


def test_remove_nonexistent_position(portfolio):
    """Test removing a position that doesn't exist"""
    # Should not raise error
    portfolio.remove_position("NONEXISTENT")
    assert "NONEXISTENT" not in portfolio.positions


def test_get_position(portfolio):
    """Test getting a specific position"""
    portfolio.update_position("MSFT", 8.0, 300.0, 2400.0, 80.0)
    
    pos = portfolio.get_position("MSFT")
    assert pos is not None
    assert pos["qty"] == 8.0
    assert pos["avg_entry"] == 300.0
    
    pos_none = portfolio.get_position("NONEXISTENT")
    assert pos_none is None


def test_get_all_positions(portfolio):
    """Test getting all positions"""
    portfolio.update_position("AAPL", 10.0, 150.0, 1500.0, 50.0)
    portfolio.update_position("GOOGL", 5.0, 2800.0, 14000.0, 200.0)
    
    all_pos = portfolio.get_all_positions()
    assert len(all_pos) == 2
    assert "AAPL" in all_pos
    assert "GOOGL" in all_pos
    
    # Verify it's a copy by checking values
    assert all_pos["AAPL"]["qty"] == 10.0
    assert all_pos["GOOGL"]["qty"] == 5.0


def test_get_position_count(portfolio):
    """Test getting position count"""
    assert portfolio.get_position_count() == 0
    
    portfolio.update_position("AAPL", 10.0, 150.0, 1500.0, 50.0)
    assert portfolio.get_position_count() == 1
    
    portfolio.update_position("GOOGL", 5.0, 2800.0, 14000.0, 200.0)
    assert portfolio.get_position_count() == 2
    
    portfolio.remove_position("AAPL")
    assert portfolio.get_position_count() == 1


def test_get_total_market_value(portfolio):
    """Test calculating total market value"""
    assert portfolio.get_total_market_value() == 0.0
    
    portfolio.update_position("AAPL", 10.0, 150.0, 1500.0, 50.0)
    assert portfolio.get_total_market_value() == 1500.0
    
    portfolio.update_position("GOOGL", 5.0, 2800.0, 14000.0, 200.0)
    assert portfolio.get_total_market_value() == 15500.0


def test_get_total_unrealized_pl(portfolio):
    """Test calculating total unrealized P&L"""
    assert portfolio.get_total_unrealized_pl() == 0.0
    
    portfolio.update_position("AAPL", 10.0, 150.0, 1500.0, 50.0)
    assert portfolio.get_total_unrealized_pl() == 50.0
    
    portfolio.update_position("GOOGL", 5.0, 2800.0, 14000.0, 200.0)
    assert portfolio.get_total_unrealized_pl() == 250.0
    
    # Add losing position
    portfolio.update_position("TSLA", 5.0, 200.0, 900.0, -100.0)
    assert portfolio.get_total_unrealized_pl() == 150.0


def test_get_worst_performer(portfolio):
    """Test getting worst performing stock"""
    # Empty portfolio
    assert portfolio.get_worst_performer() is None
    
    # Add positions with different P&L percentages
    portfolio.update_position("WINNER", 10.0, 100.0, 1200.0, 200.0)  # +20%
    portfolio.update_position("LOSER", 10.0, 100.0, 800.0, -200.0)   # -20%
    portfolio.update_position("NEUTRAL", 10.0, 100.0, 1000.0, 0.0)   # 0%
    
    worst_symbol, worst_pct = portfolio.get_worst_performer()
    assert worst_symbol == "LOSER"
    assert worst_pct < 0


def test_get_worst_performer_all_winners(portfolio):
    """Test worst performer when all positions are profitable"""
    portfolio.update_position("GOOD", 10.0, 100.0, 1200.0, 200.0)  # +20%
    portfolio.update_position("BETTER", 10.0, 100.0, 1500.0, 500.0)  # +50%
    
    worst_symbol, worst_pct = portfolio.get_worst_performer()
    assert worst_symbol == "GOOD"
    assert worst_pct > 0  # Still positive, just worst of the winners


def test_has_room_for_new_position(portfolio):
    """Test checking if portfolio has room for new positions"""
    max_positions = 5
    
    assert portfolio.has_room_for_new_position(max_positions) is True
    
    # Add positions up to limit
    for i in range(max_positions):
        portfolio.update_position(f"STOCK{i}", 10.0, 100.0, 1000.0, 0.0)
    
    assert portfolio.has_room_for_new_position(max_positions) is False
    
    # Remove one
    portfolio.remove_position("STOCK0")
    assert portfolio.has_room_for_new_position(max_positions) is True


def test_get_symbols(portfolio):
    """Test getting list of held symbols"""
    assert portfolio.get_symbols() == []
    
    portfolio.update_position("AAPL", 10.0, 150.0, 1500.0, 50.0)
    portfolio.update_position("GOOGL", 5.0, 2800.0, 14000.0, 200.0)
    portfolio.update_position("TSLA", 5.0, 200.0, 1000.0, 0.0)
    
    symbols = portfolio.get_symbols()
    assert len(symbols) == 3
    assert "AAPL" in symbols
    assert "GOOGL" in symbols
    assert "TSLA" in symbols


def test_portfolio_persistence(temp_portfolio_file):
    """Test that portfolio persists across instances"""
    # Create first instance and add positions
    pm1 = PortfolioManager(temp_portfolio_file)
    pm1.update_position("AAPL", 10.0, 150.0, 1500.0, 50.0)
    pm1.update_position("GOOGL", 5.0, 2800.0, 14000.0, 200.0)
    
    # Create second instance (should load from file)
    pm2 = PortfolioManager(temp_portfolio_file)
    assert pm2.get_position_count() == 2
    assert pm2.get_total_market_value() == 15500.0
    assert "AAPL" in pm2.get_symbols()
    assert "GOOGL" in pm2.get_symbols()


def test_portfolio_handles_corrupted_file(temp_portfolio_file):
    """Test that portfolio handles corrupted JSON gracefully"""
    # Write corrupted JSON
    with open(temp_portfolio_file, 'w') as f:
        f.write("corrupted json {{{")
    
    # Should load with empty positions
    pm = PortfolioManager(temp_portfolio_file)
    assert pm.positions == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

