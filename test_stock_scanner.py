#!/usr/bin/env python3
"""
Unit tests for Stock Scanner
Run with: pytest test_stock_scanner.py -v
"""

import pytest
from stock_scanner import (
    get_stock_universe,
    FALLBACK_STOCK_UNIVERSE,
    DEFAULT_TOP_100_STOCKS
)


def test_get_stock_universe_with_user_symbols():
    """Test getting stock universe with user-provided symbols"""
    user_symbols = ["AAPL", "msft", "googl"]
    result = get_stock_universe(user_symbols=user_symbols, use_top_100=False)
    
    assert len(result) == 3
    assert "AAPL" in result
    assert "MSFT" in result  # Should be uppercased
    assert "GOOGL" in result


def test_get_stock_universe_fallback():
    """Test getting fallback stock universe"""
    result = get_stock_universe(user_symbols=None, use_top_100=False)
    
    assert len(result) > 0
    assert result == FALLBACK_STOCK_UNIVERSE


def test_get_stock_universe_top_100():
    """Test getting top 100 stock universe"""
    # This might fetch dynamically or use default
    result = get_stock_universe(user_symbols=None, use_top_100=True)
    
    assert len(result) > 0
    assert isinstance(result, list)
    assert all(isinstance(s, str) for s in result)


def test_fallback_stock_universe_validity():
    """Test that fallback universe has valid symbols"""
    assert len(FALLBACK_STOCK_UNIVERSE) > 0
    assert "SPY" in FALLBACK_STOCK_UNIVERSE
    assert "QQQ" in FALLBACK_STOCK_UNIVERSE
    assert all(isinstance(s, str) and s.isupper() for s in FALLBACK_STOCK_UNIVERSE)


def test_default_top_100_stocks_validity():
    """Test that default top 100 list has valid symbols"""
    assert len(DEFAULT_TOP_100_STOCKS) >= 100
    assert "AAPL" in DEFAULT_TOP_100_STOCKS
    assert "MSFT" in DEFAULT_TOP_100_STOCKS
    assert "SPY" in DEFAULT_TOP_100_STOCKS
    assert all(isinstance(s, str) for s in DEFAULT_TOP_100_STOCKS)


def test_get_stock_universe_empty_user_symbols():
    """Test with empty user symbols list"""
    result = get_stock_universe(user_symbols=[], use_top_100=False)
    
    # When user provides empty list, function returns fallback universe
    # (This is the actual behavior - providing empty list is same as providing None)
    assert len(result) > 0
    assert result == FALLBACK_STOCK_UNIVERSE


def test_get_stock_universe_user_symbols_priority():
    """Test that user symbols take priority over other options"""
    user_symbols = ["CUSTOM1", "CUSTOM2"]
    result = get_stock_universe(user_symbols=user_symbols, use_top_100=True)
    
    assert result == ["CUSTOM1", "CUSTOM2"]  # User symbols take priority


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

