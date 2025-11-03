#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures
"""

import pytest
import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_config_values():
    """Provide safe mock configuration values for testing"""
    return {
        "MAX_CAPITAL": 10000.0,
        "SHORT_WINDOW": 3,
        "LONG_WINDOW": 9,
        "BASE_TAKE_PROFIT": 3.0,
        "BASE_STOP_LOSS": 1.0,
        "BASE_TRADE_SIZE_FRAC": 0.5,
        "MAX_EXPOSURE_PCT": 75.0,
        "MAX_LOSS_PER_TRADE_PCT": 2.0,
        "VIX_THRESHOLD": 30.0,
        "MAX_PRICE_DEVIATION_PCT": 10.0,
    }


@pytest.fixture
def sample_price_data():
    """Provide sample price data for testing"""
    return {
        "uptrend": [100 + i for i in range(30)],
        "downtrend": [130 - i for i in range(30)],
        "sideways": [100 + (i % 5) for i in range(30)],
        "volatile": [100 + (i % 2) * 10 * ((-1) ** i) for i in range(30)],
    }


@pytest.fixture(autouse=True)
def reset_test_environment(monkeypatch):
    """Reset environment variables before each test"""
    # Prevent tests from accidentally using real API keys
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    
    # Set test mode
    monkeypatch.setenv("TEST_MODE", "1")
    yield

