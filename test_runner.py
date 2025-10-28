#!/usr/bin/env python3
"""
Unit tests for Paper Trading Bot
Run with: pytest test_runner.py -v
Install: pip install pytest
"""

import pytest
from runner import (
    sma, decide_action, compute_confidence, pct_stddev,
    compute_order_qty_from_remaining, adjust_runtime_params
)
import config

def test_sma_basic():
    """Test simple moving average calculation"""
    closes = [100, 102, 104, 106, 108]
    assert sma(closes, 3) == 106.0
    assert sma(closes, 5) == 104.0

def test_sma_insufficient_data():
    """Test SMA with insufficient data"""
    closes = [100, 102]
    assert sma(closes, 5) == 102

def test_decide_action_buy():
    """Test buy signal when short MA > long MA"""
    closes = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122]
    action = decide_action(closes, short_w=3, long_w=9)
    assert action == "buy"

def test_decide_action_sell():
    """Test sell signal when short MA < long MA"""
    closes = [120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98]
    action = decide_action(closes, short_w=3, long_w=9)
    assert action == "sell"

def test_decide_action_hold():
    """Test hold signal when MAs are close"""
    closes = [100, 101, 100, 101, 100, 101, 100, 101, 100, 101, 100, 101]
    action = decide_action(closes, short_w=3, long_w=9)
    assert action == "hold"

def test_compute_confidence():
    """Test confidence calculation"""
    closes = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122,
              124, 126, 128, 130, 132, 134, 136, 138, 140]
    conf = compute_confidence(closes)
    assert conf > 0.05

def test_pct_stddev_low_volatility():
    """Test percentage standard deviation with low volatility"""
    closes = [100, 101, 100, 101, 100]
    vol = pct_stddev(closes)
    assert 0 < vol < 0.01

def test_pct_stddev_high_volatility():
    """Test percentage standard deviation with high volatility"""
    closes = [100, 120, 80, 110, 90]
    vol = pct_stddev(closes)
    assert vol > 0.1

def test_compute_order_qty():
    """Test order quantity calculation"""
    qty = compute_order_qty_from_remaining(100.0, 1000.0, 0.5)
    assert qty == 5.0
    
    qty = compute_order_qty_from_remaining(100.0, 550.0, 1.0)
    assert qty == 5.5

def test_adjust_runtime_params_basic():
    """Test runtime parameter adjustment"""
    tp, sl, frac = adjust_runtime_params(
        confidence=0.05, base_tp=3.0, base_sl=1.0, base_frac=0.5
    )
    assert tp >= 3.0
    assert sl >= 1.0
    assert frac >= 0.5

def test_adjust_runtime_params_bounds():
    """Test params stay within bounds"""
    tp, sl, frac = adjust_runtime_params(
        confidence=10.0, base_tp=20.0, base_sl=15.0, base_frac=0.95
    )
    assert tp <= config.MAX_TAKE_PROFIT_PERCENT
    assert sl <= config.MAX_STOP_LOSS_PERCENT
    assert frac <= config.MAX_TRADE_SIZE_FRAC

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

