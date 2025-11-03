#!/usr/bin/env python3
"""
Unit tests for Risk Control Functions in runner.py
Run with: pytest test_risk_controls.py -v
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import config


# Import functions from runner.py
try:
    from runner import (
        check_exposure_limit,
        check_kill_switch,
        verify_order_safety,
        calculate_max_position_size_for_risk,
        check_vix_filter,
        get_vix_level
    )
except ImportError:
    pytest.skip("Could not import runner functions", allow_module_level=True)


class TestExposureLimit:
    """Tests for check_exposure_limit function"""
    
    def test_exposure_limit_within_bounds(self):
        """Test when exposure is within limits"""
        can_continue, scaled_value, msg = check_exposure_limit(
            total_invested=5000.0,
            max_capital=10000.0,
            new_order_value=1000.0
        )
        
        assert can_continue is True
        assert scaled_value == 1000.0
        assert "ok" in msg.lower() or "exposure" in msg.lower()
    
    def test_exposure_limit_at_max(self):
        """Test when exposure is at maximum"""
        can_continue, scaled_value, msg = check_exposure_limit(
            total_invested=7500.0,  # 75% of 10000
            max_capital=10000.0,
            new_order_value=1000.0
        )
        
        # Should block new order
        assert can_continue is False
        assert scaled_value == 0.0
        assert "exposure limit" in msg.lower()
    
    def test_exposure_limit_partial_order(self):
        """Test when order needs to be scaled down"""
        can_continue, scaled_value, msg = check_exposure_limit(
            total_invested=7000.0,  # 70% of 10000
            max_capital=10000.0,
            new_order_value=1000.0  # Would push to 80%
        )
        
        # Should scale down order
        assert can_continue is True
        assert 0 < scaled_value < 1000.0
        assert "reducing" in msg.lower() or "order size" in msg.lower()
    
    def test_exposure_limit_disabled(self):
        """Test when max_capital is None (disabled)"""
        can_continue, scaled_value, msg = check_exposure_limit(
            total_invested=5000.0,
            max_capital=0.0,  # Use 0.0 instead of None to test disabled state
            new_order_value=1000.0
        )
        
        # When max_capital is 0, should block all orders
        assert can_continue is False
        assert scaled_value == 0.0
    
    def test_exposure_limit_zero_capital(self):
        """Test with zero max capital"""
        can_continue, scaled_value, msg = check_exposure_limit(
            total_invested=0.0,
            max_capital=0.0,
            new_order_value=1000.0
        )
        
        assert can_continue is False
        assert scaled_value == 0.0


class TestKillSwitch:
    """Tests for check_kill_switch function"""
    
    @pytest.fixture
    def kill_switch_file(self):
        """Create temporary kill switch file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_file = f.name
        yield temp_file
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    def test_kill_switch_not_present(self, kill_switch_file):
        """Test when kill switch file doesn't exist"""
        # Remove file
        if os.path.exists(kill_switch_file):
            os.remove(kill_switch_file)
        
        with patch.object(config, 'KILL_SWITCH_FILE', kill_switch_file):
            with patch.object(config, 'KILL_SWITCH_ENABLED', True):
                can_continue, msg = check_kill_switch()
                
                assert can_continue is True
                assert msg == ""
    
    def test_kill_switch_present(self, kill_switch_file):
        """Test when kill switch file exists"""
        # Create file
        with open(kill_switch_file, 'w') as f:
            f.write("STOP")
        
        with patch.object(config, 'KILL_SWITCH_FILE', kill_switch_file):
            with patch.object(config, 'KILL_SWITCH_ENABLED', True):
                can_continue, msg = check_kill_switch()
                
                assert can_continue is False
                assert "kill switch" in msg.lower()
    
    def test_kill_switch_disabled(self, kill_switch_file):
        """Test when kill switch is disabled"""
        # Create file
        with open(kill_switch_file, 'w') as f:
            f.write("STOP")
        
        with patch.object(config, 'KILL_SWITCH_FILE', kill_switch_file):
            with patch.object(config, 'KILL_SWITCH_ENABLED', False):
                can_continue, msg = check_kill_switch()
                
                assert can_continue is True
                assert msg == ""


class TestOrderVerification:
    """Tests for verify_order_safety function"""
    
    def test_order_verification_disabled(self):
        """Test when order verification is disabled"""
        mock_client = Mock()
        
        with patch.object(config, 'ORDER_VERIFICATION_ENABLED', False):
            is_safe, msg = verify_order_safety(
                mock_client, "AAPL", "buy", 10.0, 150.0, 150.0
            )
            
            assert is_safe is True
            assert msg == ""
    
    def test_order_verification_price_within_limits(self):
        """Test when price is within acceptable deviation"""
        mock_client = Mock()
        
        with patch.object(config, 'ORDER_VERIFICATION_ENABLED', True):
            with patch.object(config, 'MAX_PRICE_DEVIATION_PCT', 10.0):
                is_safe, msg = verify_order_safety(
                    mock_client, "AAPL", "buy", 10.0, 155.0, 150.0  # 3.3% deviation
                )
                
                assert is_safe is True
    
    def test_order_verification_price_too_high(self):
        """Test when price deviates too much"""
        mock_client = Mock()
        
        with patch.object(config, 'ORDER_VERIFICATION_ENABLED', True):
            with patch.object(config, 'MAX_PRICE_DEVIATION_PCT', 5.0):
                is_safe, msg = verify_order_safety(
                    mock_client, "AAPL", "buy", 10.0, 200.0, 150.0  # 33% deviation
                )
                
                assert is_safe is False
                assert "price deviation" in msg.lower()
    
    def test_order_verification_no_last_price(self):
        """Test verification without last known price"""
        mock_client = Mock()
        
        with patch.object(config, 'ORDER_VERIFICATION_ENABLED', True):
            is_safe, msg = verify_order_safety(
                mock_client, "AAPL", "buy", 10.0, 150.0, None
            )
            
            # Should pass if no reference price available
            assert is_safe is True


class TestMaxLossPerTrade:
    """Tests for calculate_max_position_size_for_risk function"""
    
    def test_max_loss_calculation_basic(self):
        """Test basic position size calculation"""
        with patch.object(config, 'MAX_LOSS_PER_TRADE_PCT', 2.0):
            max_position, msg = calculate_max_position_size_for_risk(
                total_capital=10000.0,
                stop_loss_pct=1.0,
                available_capital=5000.0
            )
            
            # Max loss = 2% of 10000 = $200
            # Stop loss = 1%
            # Max position = $200 / 0.01 = $20,000
            # But capped at available capital (5000)
            assert max_position == 5000.0
    
    def test_max_loss_limits_position(self):
        """Test when risk limit restricts position size"""
        with patch.object(config, 'MAX_LOSS_PER_TRADE_PCT', 1.0):
            max_position, msg = calculate_max_position_size_for_risk(
                total_capital=10000.0,
                stop_loss_pct=2.0,  # Larger stop loss
                available_capital=10000.0
            )
            
            # Max loss = 1% of 10000 = $100
            # Stop loss = 2%
            # Max position = $100 / 0.02 = $5,000
            assert max_position == 5000.0
            assert "risk" in msg.lower()
    
    def test_max_loss_zero_stop_loss(self):
        """Test with zero stop loss (edge case)"""
        with patch.object(config, 'MAX_LOSS_PER_TRADE_PCT', 2.0):
            max_position, msg = calculate_max_position_size_for_risk(
                total_capital=10000.0,
                stop_loss_pct=0.0,
                available_capital=5000.0
            )
            
            # With zero stop loss, max risk is 2% of 10000 = $200
            # Function calculates max position based on risk formula
            assert max_position > 0.0
            assert max_position <= 5000.0


class TestVixFilter:
    """Tests for VIX filter functions"""
    
    def test_vix_filter_disabled(self):
        """Test when VIX filter is disabled"""
        with patch.object(config, 'VIX_FILTER_ENABLED', False):
            can_trade, msg = check_vix_filter()
            
            assert can_trade is True
            assert msg == ""
    
    @patch('runner.get_vix_level')
    def test_vix_filter_below_threshold(self, mock_get_vix):
        """Test when VIX is below threshold"""
        mock_get_vix.return_value = (20.0, "VIX: 20.0")
        
        with patch.object(config, 'VIX_FILTER_ENABLED', True):
            with patch.object(config, 'VIX_THRESHOLD', 30.0):
                can_trade, msg = check_vix_filter()
                
                assert can_trade is True
                assert "vix" in msg.lower() or msg != ""
    
    @patch('runner.get_vix_level')
    def test_vix_filter_above_threshold(self, mock_get_vix):
        """Test when VIX is above threshold"""
        mock_get_vix.return_value = (35.0, "VIX: 35.0")
        
        with patch.object(config, 'VIX_FILTER_ENABLED', True):
            with patch.object(config, 'VIX_THRESHOLD', 30.0):
                can_trade, msg = check_vix_filter()
                
                assert can_trade is False
                assert "vix" in msg.lower()
    
    @patch('runner.get_vix_level')
    def test_vix_filter_fetch_failure(self, mock_get_vix):
        """Test when VIX fetch fails"""
        mock_get_vix.return_value = (None, "Failed to fetch VIX")
        
        with patch.object(config, 'VIX_FILTER_ENABLED', True):
            can_trade, msg = check_vix_filter()
            
            # Should allow trading if VIX fetch fails (don't block on data errors)
            assert can_trade is True


class TestEdgeCases:
    """Tests for edge cases and error conditions"""
    
    def test_exposure_limit_negative_values(self):
        """Test exposure limit with negative values"""
        can_continue, scaled_value, msg = check_exposure_limit(
            total_invested=-1000.0,
            max_capital=10000.0,
            new_order_value=1000.0
        )
        
        # Should handle gracefully
        assert isinstance(can_continue, bool)
        assert isinstance(scaled_value, (int, float))
    
    def test_max_loss_with_large_stop_loss(self):
        """Test max loss calculation with large stop loss percentage"""
        with patch.object(config, 'MAX_LOSS_PER_TRADE_PCT', 2.0):
            max_position, msg = calculate_max_position_size_for_risk(
                total_capital=10000.0,
                stop_loss_pct=50.0,  # Very large stop loss
                available_capital=10000.0
            )
            
            # Should limit position size appropriately
            assert 0 < max_position <= 10000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

