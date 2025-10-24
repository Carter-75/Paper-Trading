#!/usr/bin/env python3
"""
Comprehensive system test - Verifies all components work together.
Run this to ensure everything is properly configured.
"""

import sys
import os


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import config
        import runner
        import optimizer
        import optimizer_binary
        import portfolio_manager
        import stock_scanner
        import runner_multi
        import scan_best_stocks
        import validate_setup
        import test_signals
        print("  ✅ All modules import successfully")
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_config():
    """Test configuration is valid."""
    print("\nTesting configuration...")
    try:
        import config
        
        # Check critical configs exist
        assert hasattr(config, 'SHORT_WINDOW'), "Missing SHORT_WINDOW"
        assert hasattr(config, 'LONG_WINDOW'), "Missing LONG_WINDOW"
        assert hasattr(config, 'MAX_CAP_USD'), "Missing MAX_CAP_USD"
        assert hasattr(config, 'PROFITABILITY_MIN_EXPECTED_USD'), "Missing PROFITABILITY_MIN_EXPECTED_USD"
        
        # Check values are sensible
        assert config.SHORT_WINDOW < config.LONG_WINDOW, "SHORT_WINDOW must be < LONG_WINDOW"
        assert config.MAX_CAP_USD > 0, "MAX_CAP_USD must be positive"
        
        print(f"  ✅ Configuration valid")
        print(f"     SHORT_WINDOW: {config.SHORT_WINDOW}")
        print(f"     LONG_WINDOW: {config.LONG_WINDOW}")
        print(f"     MAX_CAP_USD: ${config.MAX_CAP_USD}")
        print(f"     PROFITABILITY_MIN: ${config.PROFITABILITY_MIN_EXPECTED_USD}")
        return True
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        return False


def test_portfolio_manager():
    """Test portfolio manager functionality."""
    print("\nTesting portfolio manager...")
    try:
        from portfolio_manager import PortfolioManager
        
        # Create test portfolio
        pm = PortfolioManager(portfolio_file="test_portfolio.json")
        
        # Test adding position
        pm.update_position("AAPL", 10.0, 150.0, 1550.0, 50.0)
        assert pm.get_position_count() == 1, "Should have 1 position"
        
        # Test getting position
        pos = pm.get_position("AAPL")
        assert pos is not None, "Should find AAPL position"
        assert pos['qty'] == 10.0, "Quantity should be 10"
        
        # Test total value
        total = pm.get_total_market_value()
        assert total == 1550.0, f"Total should be 1550, got {total}"
        
        # Test removing position
        pm.remove_position("AAPL")
        assert pm.get_position_count() == 0, "Should have 0 positions after removal"
        
        # Cleanup
        if os.path.exists("test_portfolio.json"):
            os.remove("test_portfolio.json")
        
        print("  ✅ Portfolio manager works correctly")
        return True
    except Exception as e:
        print(f"  ❌ Portfolio manager test failed: {e}")
        # Cleanup on error
        if os.path.exists("test_portfolio.json"):
            os.remove("test_portfolio.json")
        return False


def test_stock_scanner_logic():
    """Test stock scanner logic (without API calls)."""
    print("\nTesting stock scanner logic...")
    try:
        from stock_scanner import DEFAULT_STOCK_UNIVERSE, get_stock_universe
        
        # Check universe is populated
        assert len(DEFAULT_STOCK_UNIVERSE) >= 20, "Universe should have at least 20 stocks"
        
        # Test get_stock_universe
        universe = get_stock_universe()
        assert len(universe) >= 20, "Universe function should return stocks"
        
        # Test with custom symbols
        custom = get_stock_universe(["AAPL", "MSFT"])
        assert custom == ["AAPL", "MSFT"], "Should return custom symbols"
        
        print(f"  ✅ Stock scanner logic works")
        print(f"     Default universe size: {len(DEFAULT_STOCK_UNIVERSE)}")
        return True
    except Exception as e:
        print(f"  ❌ Stock scanner test failed: {e}")
        return False


def test_validation_logic():
    """Test validation functions."""
    print("\nTesting validation logic...")
    try:
        from validate_setup import validate_stock_args, validate_capital, validate_interval
        
        # Test valid config
        valid, msg = validate_stock_args(["AAPL", "MSFT"], 5)
        assert valid, f"Should be valid: {msg}"
        
        # Test invalid - too many stocks
        valid, msg = validate_stock_args(["A", "B", "C"], 2)
        assert not valid, "Should be invalid (too many stocks)"
        
        # Test capital validation
        valid, msg = validate_capital(1000.0, 100.0, 10)
        assert valid, f"Should be valid: {msg}"
        
        # Test interval validation
        valid, msg = validate_interval(0.25)
        assert valid, f"Should be valid: {msg}"
        
        print("  ✅ Validation logic works correctly")
        return True
    except Exception as e:
        print(f"  ❌ Validation test failed: {e}")
        return False


def test_env_file():
    """Test .env file exists and has required keys."""
    print("\nTesting .env file...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        required_keys = [
            "APCA_API_KEY_ID",
            "APCA_API_SECRET_KEY",
            "POLYGON_API_KEY"
        ]
        
        missing = []
        for key in required_keys:
            if not os.getenv(key):
                missing.append(key)
        
        if missing:
            print(f"  ⚠️  Missing keys in .env: {', '.join(missing)}")
            print(f"     Bot may not work without these!")
            return False
        else:
            print(f"  ✅ All required environment variables present")
            return True
    except Exception as e:
        print(f"  ❌ .env test failed: {e}")
        return False


def main():
    print("="*70)
    print(" COMPREHENSIVE SYSTEM TEST".center(70))
    print("="*70)
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Portfolio Manager", test_portfolio_manager()))
    results.append(("Stock Scanner", test_stock_scanner_logic()))
    results.append(("Validation", test_validation_logic()))
    results.append(("Environment", test_env_file()))
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY".center(70))
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")
    
    print("\n" + "="*70)
    print(f" {passed}/{total} tests passed".center(70))
    print("="*70)
    
    if passed == total:
        print("\n🎉 All systems operational! Ready to trade.")
        print("\nNext steps:")
        print("  1. Run: python quick_start.py")
        print("  2. Or manually: python scan_best_stocks.py --interval 0.25 --cap 100 --top 5 -v")
        return 0
    else:
        print("\n⚠️  Some tests failed. Fix issues before running bot.")
        print("\nCommon fixes:")
        print("  - Ensure .env file has API keys")
        print("  - Run: pip install -r requirements.txt")
        print("  - Check config.py for valid values")
        return 1


if __name__ == "__main__":
    sys.exit(main())

