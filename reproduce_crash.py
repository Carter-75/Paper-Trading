
import sys
import os

# Add parent directory to path
sys.path.append(os.getcwd())

try:
    from risk.allocation_engine import AllocationEngine
    print("Import successful")
    
    class MockConfig:
        trade_size_frac_of_cap = 0.1
        max_exposure_pct = 100
        enable_kelly_sizing = False
        max_loss_per_trade_pct = 1.0
        stop_loss_percent = 2.0
        simulate_fees_enabled = False
        min_notional_usd = 10.0
        def wants_live_mode(self): return False
        
    class MockPM:
        def get_total_market_value(self): return 0.0
        
    ae = AllocationEngine(MockPM())
    ae.config = MockConfig()
    print("Instantiation successful")
    
except Exception as e:
    print(f"CRASH: {e}")
    import traceback
    traceback.print_exc()
