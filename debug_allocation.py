
from risk.allocation_engine import AllocationResult

try:
    print("Attempting to instantiate AllocationResult...")
    ar = AllocationResult("SYM", 0, 0.0, 0.0, "Reason", False)
    print(f"Success: {ar}")
except Exception as e:
    print(f"FAILED: {e}")
    import inspect
    print(f"Signature: {inspect.signature(AllocationResult)}")
