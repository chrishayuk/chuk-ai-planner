#!/usr/bin/env python
# test_serialization.py
"""
Quick test to verify the serialization utilities work correctly
"""

import json
from types import MappingProxyType

# Test the existing utility
try:
    from chuk_ai_planner.utils.serialization import unfreeze_data
    print("✅ Successfully imported unfreeze_data")
    
    # Test with a mappingproxy
    test_dict = {"key1": "value1", "key2": {"nested": "value"}}
    proxy = MappingProxyType(test_dict)
    
    print(f"Original proxy type: {type(proxy)}")
    
    # Test unfreezing
    unfrozen = unfreeze_data(proxy)
    print(f"Unfrozen type: {type(unfrozen)}")
    
    # Test JSON serialization
    json_str = json.dumps(unfrozen)
    print("✅ JSON serialization successful")
    print(f"JSON result: {json_str}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Test failed: {e}")

# Test a more comprehensive approach
def comprehensive_unfreeze(obj):
    """More comprehensive unfreezing function"""
    if isinstance(obj, MappingProxyType):
        return {k: comprehensive_unfreeze(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: comprehensive_unfreeze(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [comprehensive_unfreeze(item) for item in obj]
    elif isinstance(obj, frozenset):
        return [comprehensive_unfreeze(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return comprehensive_unfreeze(obj.__dict__)
    else:
        return obj

print("\nTesting comprehensive unfreezing...")
test_data = {
    "proxy": MappingProxyType({"a": 1, "b": 2}),
    "tuple": (1, 2, 3),
    "frozenset": frozenset([1, 2, 3]),
    "nested": {
        "proxy": MappingProxyType({"nested": "value"}),
        "normal": "value"
    }
}

try:
    unfrozen = comprehensive_unfreeze(test_data)
    json_str = json.dumps(unfrozen)
    print("✅ Comprehensive unfreezing successful")
    print(f"Result: {json_str}")
except Exception as e:
    print(f"❌ Comprehensive test failed: {e}")