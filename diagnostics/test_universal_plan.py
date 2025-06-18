#!/usr/bin/env python
# test_universal_plan.py
"""
Test UniversalPlan serialization to find the exact issue
"""

import json
from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.utils.serialization import unfreeze_data

print("Testing UniversalPlan serialization...")

# Create a plan similar to the demo
plan = UniversalPlan(
    title="Test Plan",
    description="Test description",
    tags=["test", "debug"]
)

# Add metadata
plan.add_metadata("created_by", "test")
plan.add_metadata("priority", "medium")

# Add variables
plan.set_variable("query", "test query")
plan.set_variable("max_results", 5)

# Add steps
plan.step("Initial Research")

# Add a tool step
search_step_id = plan.add_tool_step(
    title="Search for information",
    tool="search",
    args={"query": "test"},
    result_variable="search_results"
)

print("✅ Plan created successfully")

# Test the to_dict method
try:
    print("Testing plan.to_dict()...")
    plan_dict = plan.to_dict()
    print("✅ plan.to_dict() succeeded")
    print(f"Keys: {list(plan_dict.keys())}")
    
    # Check types in the dict
    print("\nChecking types in plan_dict:")
    for key, value in plan_dict.items():
        print(f"  {key}: {type(value)}")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {type(subvalue)}")
                if hasattr(subvalue, '__dict__'):
                    print(f"      __dict__: {type(subvalue.__dict__)}")
    
    # Test direct JSON serialization
    print("\nTesting direct JSON serialization...")
    try:
        json_str = json.dumps(plan_dict)
        print("✅ Direct JSON serialization succeeded")
    except Exception as json_error:
        print(f"❌ Direct JSON serialization failed: {json_error}")
        print(f"Error type: {type(json_error)}")
        
        # Test with unfreeze_data
        print("\nTesting with unfreeze_data...")
        try:
            unfrozen = unfreeze_data(plan_dict)
            json_str = json.dumps(unfrozen)
            print("✅ JSON serialization with unfreeze_data succeeded")
        except Exception as unfreeze_error:
            print(f"❌ JSON serialization with unfreeze_data failed: {unfreeze_error}")
            
            # Find the problematic object
            print("\nDeep inspection to find problematic object...")
            def find_problem(obj, path="root"):
                try:
                    json.dumps(obj)
                except Exception as e:
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            find_problem(value, f"{path}.{key}")
                    elif isinstance(obj, (list, tuple)):
                        for i, item in enumerate(obj):
                            find_problem(item, f"{path}[{i}]")
                    else:
                        print(f"Problem found at {path}: {type(obj)} = {repr(obj)[:100]}")
            
            find_problem(plan_dict)

except Exception as dict_error:
    print(f"❌ plan.to_dict() failed: {dict_error}")
    import traceback
    traceback.print_exc()