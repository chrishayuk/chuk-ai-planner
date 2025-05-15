#!/usr/bin/env python
# examples/simple_plan_registry_demo.py
"""
simple_plan_registry_demo.py
===========================

A simple demonstration of the PlanRegistry with only the essential operations:
1. Create a basic plan with a tool step
2. Register it in the registry
3. Retrieve it and display its structure
"""

import os
import shutil
from typing import Dict, List, Any

from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.planner.plan_registry import PlanRegistry

# Create a temporary directory for the registry
REGISTRY_DIR = "temp_registry"
if os.path.exists(REGISTRY_DIR):
    shutil.rmtree(REGISTRY_DIR)
os.makedirs(REGISTRY_DIR)

def main():
    # Create a registry
    registry = PlanRegistry(storage_dir=REGISTRY_DIR)
    print(f"Created PlanRegistry in {REGISTRY_DIR}")
    
    # Create a simple plan
    plan = UniversalPlan(
        title="Simple Weather Check Plan",
        description="A plan that checks the weather in New York",
        tags=["weather", "simple"]
    )
    
    # Add metadata and variables
    plan.add_metadata("creator", "simple_demo")
    plan.set_variable("location", "New York")
    
    # Add a tool step
    step_id = plan.add_tool_step(
        title="Check Weather",
        tool="weather",
        args={"location": "${location}"},
        result_variable="weather_data"
    )
    
    # Register the plan
    plan_id = registry.register_plan(plan)
    print(f"Registered plan with ID: {plan_id}")
    
    # Clear memory and get plan from registry
    registry.plans = {}
    retrieved_plan = registry.get_plan(plan_id)
    
    # Display retrieved plan
    print("\nRetrieved Plan:")
    print(f"  Title: {retrieved_plan.title}")
    print(f"  Description: {retrieved_plan.description}")
    print(f"  Tags: {retrieved_plan.tags}")
    print(f"  Variables: {retrieved_plan.variables}")
    print(f"  Metadata: {retrieved_plan.metadata}")
    
    # Display plan structure
    print("\nPlan Structure:")
    print(retrieved_plan.outline())
    
    # Get plan as dictionary
    plan_dict = retrieved_plan.to_dict()
    print("\nPlan Steps:")
    for step in plan_dict["steps"]:
        print(f"  - {step['title']}")
        if "tool_calls" in step:
            for tool_call in step["tool_calls"]:
                print(f"    Tool: {tool_call['name']}")
                print(f"    Args: {tool_call['args']}")
        if "result_variable" in step and step["result_variable"]:
            print(f"    Result Variable: {step['result_variable']}")

if __name__ == "__main__":
    main()
    
    # Clean up
    print(f"\nCleaning up {REGISTRY_DIR}")
    shutil.rmtree(REGISTRY_DIR)