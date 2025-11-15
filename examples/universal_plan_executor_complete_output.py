#!/usr/bin/env python
# examples/universal_plan_executor_complete_output.py
"""
Universal Plan Working Demo - FIXED VERSION
===========================================

This version properly demonstrates:
1. Correct Universal Plan creation with flat step structure
2. Tool execution with visible logs
3. Variable resolution and result capture
4. Complete output display

The key fixes:
- Proper step creation without nesting
- Correct tool registration
- Variable resolution verification
- Result display enhancement
"""

import asyncio
import json
import pprint
from typing import Any, Dict

# Universal plan imports
from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor
from chuk_ai_planner.store.memory import InMemoryGraphStore

# ───────────────────── Simple Tool Implementations ─────────────────
async def weather_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get weather for a location with visible output"""
    location = args.get("location", "Unknown")
    print(f"🌤️ Getting weather for: {location}")
    
    # Mock weather data
    weather_data = {
        "New York": {"temperature": 72, "conditions": "Partly cloudy", "humidity": 65, "wind_speed": 5},
        "London": {"temperature": 62, "conditions": "Rainy", "humidity": 80, "wind_speed": 12},
        "Tokyo": {"temperature": 78, "conditions": "Sunny", "humidity": 70, "wind_speed": 3},
    }
    
    result = weather_data.get(location, {"temperature": 75, "conditions": "Unknown", "humidity": 50, "wind_speed": 0})
    print(f"🌤️ Weather result: {result['temperature']}°F, {result['conditions']}")
    return result


async def calculator_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Perform calculation with visible output"""
    operation = args.get("operation", "add")
    a = float(args.get("a", 0))
    b = float(args.get("b", 0))
    
    print(f"🧮 Calculating: {a} {operation} {b}")
    
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return {"error": "Division by zero"}
        result = a / b
    else:
        return {"error": f"Unknown operation: {operation}"}
    
    print(f"🧮 Calculation result: {result}")
    return {"result": result, "operation": operation, "operands": [a, b]}


async def search_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Search for information with visible output"""
    query = args.get("query", "")
    print(f"🔍 Searching for: '{query}'")
    
    # Mock search results
    results = [
        {"title": f"Complete Guide to {query}", "url": f"https://example.com/{query.replace(' ', '-')}", "snippet": f"Comprehensive information about {query}"},
        {"title": f"{query} - Latest Research", "url": f"https://research.org/{query.replace(' ', '_')}", "snippet": f"Recent developments in {query}"},
        {"title": f"Best Practices for {query}", "url": f"https://bestpractices.com/{query.replace(' ', '-')}", "snippet": f"Expert recommendations for {query}"}
    ]
    
    print(f"🔍 Found {len(results)} results")
    return {"query": query, "total_results": len(results), "results": results}


async def analyzer_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze data with visible output"""
    weather_data = args.get("weather_data", {})
    calculation_result = args.get("calculation_result", {})
    
    print("📊 Analyzing weather and calculation data...")
    
    # Extract values
    temperature = weather_data.get("temperature", 0)
    calc_result = calculation_result.get("result", 0)
    
    # Perform analysis
    temp_category = "hot" if temperature > 75 else "moderate" if temperature > 60 else "cold"
    calc_magnitude = "large" if calc_result > 1000 else "medium" if calc_result > 100 else "small"
    
    analysis = {
        "temperature_analysis": {
            "value": temperature,
            "category": temp_category
        },
        "calculation_analysis": {
            "value": calc_result,
            "magnitude": calc_magnitude
        },
        "combined_score": temperature + (calc_result / 100),
        "summary": f"Temperature: {temperature}°F ({temp_category}), Calculation: {calc_result} ({calc_magnitude})"
    }
    
    print(f"📊 Analysis complete - Combined score: {analysis['combined_score']:.2f}")
    return analysis


# ───────────────────── Create Proper Universal Plan ───────────────
def create_working_plan() -> UniversalPlan:
    """Create a Universal Plan with correct flat structure"""
    print("🟢 Creating Universal Plan with correct structure...")
    
    # Create plan with custom graph store
    graph_store = InMemoryGraphStore()
    plan = UniversalPlan(
        title="Working Demo Plan",
        description="Demonstrates proper Universal Plan execution",
        graph=graph_store,
        tags=["demo", "working"]
    )
    
    # Set variables
    plan.set_variable("target_city", "New York")
    plan.set_variable("num_a", 235.5)
    plan.set_variable("num_b", 18.75)
    plan.set_variable("search_query", "renewable energy solutions")
    
    print("📋 Variables set:")
    print(f"   - target_city: {plan.variables['target_city']}")
    print(f"   - num_a: {plan.variables['num_a']}")
    print(f"   - num_b: {plan.variables['num_b']}")
    print(f"   - search_query: {plan.variables['search_query']}")
    
    # Add steps using the direct method to avoid nesting
    step1_id = plan.add_tool_step(
        title="Get Weather Data",
        tool="weather",
        args={"location": "${target_city}"},
        result_variable="weather_info"
    )
    
    step2_id = plan.add_tool_step(
        title="Perform Calculation",
        tool="calculator",
        args={"operation": "multiply", "a": "${num_a}", "b": "${num_b}"},
        result_variable="calc_info"
    )
    
    step3_id = plan.add_tool_step(
        title="Search Information",
        tool="search",
        args={"query": "${search_query}"},
        result_variable="search_info"
    )
    
    step4_id = plan.add_tool_step(
        title="Analyze Results",
        tool="analyzer",
        args={"weather_data": "${weather_info}", "calculation_result": "${calc_info}"},
        result_variable="analysis_info",
        depends_on=[step1_id, step2_id]  # Depends on weather and calculation
    )
    
    print("\n📊 Plan structure created:")
    print(f"   - Step 1: Weather ({step1_id[:8]})")
    print(f"   - Step 2: Calculation ({step2_id[:8]})")
    print(f"   - Step 3: Search ({step3_id[:8]})")
    print(f"   - Step 4: Analysis ({step4_id[:8]}) - depends on steps 1 & 2")
    
    # Display plan outline
    print("\n📋 Plan Outline:")
    print(plan.outline())
    
    return plan


# ───────────────────── Execute with Detailed Logging ──────────────
async def execute_with_logging(plan: UniversalPlan):
    """Execute plan with detailed logging and variable tracking"""
    print("\n🔄 Setting up Universal Executor...")
    
    # Create executor
    executor = UniversalExecutor(graph_store=plan.graph)
    
    # Register tools
    executor.register_tool("weather", weather_tool)
    executor.register_tool("calculator", calculator_tool)
    executor.register_tool("search", search_tool)
    executor.register_tool("analyzer", analyzer_tool)
    
    print("✅ Tools registered: weather, calculator, search, analyzer")
    
    # Show initial variables
    print("\n📋 Initial Variables:")
    for name, value in plan.variables.items():
        print(f"   - {name}: {value}")
    
    # Execute plan
    print("\n🏃 Executing plan...")
    print("=" * 50)
    
    result = await executor.execute_plan(plan)
    
    print("=" * 50)
    print("🏁 Plan execution completed!")
    
    return result


# ───────────────────── Display Complete Results ───────────────────
def display_results(result: Dict[str, Any], plan: UniversalPlan):
    """Display comprehensive execution results"""
    print("\n🎉 EXECUTION RESULTS")
    print("=" * 60)
    
    if result["success"]:
        print("✅ Status: SUCCESS")
        
        # Show all variables
        variables = result["variables"]
        print(f"\n📋 All Variables ({len(variables)} total):")
        
        # Input variables
        input_vars = ["target_city", "num_a", "num_b", "search_query"]
        print("\n🔸 Input Variables:")
        for var_name in input_vars:
            if var_name in variables:
                print(f"   - {var_name}: {variables[var_name]}")
        
        # Output variables
        output_vars = [k for k in variables.keys() if k not in input_vars]
        print(f"\n🔸 Output Variables ({len(output_vars)} generated):")
        for var_name in output_vars:
            print(f"\n   📄 {var_name}:")
            value = variables[var_name]
            if isinstance(value, (dict, list)):
                # Pretty print complex structures with indentation
                lines = pprint.pformat(value, width=70, sort_dicts=False).split('\n')
                for line in lines:
                    print(f"      {line}")
            else:
                print(f"      {value}")
        
        # Summary statistics
        print("\n📊 Execution Summary:")
        print(f"   - Plan: {plan.title}")
        print(f"   - Total Variables: {len(variables)}")
        print(f"   - Input Variables: {len(input_vars)}")
        print(f"   - Output Variables: {len(output_vars)}")
        print("   - Success: ✅")
        
        # Save results
        output_file = "working_demo_results.json"
        with open(output_file, "w") as f:
            # Convert to JSON-safe format
            json_safe = {}
            for k, v in variables.items():
                try:
                    json.dumps(v)
                    json_safe[k] = v
                except (TypeError, ValueError):
                    json_safe[k] = str(v)
            json.dump(json_safe, f, indent=2)
        
        print(f"   - Results saved to: {output_file}")
        
    else:
        print("❌ Status: FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")


# ───────────────────── Main Demo Function ──────────────────────────
async def main():
    """Run the working Universal Plan demo"""
    print("🚀 Universal Plan Working Demo")
    print("=" * 50)
    
    # Create plan
    plan = create_working_plan()
    
    # Execute plan
    result = await execute_with_logging(plan)
    
    # Display results
    display_results(result, plan)
    
    print("\n🎊 Demo Complete!")
    print("This demo showed:")
    print("  ✅ Proper Universal Plan creation")
    print("  ✅ Tool execution with visible logs")
    print("  ✅ Variable resolution in action")
    print("  ✅ Complete result capture and display")


if __name__ == "__main__":
    asyncio.run(main())