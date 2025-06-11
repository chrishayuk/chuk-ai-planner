#!/usr/bin/env python
# examples/universal_plan_executor_demo_simple.py
"""
Universal Executor Demo - Clean Simple Version
==============================================

• Uses the main UniversalExecutor (no reinventing the wheel!)
• Focuses on demonstrating workflow creation and execution
• Clean, readable code that showcases the framework's ease of use
• Proper variable resolution handled by the framework
"""

import asyncio
import json
import pprint
from typing import Any, Dict

from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor


# --------------------------------------------------------------------------- custom tools / fns
async def batch_weather_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get weather data for multiple locations."""
    locations = args.get("locations", [])
    print(f"📍 Getting weather for {len(locations)} locations")
    
    # Sample weather data
    weather_data = {
        "New York": {"temperature": 72, "conditions": "Partly cloudy", "humidity": 65},
        "London":   {"temperature": 62, "conditions": "Rainy",          "humidity": 80},
        "Tokyo":    {"temperature": 78, "conditions": "Sunny",          "humidity": 70},
        "Sydney":   {"temperature": 68, "conditions": "Clear",          "humidity": 60},
        "Cairo":    {"temperature": 90, "conditions": "Hot",            "humidity": 30},
    }
    
    # Get weather for each location
    results = {}
    for loc in locations:
        results[loc] = weather_data.get(loc, {"temperature": 75, "conditions": "Unknown", "humidity": 50})
    
    return {"results": results}


def analyze_weather_function(**kwargs) -> Dict[str, Any]:
    """Analyze weather data to extract statistics."""
    weather_data = kwargs.get("weather_data", {})
    print(f"📊 Analyzing weather data...")
    
    # Extract results from the weather data
    results = weather_data.get("results", weather_data)
    
    if not results:
        return {
            "average_temperature": 0,
            "average_humidity": 0,
            "most_common_condition": "Unknown",
            "condition_distribution": {},
            "locations_analyzed": 0
        }
    
    # Extract data and calculate statistics
    temperatures = []
    humidities = []
    conditions = {}
    
    for location, data in results.items():
        if isinstance(data, dict):
            temperatures.append(data.get("temperature", 0))
            humidities.append(data.get("humidity", 0))
            condition = data.get("conditions", "Unknown")
            conditions[condition] = conditions.get(condition, 0) + 1
    
    # Calculate averages
    avg_temp = sum(temperatures) / len(temperatures) if temperatures else 0
    avg_humidity = sum(humidities) / len(humidities) if humidities else 0
    
    # Find most common condition
    most_common = max(conditions.items(), key=lambda x: x[1])[0] if conditions else "Unknown"
    
    return {
        "average_temperature": round(avg_temp, 1),
        "average_humidity": round(avg_humidity, 1),
        "most_common_condition": most_common,
        "condition_distribution": conditions,
        "locations_analyzed": len(results),
    }


def create_report_function(**kwargs) -> Dict[str, Any]:
    """Create a report from analysis data."""
    analysis = kwargs.get("analysis", {})
    print(f"📝 Generating weather report...")
    
    return {
        "title": "Global Weather Analysis Report",
        "summary": (
            f"{analysis.get('locations_analyzed', 0)} cities analysed. "
            f"Avg T = {analysis.get('average_temperature', 'N/A')} °F, "
            f"Avg RH = {analysis.get('average_humidity', 'N/A')} %. "
            f"Most common: {analysis.get('most_common_condition', 'Unknown')}."
        ),
        "details": analysis,
    }


def format_visualization_function(**kwargs) -> Dict[str, Any]:
    """Format data for visualization."""
    weather_data = kwargs.get("weather_data", {})
    analysis = kwargs.get("analysis", {})
    print(f"🎨 Formatting visualization data...")
    
    # Extract results safely
    results = weather_data.get("results", weather_data)
    
    # Create temperature data for visualization
    temps = []
    for location, data in results.items():
        if isinstance(data, dict):
            temps.append({"location": location, "temperature": data.get("temperature", 0)})
    
    # Sort by temperature
    temps.sort(key=lambda x: x["temperature"], reverse=True)
    
    # Create condition data for visualization
    conds = []
    for condition, count in analysis.get("condition_distribution", {}).items():
        conds.append({"condition": condition, "count": count})
    
    return {
        "title": "Global Weather Visualization",
        "temperature_data": temps,
        "condition_data": conds,
    }


# --------------------------------------------------------------------------- plan factory
def create_weather_analysis_plan() -> UniversalPlan:
    """Create a weather analysis plan with proper variable flow."""
    plan = UniversalPlan(
        title="Global Weather Analysis",
        description="Analyse weather for multiple cities",
        tags=["weather", "analysis", "demo"]
    )
    
    # Define the target cities
    target_cities = ["New York", "London", "Tokyo", "Sydney", "Cairo"]
    plan.set_variable("target_cities", target_cities)
    
    # Step 1: Collect weather data
    collect_step = plan.add_tool_step(
        title="Collect Weather Data",
        tool="batch_weather",
        args={"locations": target_cities},
        result_variable="weather_data"
    )
    
    # Step 2: Analyze the data
    analyze_step = plan.add_function_step(
        title="Analyze Weather Data",
        function="analyze_weather",
        args={"weather_data": "${weather_data}"},
        result_variable="analysis",
        depends_on=[collect_step]
    )
    
    # Step 3: Generate report
    report_step = plan.add_function_step(
        title="Generate Weather Report",
        function="create_report",
        args={"analysis": "${analysis}"},
        result_variable="report",
        depends_on=[analyze_step]
    )
    
    # Step 4: Format visualization data (parallel with report)
    viz_step = plan.add_function_step(
        title="Format Visualization Data",
        function="format_visualization", 
        args={"weather_data": "${weather_data}", "analysis": "${analysis}"},
        result_variable="viz",
        depends_on=[analyze_step]  # Can run in parallel with report
    )
    
    return plan


# --------------------------------------------------------------------------- main
async def main():
    print("🌤️  Universal Executor Demo - Simple Version")
    print("=" * 50)

    # Create the executor (uses the robust framework implementation)
    executor = UniversalExecutor()
    
    # Register our custom tools and functions
    executor.register_tool("batch_weather", batch_weather_tool)
    executor.register_function("analyze_weather", analyze_weather_function)
    executor.register_function("create_report", create_report_function)
    executor.register_function("format_visualization", format_visualization_function)

    # Create the weather analysis plan
    plan = create_weather_analysis_plan()
    
    print(f"\n📋 Created plan: {plan.title}")
    print(f"📋 Plan ID: {plan.id}")
    print(f"📋 Target cities: {plan.variables['target_cities']}")
    
    print(f"\n📋 Plan structure:")
    print(plan.outline())

    print(f"\n▶️ Executing plan...")
    
    # Execute the plan
    result = await executor.execute_plan(plan)

    # Check execution result
    if not result["success"]:
        print(f"\n❌ Plan execution failed: {result.get('error', 'Unknown error')}")
        return

    print(f"\n✅ Plan executed successfully!")

    # Extract results for display
    weather_data = result["variables"].get("weather_data", {})
    analysis = result["variables"].get("analysis", {})
    report = result["variables"].get("report", {})
    viz = result["variables"].get("viz", {})

    # Display results
    print(f"\n" + "=" * 50)
    print("EXECUTION RESULTS")
    print("=" * 50)

    print(f"\n🌡️  WEATHER DATA")
    print("-" * 20)
    pprint.pprint(weather_data, width=100, sort_dicts=False)

    print(f"\n📊 ANALYSIS")
    print("-" * 20)
    pprint.pprint(analysis, width=100, sort_dicts=False)

    print(f"\n📝 REPORT")
    print("-" * 20)
    pprint.pprint(report, width=100, sort_dicts=False)

    print(f"\n📈 VISUALIZATION DATA")
    print("-" * 20)
    pprint.pprint(viz, width=100, sort_dicts=False)

    # Save results to file
    try:
        output_file = "simple_weather_analysis_results.json"
        with open(output_file, "w") as fp:
            # Use the framework's JSON serialization helper
            serializable_data = {}
            for key, value in result["variables"].items():
                try:
                    # Test if it's JSON serializable
                    json.dumps(value)
                    serializable_data[key] = value
                except (TypeError, ValueError):
                    # Convert to string if not serializable
                    serializable_data[key] = str(value)
            
            json.dump(serializable_data, fp, indent=2)
        print(f"\n💾 Results saved to {output_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results to file: {e}")

    print(f"\n🎉 Demo completed successfully!")


# --------------------------------------------------------------------------- entry point
if __name__ == "__main__":
    asyncio.run(main())