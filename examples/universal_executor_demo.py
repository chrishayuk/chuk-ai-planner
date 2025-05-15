#!/usr/bin/env python
# examples/universal_executor_demo.py
"""
Universal Executor Demo
======================

This script demonstrates the improved UniversalExecutor with:

- Proper variable substitution for plan steps
- Correct handling of dependencies between steps
- Robust execution of tools and functions
- Clean result processing

The demo creates a weather analysis plan with several steps:
1. Collect weather data for multiple cities
2. Analyze the data to extract statistics
3. Generate a report based on the analysis
4. Format data for visualization
"""

import asyncio
import json
import pprint
from typing import Any, Dict

from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor


# --------------------------------------------------------------------------- custom tools / fns
async def batch_weather_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get weather data for multiple locations.
    
    Parameters
    ----------
    args : Dict[str, Any]
        Dictionary with a 'locations' key containing a list of location names
        
    Returns
    -------
    Dict[str, Any]
        Weather data for each location
    """
    print(f"📍 Getting weather for {len(args.get('locations', []))} locations")
    
    # Get locations from args
    locations = args.get("locations", [])
    
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


def analyze_weather_function(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze weather data to extract statistics.
    
    Parameters
    ----------
    weather_data : Dict[str, Any]
        Weather data with a 'results' key containing location data
        
    Returns
    -------
    Dict[str, Any]
        Statistical analysis of the weather data
    """
    print("📊 Analyzing weather data")
    
    # Extract results from the weather data
    results = weather_data.get("results", weather_data)
    
    # Calculate statistics
    n = len(results)
    if n == 0:
        return {
            "average_temperature": 0,
            "average_humidity": 0,
            "most_common_condition": "Unknown",
            "condition_distribution": {},
            "locations_analyzed": 0
        }
    
    # Extract data
    temperatures = []
    humidities = []
    conditions = {}
    
    for location, data in results.items():
        # Extract temperature
        temp = data.get("temperature")
        if temp is not None:
            temperatures.append(temp)
        
        # Extract humidity
        humidity = data.get("humidity")
        if humidity is not None:
            humidities.append(humidity)
        
        # Extract condition
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
        "locations_analyzed": n,
    }


def create_report_function(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a report from analysis data.
    
    Parameters
    ----------
    analysis : Dict[str, Any]
        Statistical analysis of weather data
        
    Returns
    -------
    Dict[str, Any]
        Formatted report with title and summary
    """
    print("📝 Generating weather report")
    
    # Create the report
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


def format_visualization_function(weather_data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format data for visualization.
    
    Parameters
    ----------
    weather_data : Dict[str, Any]
        Weather data with a 'results' key containing location data
    analysis : Dict[str, Any]
        Statistical analysis of weather data
        
    Returns
    -------
    Dict[str, Any]
        Data formatted for visualization
    """
    print("🎨 Formatting visualization data")
    
    # Extract results safely
    results = weather_data.get("results", weather_data)
    
    # Create temperature data for visualization
    temps = []
    for location, data in results.items():
        temps.append({"location": location, "temperature": data.get("temperature", 0)})
    
    # Sort temperatures
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
def make_plan(store=None) -> UniversalPlan:
    """
    Create a weather analysis plan.
    
    Parameters
    ----------
    store : GraphStore, optional
        Graph store to use for the plan
        
    Returns
    -------
    UniversalPlan
        The created plan
    """
    plan = UniversalPlan(
        title="Global Weather Analysis",
        description="Analyse weather for multiple cities",
        tags=["weather", "analysis", "demo"],
        graph=store,
    )
    
    # Define the target cities directly in the plan
    target_cities = ["New York", "London", "Tokyo", "Sydney", "Cairo"]
    plan.set_variable("target_cities", target_cities)
    plan.save()

    # Add steps with dependencies
    s1 = plan.add_tool_step(
        "Collect Weather Data",
        tool="batch_weather",
        args={"locations": target_cities},  # Direct reference
        result_variable="weather_data",
    )
    s2 = plan.add_function_step(
        "Analyze Weather Data",
        function="analyze_weather",
        args={"weather_data": "${weather_data}"},
        result_variable="analysis",
        depends_on=[s1],
    )
    s3 = plan.add_function_step(
        "Generate Weather Report",
        function="create_report",
        args={"analysis": "${analysis}"},
        result_variable="report",
        depends_on=[s2],
    )
    s4 = plan.add_function_step(
        "Format Visualization Data",
        function="format_visualization",
        args={"weather_data": "${weather_data}", "analysis": "${analysis}"},
        result_variable="viz",
        depends_on=[s2],
    )
    plan.save()
    return plan


# --------------------------------------------------------------------------- main
async def main():
    """Main function that creates and executes the plan."""
    print("🌤️  Universal Executor Demo\n" + "=" * 35)

    # Create executor
    executor = UniversalExecutor()
    
    # Create plan
    plan = make_plan(executor.graph_store)

    # Register tools / functions
    executor.register_tool("batch_weather", batch_weather_tool)
    executor.register_function("analyze_weather", analyze_weather_function)
    executor.register_function("create_report", create_report_function)
    executor.register_function("format_visualization", format_visualization_function)

    print("\n▶️ Executing Plan...")
    try:
        res = await executor.execute_plan(plan)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Execution failed with error: {e}")
        return

    if not res["success"]:
        print(f"\n❌ Plan failed: {res['error']}")
        return

    # Extract results
    weather_data = res["variables"].get("weather_data", {})
    analysis = res["variables"].get("analysis", {})
    report = res["variables"].get("report", {})
    viz = res["variables"].get("viz", {})

    print("\n✅ Success! Plan executed successfully\n")

    print("=== WEATHER DATA ===")
    pprint.pprint(weather_data, width=100, sort_dicts=False)

    print("\n=== ANALYSIS ===")
    pprint.pprint(analysis, width=100, sort_dicts=False)

    print("\n=== REPORT ===")
    pprint.pprint(report, width=100, sort_dicts=False)

    print("\n=== VISUALIZATION DATA ===")
    pprint.pprint(viz, width=100, sort_dicts=False)

    try:
        with open("weather_analysis_results.json", "w") as fp:
            json.dump(res["variables"], fp, indent=2, default=str)
        print("\n💾 Results written to weather_analysis_results.json")
    except Exception as e:
        print(f"\n❌ Failed to write results: {e}")


# --------------------------------------------------------------------------- entry point
if __name__ == "__main__":
    asyncio.run(main())