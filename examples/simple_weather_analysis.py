
#!/usr/bin/env python
# examples/simple_weather_analysis.py
"""
Simple Weather Analysis Demo
===========================

This script demonstrates a simple weather analysis workflow without using
the UniversalPlan/UniversalExecutor infrastructure. It performs the same
tasks as universal_executor_demo.py but with a direct implementation:

1. Retrieve weather data for multiple cities
2. Calculate average temperatures and analyze conditions
3. Generate a weather report with recommendations
4. Format visualization data
"""

import asyncio
import json
from typing import Dict, List, Any, Callable

# ---- Custom Tools and Functions ----

async def get_weather_data(locations: List[str]) -> Dict[str, Any]:
    """Retrieves weather data for multiple locations"""
    print(f"\n🌎 Retrieving weather data for multiple locations: {', '.join(locations)}...")
    
    # Mock weather data
    weather_data = {
        "New York": {"temperature": 72, "conditions": "Partly cloudy", "humidity": 65},
        "London": {"temperature": 62, "conditions": "Rainy", "humidity": 80},
        "Tokyo": {"temperature": 78, "conditions": "Sunny", "humidity": 70},
        "Sydney": {"temperature": 68, "conditions": "Clear", "humidity": 60},
        "Cairo": {"temperature": 90, "conditions": "Hot", "humidity": 30}
    }
    
    # Simulate API call delay
    await asyncio.sleep(1)
    
    results = {}
    for location in locations:
        if location in weather_data:
            results[location] = weather_data[location]
        else:
            results[location] = {"temperature": 75, "conditions": "Unknown", "humidity": 50}
    
    return {"results": results}

def analyze_weather_data(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyzes weather data to extract insights"""
    print(f"\n🔍 Analyzing weather data for {len(weather_data['results'])} locations...")
    
    data = weather_data.get("results", {})
    
    if not data:
        return {"error": "No weather data provided"}
    
    # Calculate averages
    total_temp = sum(city_data.get("temperature", 0) for city_data in data.values())
    total_humidity = sum(city_data.get("humidity", 0) for city_data in data.values())
    
    # Count conditions
    conditions = {}
    for city_data in data.values():
        condition = city_data.get("conditions", "Unknown")
        conditions[condition] = conditions.get(condition, 0) + 1
    
    # Calculate averages
    avg_temp = total_temp / len(data) if data else 0
    avg_humidity = total_humidity / len(data) if data else 0
    
    # Find most common condition
    most_common_condition = max(conditions.items(), key=lambda x: x[1])[0] if conditions else "Unknown"
    
    return {
        "average_temperature": round(avg_temp, 1),
        "average_humidity": round(avg_humidity, 1),
        "most_common_condition": most_common_condition,
        "condition_distribution": conditions,
        "locations_analyzed": len(data)
    }

def create_weather_report(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a weather report from analysis results"""
    print("\n📊 Creating weather report from analysis...")
    
    # Check for valid analysis data
    if not analysis or "error" in analysis:
        return {"error": "Invalid analysis data"}
    
    # Create a formatted report
    report = {
        "title": "Global Weather Analysis Report",
        "summary": (
            f"Analysis of {analysis.get('locations_analyzed', 0)} locations shows an "
            f"average temperature of {analysis.get('average_temperature', 0)}°F "
            f"with {analysis.get('average_humidity', 0)}% average humidity. "
            f"The most common condition is {analysis.get('most_common_condition', 'Unknown')}."
        ),
        "details": analysis,
        "recommendations": [
            "Pack for varied weather conditions" if len(analysis.get('condition_distribution', {})) > 2 else "Weather is fairly consistent",
            "Bring rain gear" if "Rainy" in analysis.get('condition_distribution', {}) else "No need for rain gear",
            "Dress for warm weather" if analysis.get('average_temperature', 0) > 75 else "Pack some warmer clothes"
        ]
    }
    
    return report

def format_visualization_data(weather_data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Formats data for visualization"""
    print("\n🎨 Formatting data for visualization...")
    
    # Get city data
    city_data = weather_data.get("results", {})
    
    # Prepare temperature data for a bar chart
    temperatures = []
    for location, data in city_data.items():
        temperatures.append({
            "location": location,
            "temperature": data.get("temperature", 0)
        })
    
    # Sort by temperature
    temperatures.sort(key=lambda x: x["temperature"], reverse=True)
    
    # Prepare condition data for a pie chart
    conditions = []
    for condition, count in analysis.get("condition_distribution", {}).items():
        conditions.append({
            "condition": condition,
            "count": count
        })
    
    return {
        "title": "Global Weather Visualization",
        "temperature_data": temperatures,
        "condition_data": conditions,
        "average_temperature": analysis.get("average_temperature", 0),
        "average_humidity": analysis.get("average_humidity", 0)
    }

# ---- Main Analysis Pipeline ----

async def run_weather_analysis_pipeline(locations: List[str]) -> Dict[str, Any]:
    """Run the complete weather analysis pipeline"""
    # Step 1: Get weather data
    weather_data = await get_weather_data(locations)
    
    # Step 2: Analyze weather data
    analysis = analyze_weather_data(weather_data)
    
    # Step 3 & 4: Create report and visualization data (parallel)
    report_task = asyncio.create_task(asyncio.to_thread(create_weather_report, analysis))
    viz_task = asyncio.create_task(asyncio.to_thread(format_visualization_data, weather_data, analysis))
    
    # Wait for both tasks to complete
    report = await report_task
    viz_data = await viz_task
    
    # Return all results
    return {
        "weather_data": weather_data,
        "analysis": analysis,
        "report": report,
        "visualization": viz_data
    }

# ---- Main Function ----

async def main():
    print("🌤️  Simple Weather Analysis Demo")
    print("================================")
    
    # Define target cities
    target_cities = ["New York", "London", "Tokyo", "Sydney", "Cairo"]
    print(f"\n📍 Target cities: {', '.join(target_cities)}")
    
    # Run the analysis pipeline
    print("\n▶️ Running analysis pipeline...")
    results = await run_weather_analysis_pipeline(target_cities)
    
    # Display the report
    report = results["report"]
    print("\n📝 Weather Report:")
    print(f"Title: {report['title']}")
    print(f"Summary: {report['summary']}")
    print("Recommendations:")
    for rec in report["recommendations"]:
        print(f"- {rec}")
    
    # Display visualization data
    viz_data = results["visualization"]
    print("\n📊 Visualization Data Summary:")
    print(f"Title: {viz_data['title']}")
    print(f"City Temperature Ranking:")
    for city in viz_data["temperature_data"]:
        print(f"- {city['location']}: {city['temperature']}°F")
    
    print(f"Weather Conditions Distribution:")
    for condition in viz_data["condition_data"]:
        print(f"- {condition['condition']}: {condition['count']} cities")
    
    # Save the results to a file
    results_file = "weather_analysis_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")

if __name__ == "__main__":
    asyncio.run(main())