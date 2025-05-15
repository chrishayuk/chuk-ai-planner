#!/usr/bin/env python
# examples/advanced_executor_demo.py
"""
Advanced Universal Executor Demo
===============================

This script demonstrates more advanced features of the UniversalPlan and
UniversalExecutor, including:

1. Using subplans (plans that call other plans)
2. Complex variable substitution
3. Dynamic tool/function registration
4. Error handling and recovery
5. Conditional execution based on results

The demo implements a multi-stage data processing pipeline:
- Retrieve data from multiple sources
- Clean and validate the data
- Analyze the data and generate insights
- Create reports and visualizations
"""

import asyncio
import json
import os
import random
import time
from typing import Dict, List, Any, Optional

from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor
from chuk_ai_planner.planner.plan_registry import PlanRegistry

# ---- Mock Data Sources and Processors ----

async def data_source_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Tool that retrieves data from a specified source"""
    source = args.get("source", "unknown")
    print(f"\n📥 Retrieving data from source: {source}")
    
    # Simulate API call delay
    await asyncio.sleep(0.3)
    
    # Generate random mock data based on source
    if source == "weather":
        return {
            "data": [
                {"city": "New York", "temperature": random.randint(60, 85), "conditions": random.choice(["Sunny", "Cloudy", "Rainy"])},
                {"city": "London", "temperature": random.randint(50, 70), "conditions": random.choice(["Cloudy", "Rainy", "Foggy"])},
                {"city": "Tokyo", "temperature": random.randint(65, 90), "conditions": random.choice(["Sunny", "Cloudy", "Rainy"])},
                {"city": "Sydney", "temperature": random.randint(65, 85), "conditions": random.choice(["Sunny", "Partly Cloudy", "Clear"])}
            ],
            "source_info": {"name": "Global Weather API", "timestamp": time.time()}
        }
    elif source == "stocks":
        return {
            "data": [
                {"symbol": "AAPL", "price": round(random.uniform(170, 190), 2), "change": round(random.uniform(-5, 5), 2)},
                {"symbol": "MSFT", "price": round(random.uniform(340, 360), 2), "change": round(random.uniform(-8, 8), 2)},
                {"symbol": "GOOGL", "price": round(random.uniform(120, 140), 2), "change": round(random.uniform(-4, 4), 2)},
                {"symbol": "AMZN", "price": round(random.uniform(140, 160), 2), "change": round(random.uniform(-6, 6), 2)}
            ],
            "source_info": {"name": "Stock Market API", "timestamp": time.time()}
        }
    elif source == "news":
        return {
            "data": [
                {"headline": "New Technology Breakthrough Announced", "category": "Technology", "sentiment": random.choice(["Positive", "Neutral", "Negative"])},
                {"headline": "Global Markets React to Economic Data", "category": "Finance", "sentiment": random.choice(["Positive", "Neutral", "Negative"])},
                {"headline": "Scientists Discover New Species", "category": "Science", "sentiment": random.choice(["Positive", "Neutral", "Negative"])},
                {"headline": "Sports Team Wins Championship", "category": "Sports", "sentiment": random.choice(["Positive", "Neutral", "Negative"])}
            ],
            "source_info": {"name": "Global News API", "timestamp": time.time()}
        }
    else:
        # Generic data for unknown sources
        return {
            "data": [{"item": f"Sample data item {i+1}"} for i in range(4)],
            "source_info": {"name": f"{source.capitalize()} API", "timestamp": time.time()}
        }

def clean_data_function(args: Dict[str, Any]) -> Dict[str, Any]:
    """Function that cleans and validates data"""
    source_data = args.get("data", {})
    data = source_data.get("data", [])
    source_info = source_data.get("source_info", {})
    print(f"\n🧹 Cleaning data from {source_info.get('name', 'unknown source')}...")
    
    if not data:
        return {"error": "No data to clean", "cleaned_data": []}
    
    # Simple cleaning: remove items with missing values
    cleaned_data = []
    removed_count = 0
    
    for item in data:
        # Check if any value is None or empty string
        has_empty = any(v is None or v == "" for v in item.values())
        if not has_empty:
            cleaned_data.append(item)
        else:
            removed_count += 1
    
    return {
        "cleaned_data": cleaned_data,
        "original_count": len(data),
        "cleaned_count": len(cleaned_data),
        "removed_count": removed_count,
        "source_info": source_info
    }

def analyze_function(args: Dict[str, Any]) -> Dict[str, Any]:
    """Function that analyzes data and generates insights"""
    cleaned_data = args.get("cleaned_data", [])
    data_type = args.get("data_type", "unknown")
    print(f"\n🔍 Analyzing {data_type} data ({len(cleaned_data)} items)...")
    
    if not cleaned_data:
        return {"error": "No data to analyze", "insights": []}
    
    # Different analysis based on data type
    if data_type == "weather":
        # Calculate average temperature and condition counts
        total_temp = sum(item.get("temperature", 0) for item in cleaned_data)
        avg_temp = total_temp / len(cleaned_data) if cleaned_data else 0
        
        conditions = {}
        for item in cleaned_data:
            condition = item.get("conditions", "Unknown")
            conditions[condition] = conditions.get(condition, 0) + 1
        
        return {
            "insights": [
                {"type": "average", "metric": "temperature", "value": round(avg_temp, 1)},
                {"type": "distribution", "metric": "conditions", "value": conditions}
            ],
            "summary": f"Average temperature: {round(avg_temp, 1)}°F across {len(cleaned_data)} cities"
        }
    
    elif data_type == "stocks":
        # Calculate average price change and find biggest mover
        total_change = sum(item.get("change", 0) for item in cleaned_data)
        avg_change = total_change / len(cleaned_data) if cleaned_data else 0
        
        # Find biggest gainer and loser
        biggest_gainer = max(cleaned_data, key=lambda x: x.get("change", 0))
        biggest_loser = min(cleaned_data, key=lambda x: x.get("change", 0))
        
        return {
            "insights": [
                {"type": "average", "metric": "price_change", "value": round(avg_change, 2)},
                {"type": "extreme", "metric": "biggest_gainer", "value": biggest_gainer},
                {"type": "extreme", "metric": "biggest_loser", "value": biggest_loser}
            ],
            "summary": f"Average price change: {round(avg_change, 2)}%, Biggest gainer: {biggest_gainer['symbol']}"
        }
    
    elif data_type == "news":
        # Analyze sentiment distribution
        sentiments = {}
        categories = {}
        
        for item in cleaned_data:
            sentiment = item.get("sentiment", "Unknown")
            category = item.get("category", "Unknown")
            
            sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
            categories[category] = categories.get(category, 0) + 1
        
        return {
            "insights": [
                {"type": "distribution", "metric": "sentiment", "value": sentiments},
                {"type": "distribution", "metric": "categories", "value": categories}
            ],
            "summary": f"Analyzed {len(cleaned_data)} news items across {len(categories)} categories"
        }
    
    else:
        # Generic analysis for unknown data types
        return {
            "insights": [
                {"type": "count", "metric": "items", "value": len(cleaned_data)}
            ],
            "summary": f"Processed {len(cleaned_data)} items of {data_type} data"
        }

def generate_report_function(args: Dict[str, Any]) -> Dict[str, Any]:
    """Function that generates a report from analysis results"""
    analysis_results = args.get("analysis_results", [])
    print("\n📝 Generating report from analysis results...")
    
    if not analysis_results:
        return {"error": "No analysis results to report on"}
    
    # Combine insights from all analyses
    all_insights = []
    summaries = []
    
    for result in analysis_results:
        data_type = result.get("data_type", "Unknown")
        insights = result.get("insights", [])
        summary = result.get("summary", "")
        
        all_insights.append({"data_type": data_type, "insights": insights})
        summaries.append(f"{data_type}: {summary}")
    
    # Create the report
    report = {
        "title": "Multi-Source Data Analysis Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": "\n".join(summaries),
        "insights": all_insights,
        "sections": [
            {"title": f"Analysis of {result.get('data_type', 'Unknown')} Data", 
             "content": result.get("summary", "")}
            for result in analysis_results
        ]
    }
    
    return report

# ---- Plan Creation Functions ----

def create_data_processing_subplan(data_type: str) -> UniversalPlan:
    """Create a subplan for processing a specific type of data"""
    plan = UniversalPlan(
        title=f"{data_type.capitalize()} Data Processing",
        description=f"Retrieve, clean, and analyze {data_type} data",
        tags=[data_type, "processing", "subplan"]
    )
    
    # Add metadata
    plan.add_metadata("created_by", "advanced_executor_demo")
    plan.add_metadata("data_type", data_type)
    
    # Save the plan first to get an ID
    plan_id = plan.save()
    
    # Add steps
    # 1. Retrieve data
    retrieve_step_id = plan.add_tool_step(
        title=f"Retrieve {data_type} data",
        tool="data_source",
        args={"source": data_type},
        result_variable="source_data"
    )
    
    # 2. Clean data
    clean_step_id = plan.add_function_step(
        title=f"Clean {data_type} data",
        function="clean_data",
        args={"data": "${source_data}"},
        result_variable="cleaned_data",
        depends_on=[retrieve_step_id]
    )
    
    # 3. Analyze data
    analyze_step_id = plan.add_function_step(
        title=f"Analyze {data_type} data",
        function="analyze",
        args={
            "cleaned_data": "${cleaned_data.cleaned_data}",
            "data_type": data_type
        },
        result_variable="analysis_result",
        depends_on=[clean_step_id]
    )
    
    # Add data type to analysis result for identification in parent plan
    plan.add_function_step(
        title="Add metadata to result",
        function="add_metadata",
        args={
            "data": "${analysis_result}",
            "metadata": {"data_type": data_type}
        },
        result_variable="final_result",
        depends_on=[analyze_step_id]
    )
    
    # Save the plan again after adding steps
    plan.save()
    
    return plan

def create_main_plan(subplan_ids: Dict[str, str]) -> UniversalPlan:
    """Create the main plan that coordinates subplans"""
    plan = UniversalPlan(
        title="Multi-Source Data Analysis",
        description="Coordinate processing of multiple data sources and generate a comprehensive report",
        tags=["main", "multi-source", "report"]
    )
    
    # Add metadata
    plan.add_metadata("created_by", "advanced_executor_demo")
    plan.add_metadata("priority", "high")
    
    # Track the subplan step IDs
    subplan_step_ids = {}
    
    # Add steps for each subplan
    for data_type, subplan_id in subplan_ids.items():
        step_id = plan.add_plan_step(
            title=f"Process {data_type} data",
            plan_id=subplan_id,
            args={},
            result_variable=f"{data_type}_result"
        )
        subplan_step_ids[data_type] = step_id
    
    # Add step to combine analysis results
    combine_step_id = plan.add_function_step(
        title="Combine analysis results",
        function="combine_results",
        args={
            "weather_result": "${weather_result.final_result}",
            "stocks_result": "${stocks_result.final_result}",
            "news_result": "${news_result.final_result}"
        },
        result_variable="combined_results",
        depends_on=list(subplan_step_ids.values())
    )
    
    # Add step to generate report
    report_step_id = plan.add_function_step(
        title="Generate comprehensive report",
        function="generate_report",
        args={"analysis_results": "${combined_results}"},
        result_variable="final_report",
        depends_on=[combine_step_id]
    )
    
    return plan

# ---- Helper Functions for Execution ----

def add_metadata_function(args: Dict[str, Any]) -> Dict[str, Any]:
    """Function that adds metadata to a data object"""
    data = args.get("data", {})
    metadata = args.get("metadata", {})
    
    # Create a new dictionary with combined data
    result = data.copy() if isinstance(data, dict) else {"data": data}
    
    # Add each metadata item
    for key, value in metadata.items():
        result[key] = value
    
    return result

def combine_results_function(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Function that combines results from multiple analyses"""
    print("\n🔄 Combining results from multiple analyses...")
    
    # Get all the results
    results = []
    
    for key, value in args.items():
        if key.endswith("_result") and value:
            results.append(value)
    
    return results

# ---- Main Function ----

async def main():
    print("🚀 Advanced Universal Executor Demo")
    print("==================================")
    
    # Create a registry for plans
    registry = PlanRegistry(storage_dir="temp_registry")
    
    # Create and register subplans
    print("\n📋 Creating and registering subplans...")
    subplan_ids = {}
    
    for data_type in ["weather", "stocks", "news"]:
        subplan = create_data_processing_subplan(data_type)
        subplan_id = registry.register_plan(subplan)
        subplan_ids[data_type] = subplan_id
        print(f"- Created {data_type} processing subplan (ID: {subplan_id[:8]})")
    
    # Create the main plan
    main_plan = create_main_plan(subplan_ids)
    main_plan_id = registry.register_plan(main_plan)
    print(f"- Created main plan (ID: {main_plan_id[:8]})")
    
    # Display the main plan
    print("\n📋 Main Plan Structure:")
    print(main_plan.outline())
    
    # Create the executor
    executor = UniversalExecutor()
    
    # Register all necessary tools and functions
    print("\n🔧 Registering tools and functions...")
    executor.register_tool("data_source", data_source_tool)
    executor.register_function("clean_data", clean_data_function)
    executor.register_function("analyze", analyze_function)
    executor.register_function("add_metadata", add_metadata_function)
    executor.register_function("combine_results", combine_results_function)
    executor.register_function("generate_report", generate_report_function)
    
    # Execute the main plan
    print("\n▶️ Executing the main plan...")
    results = await executor.execute_plan(main_plan)
    
    # Check execution success
    if not results["success"]:
        print(f"\n❌ Plan execution failed: {results.get('error', 'Unknown error')}")
        return
    
    print("\n✅ Plan executed successfully!")
    
    # Display the report
    if "final_report" in results["variables"]:
        report = results["variables"]["final_report"]
        print("\n📝 Multi-Source Data Analysis Report:")
        print(f"Title: {report['title']}")
        print(f"Time: {report['timestamp']}")
        print("\nSummary:")
        print(report['summary'])
        
        print("\nSections:")
        for section in report['sections']:
            print(f"- {section['title']}")
            print(f"  {section['content']}")
    
    # Save the results to a file
    results_file = "multi_source_analysis_results.json"
    with open(results_file, "w") as f:
        json.dump(results["variables"]["final_report"], f, indent=2)
    print(f"\n💾 Results saved to {results_file}")
    
    # Clean up registry
    os.system("rm -rf temp_registry")

if __name__ == "__main__":
    asyncio.run(main())