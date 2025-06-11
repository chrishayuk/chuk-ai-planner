#!/usr/bin/env python
# examples/advanced_universal_executor_demo.py
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
from types import MappingProxyType

from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor

# ---- JSON Serialization Helper ----
def make_json_serializable(obj: Any) -> Any:
    """Convert potentially frozen data structures to JSON-serializable format."""
    try:
        # Try to import _ReadOnlyList if it exists
        from chuk_ai_planner.models.base import _ReadOnlyList
    except ImportError:
        # If not available, create a dummy class that will never match
        class _ReadOnlyList:
            pass
    
    if isinstance(obj, MappingProxyType):
        # Convert MappingProxyType to regular dict
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, _ReadOnlyList):
        # Convert _ReadOnlyList to regular list
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        # Handle nested dicts that might contain frozen structures
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Handle regular lists and tuples
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, frozenset):
        # Convert frozensets to lists for JSON compatibility
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__iter__') and hasattr(obj, '__getitem__') and hasattr(obj, '__len__'):
        # This catches other list-like objects, but we need to be careful not to catch strings or dicts
        if isinstance(obj, (str, bytes, dict)):
            # These are iterable but should not be converted to lists
            return obj
        else:
            # It's a list-like object, convert to list
            try:
                return [make_json_serializable(item) for item in obj]
            except (TypeError, AttributeError):
                # If iteration fails, return as-is
                return obj
    else:
        # Primitive types are already JSON serializable
        return obj

# ---- Simple Plan Registry (In-Memory) ----
class SimplePlanRegistry:
    """Simple in-memory plan registry that avoids disk serialization issues."""
    
    def __init__(self):
        self.plans: Dict[str, UniversalPlan] = {}
    
    def register_plan(self, plan: UniversalPlan) -> str:
        """Register a plan and return its ID."""
        plan_id = plan.id
        self.plans[plan_id] = plan
        return plan_id
    
    def get_plan(self, plan_id: str) -> Optional[UniversalPlan]:
        """Retrieve a plan by ID."""
        return self.plans.get(plan_id)
    
    def list_plans(self) -> List[str]:
        """List all registered plan IDs."""
        return list(self.plans.keys())

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

def clean_data_function(**kwargs) -> Dict[str, Any]:
    """Function that cleans and validates data"""
    print(f"🧹 clean_data_function received kwargs: {list(kwargs.keys())}")
    
    source_data = kwargs.get("data", {})
    print(f"🧹 source_data type: {type(source_data)}")
    print(f"🧹 source_data content: {source_data}")
    
    # Handle case where source_data might be a string (shouldn't happen but let's be safe)
    if isinstance(source_data, str):
        print("🧹 WARNING: source_data is a string, expected dict")
        try:
            import json
            source_data = json.loads(source_data)
            print("🧹 Successfully parsed as JSON")
        except json.JSONDecodeError:
            print("🧹 Failed to parse as JSON")
            return {"error": "Invalid source data format", "cleaned_data": []}
    
    if not isinstance(source_data, dict):
        print(f"🧹 ERROR: Expected dict, got {type(source_data)}")
        return {"error": "Invalid source data format", "cleaned_data": []}
    
    data = source_data.get("data", [])
    source_info = source_data.get("source_info", {})
    print(f"🧹 Cleaning data from {source_info.get('name', 'unknown source')}...")
    
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
    
    print(f"🧹 Cleaned {len(cleaned_data)} items (removed {removed_count})")
    
    return {
        "cleaned_data": cleaned_data,
        "original_count": len(data),
        "cleaned_count": len(cleaned_data),
        "removed_count": removed_count,
        "source_info": source_info
    }

def analyze_function(**kwargs) -> Dict[str, Any]:
    """Function that analyzes data and generates insights"""
    print(f"🔍 analyze_function received kwargs: {list(kwargs.keys())}")
    
    cleaned_data_obj = kwargs.get("cleaned_data", {})
    data_type = kwargs.get("data_type", "unknown")
    
    print(f"🔍 cleaned_data_obj type: {type(cleaned_data_obj)}")
    print(f"🔍 data_type: {data_type}")
    
    # Handle case where cleaned_data_obj might be a string
    if isinstance(cleaned_data_obj, str):
        print("🔍 WARNING: cleaned_data_obj is a string, expected dict")
        try:
            import json
            cleaned_data_obj = json.loads(cleaned_data_obj)
            print("🔍 Successfully parsed as JSON")
        except json.JSONDecodeError:
            print("🔍 Failed to parse as JSON")
            return {"error": "Invalid cleaned data format", "insights": []}
    
    # Extract the actual cleaned_data list from the object
    if isinstance(cleaned_data_obj, dict):
        cleaned_data = cleaned_data_obj.get("cleaned_data", [])
    else:
        print(f"🔍 ERROR: Expected dict, got {type(cleaned_data_obj)}")
        return {"error": "Invalid cleaned data format", "insights": []}
    
    print(f"🔍 Extracted cleaned_data type: {type(cleaned_data)}")
    print(f"🔍 Analyzing {data_type} data ({len(cleaned_data) if isinstance(cleaned_data, list) else 'unknown count'} items)...")
    
    if not isinstance(cleaned_data, list):
        print(f"🔍 ERROR: Expected list for cleaned_data, got {type(cleaned_data)}")
        return {"error": "Invalid cleaned data format", "insights": []}
    
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

def generate_report_function(**kwargs) -> Dict[str, Any]:
    """Function that generates a report from analysis results"""
    analysis_results = kwargs.get("analysis_results", [])
    print(f"\n📝 Generating report from analysis results...")
    print(f"📝 Received analysis_results type: {type(analysis_results)}")
    print(f"📝 Received analysis_results content: {analysis_results}")
    
    # Handle case where analysis_results might be a string
    if isinstance(analysis_results, str):
        print("📝 WARNING: analysis_results is a string, expected list/dict")
        try:
            import json
            analysis_results = json.loads(analysis_results)
            print("📝 Successfully parsed as JSON")
        except json.JSONDecodeError:
            print("📝 Failed to parse as JSON, returning error")
            return {"error": "Invalid analysis results format", "received": str(analysis_results)}
    
    if not analysis_results:
        return {"error": "No analysis results to report on"}
    
    # Ensure analysis_results is a list
    if not isinstance(analysis_results, list):
        analysis_results = [analysis_results]
    
    # Combine insights from all analyses
    all_insights = []
    summaries = []
    
    for result in analysis_results:
        if not isinstance(result, dict):
            print(f"📝 WARNING: Expected dict, got {type(result)}: {result}")
            continue
            
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
            for result in analysis_results if isinstance(result, dict)
        ]
    }
    
    return report

# ---- Plan Creation Functions ----

def create_data_processing_subplan(data_type: str) -> UniversalPlan:
    """Create a subplan for processing a specific type of data"""
    # Create subplan with its own graph store to avoid conflicts
    plan = UniversalPlan(
        title=f"{data_type.capitalize()} Data Processing",
        description=f"Retrieve, clean, and analyze {data_type} data",
        tags=[data_type, "processing", "subplan"]
        # Don't pass executor.graph_store - let it create its own
    )
    
    # Add metadata
    plan.add_metadata("created_by", "advanced_executor_demo")
    plan.add_metadata("data_type", data_type)
    
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
            "cleaned_data": "${cleaned_data}",  # Pass the whole cleaned_data object
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
    
    # Save the plan
    plan.save()
    
    return plan

def create_main_plan(subplan_ids: Dict[str, str]) -> UniversalPlan:
    """Create the main plan that coordinates subplans"""
    # Create main plan with its own graph store
    plan = UniversalPlan(
        title="Multi-Source Data Analysis",
        description="Coordinate processing of multiple data sources and generate a comprehensive report",
        tags=["main", "multi-source", "report"]
        # Don't pass executor.graph_store - let it create its own
    )
    
    # Add metadata
    plan.add_metadata("created_by", "advanced_executor_demo")
    plan.add_metadata("priority", "high")
    
    # Track the subplan step IDs
    subplan_step_ids = {}
    
    # Add steps for each subplan - use add_tool_step with subplan tool
    for data_type, subplan_id in subplan_ids.items():
        step_id = plan.add_tool_step(
            title=f"Process {data_type} data",
            tool="subplan",
            args={"plan_id": subplan_id, "args": {}},
            result_variable=f"{data_type}_result"
        )
        subplan_step_ids[data_type] = step_id
    
    # Add step to combine analysis results - use proper dependency format
    combine_step_id = plan.add_function_step(
        title="Combine analysis results",
        function="combine_results",
        args={
            "weather_result": "${weather_result}",
            "stocks_result": "${stocks_result}",
            "news_result": "${news_result}"
        },
        result_variable="combined_results",
        depends_on=[subplan_step_ids["weather"], subplan_step_ids["stocks"], subplan_step_ids["news"]]
    )
    
    # Add step to generate report
    report_step_id = plan.add_function_step(
        title="Generate comprehensive report",
        function="generate_report",
        args={"analysis_results": "${combined_results}"},
        result_variable="final_report",
        depends_on=[combine_step_id]
    )
    
    # Save the plan to ensure all steps are properly indexed
    plan.save()
    
    return plan

# ---- Helper Functions for Execution ----

def add_metadata_function(**kwargs) -> Dict[str, Any]:
    """Function that adds metadata to a data object"""
    print(f"📝 add_metadata_function received kwargs: {list(kwargs.keys())}")
    
    data = kwargs.get("data", {})
    metadata = kwargs.get("metadata", {})
    
    print(f"📝 data type: {type(data)}")
    print(f"📝 metadata type: {type(metadata)}")
    
    # Handle case where data might be a string
    if isinstance(data, str):
        print("📝 WARNING: data is a string, expected dict")
        try:
            import json
            data = json.loads(data)
            print("📝 Successfully parsed as JSON")
        except json.JSONDecodeError:
            print("📝 Failed to parse as JSON, treating as string value")
            data = {"value": data}
    
    # Create a new dictionary with combined data
    result = data.copy() if isinstance(data, dict) else {"data": data}
    
    # Add each metadata item
    for key, value in metadata.items():
        result[key] = value
    
    print(f"📝 add_metadata result keys: {list(result.keys())}")
    
    return result

def combine_results_function(**kwargs) -> List[Dict[str, Any]]:
    """Function that combines results from multiple analyses"""
    print("\n🔄 Combining results from multiple analyses...")
    print(f"🔄 Received kwargs: {list(kwargs.keys())}")
    
    # Get all the results
    results = []
    
    for key, value in kwargs.items():
        print(f"🔄 Processing {key}: {type(value)}")
        if key.endswith("_result") and value:
            print(f"🔄 Adding result from {key}")
            results.append(value)
    
    print(f"🔄 Combined {len(results)} results")
    return results

# ---- Subplan Execution Tool ----

async def subplan_execution_tool(args: Dict[str, Any], registry: SimplePlanRegistry, executor: UniversalExecutor) -> Dict[str, Any]:
    """Tool that executes a subplan by ID"""
    plan_id = args.get("plan_id")
    subplan_args = args.get("args", {})
    
    print(f"\n🔄 Executing subplan: {plan_id}")
    
    # Get the subplan from registry
    subplan = registry.get_plan(plan_id)
    if not subplan:
        raise ValueError(f"Subplan {plan_id} not found in registry")
    
    # Copy subplan graph into executor's graph store if necessary
    if subplan.graph is not executor.graph_store:
        print(f"🔄 Copying subplan graph nodes and edges...")
        for node in subplan.graph.nodes.values():
            if node.id not in executor.graph_store.nodes:
                executor.graph_store.add_node(node)
        for edge in subplan.graph.edges:
            # Check if edge already exists to avoid duplicates
            existing_edges = executor.graph_store.get_edges(src=edge.src, dst=edge.dst, kind=edge.kind)
            if not any(e.id == edge.id for e in existing_edges):
                executor.graph_store.add_edge(edge)
    
    # Execute the subplan with a clean variable context
    result = await executor.execute_plan(subplan, variables=subplan_args)
    
    if not result["success"]:
        raise RuntimeError(f"Subplan execution failed: {result.get('error', 'Unknown error')}")
    
    print(f"🔄 Subplan {plan_id} completed. Variables: {list(result['variables'].keys())}")
    
    # Return the variables from the subplan execution
    # But we need to return the final_result specifically since that's what the main plan expects
    if "final_result" in result["variables"]:
        return result["variables"]["final_result"]
    else:
        # If no final_result, return all variables
        return result["variables"]

# ---- Main Function ----

async def main():
    print("🚀 Advanced Universal Executor Demo")
    print("==================================")
    
    # Create a registry for plans
    registry = SimplePlanRegistry()
    
    # Create the executor first
    executor = UniversalExecutor()
    
    # Register basic tools and functions first
    print("\n🔧 Registering tools and functions...")
    executor.register_tool("data_source", data_source_tool)
    executor.register_function("clean_data", clean_data_function)
    executor.register_function("analyze", analyze_function)
    executor.register_function("add_metadata", add_metadata_function)
    executor.register_function("combine_results", combine_results_function)
    executor.register_function("generate_report", generate_report_function)
    
    # Create and register the subplan execution tool with closure
    async def subplan_tool(args):
        return await subplan_execution_tool(args, registry, executor)
    
    executor.register_tool("subplan", subplan_tool)
    
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
    
    # Execute the main plan
    print("\n▶️ Executing the main plan...")
    print("🔍 Debug: Starting execution...")
    
    # Let's manually check what steps exist before execution
    print("\n🔍 Debug: Checking plan steps before execution...")
    steps = [node for node in executor.graph_store.nodes.values() 
             if node.kind.value == "plan_step"]
    print(f"🔍 Found {len(steps)} plan steps in graph store")
    
    for step in steps:
        step_title = step.data.get("description", "No title")
        step_index = step.data.get("index", "No index")
        print(f"🔍 Step {step_index}: {step_title} (ID: {step.id[:8]})")
        
        # Check for tool calls linked to this step
        tool_calls = []
        for edge in executor.graph_store.get_edges(src=step.id):
            if edge.kind.value == "plan_link":
                tool_node = executor.graph_store.get_node(edge.dst)
                if tool_node and tool_node.kind.value == "tool_call":
                    tool_name = tool_node.data.get("name", "unknown")
                    tool_calls.append(tool_name)
        print(f"🔍   Tool calls: {tool_calls}")
        
        # Check dependencies
        dependencies = []
        for edge in executor.graph_store.get_edges(dst=step.id):
            if edge.kind.value == "step_order":
                dep_node = executor.graph_store.get_node(edge.src)
                if dep_node:
                    dependencies.append(dep_node.data.get("index", "unknown"))
        print(f"🔍   Dependencies: {dependencies}")
    
    results = await executor.execute_plan(main_plan)
    print(f"🔍 Debug: Execution completed. Success: {results['success']}")
    print(f"🔍 Debug: Available variables: {list(results['variables'].keys())}")
    for var_name, var_value in results['variables'].items():
        print(f"🔍   {var_name}: {type(var_value)}")
    
    # Check execution success
    if not results["success"]:
        print(f"\n❌ Plan execution failed: {results.get('error', 'Unknown error')}")
        return
    
    print("\n✅ Plan executed successfully!")
    
    # Display the report
    if "final_report" in results["variables"]:
        report = results["variables"]["final_report"]
        
        # Check if report is an error result
        if isinstance(report, dict) and "error" in report:
            print(f"\n❌ Report generation failed: {report['error']}")
            print(f"📝 Received data: {report.get('received', 'N/A')}")
            return
        
        print("\n📝 Multi-Source Data Analysis Report:")
        print(f"Title: {report.get('title', 'No title')}")
        print(f"Time: {report.get('timestamp', 'No timestamp')}")
        print("\nSummary:")
        print(report.get('summary', 'No summary available'))
        
        print("\nSections:")
        for section in report.get('sections', []):
            print(f"- {section.get('title', 'Untitled section')}")
            print(f"  {section.get('content', 'No content')}")
    else:
        print("\n❌ No final report found in results")
        print("Available variables:")
        for key, value in results["variables"].items():
            print(f"- {key}: {type(value)}")
    
    # Save the results to a file (only if we have a valid report)
    if "final_report" in results["variables"] and isinstance(results["variables"]["final_report"], dict) and "title" in results["variables"]["final_report"]:
        results_file = "multi_source_analysis_results.json"
        with open(results_file, "w") as f:
            json.dump(make_json_serializable(results["variables"]["final_report"]), f, indent=2)
        print(f"\n💾 Results saved to {results_file}")
    else:
        print("\n⚠️  Report generation failed, results not saved")

if __name__ == "__main__":
    asyncio.run(main())