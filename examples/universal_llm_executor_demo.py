#!/usr/bin/env python
# examples/universal_llm_executor_demo.py
"""
Universal LLM-to-Execution Demo
===============================

End-to-end demo that showcases:
1. Converting a natural language task to a JSON plan via LLM
2. Creating a UniversalPlan from the LLM-generated JSON
3. Executing the plan with the enhanced UniversalExecutor

This demonstrates the complete workflow from natural language request
to executed plan with results.
"""

import argparse
import asyncio
import json
import os
import pprint
import uuid
from typing import Any, Dict, List, Optional

# Import the official UniversalPlan implementation
from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor

# Import necessary graph components
from chuk_ai_planner.models import ToolCall
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind
from chuk_ai_planner.store.memory import InMemoryGraphStore

# For LLM simulation or live LLM calls
from dotenv import load_dotenv
load_dotenv()

# Try to import OpenAI (optional)
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# -------------------------------------------------------------------- Mock Tool Implementations
async def weather_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get weather for a location."""
    print(f"📍 Getting weather for: {args.get('location', 'Unknown')}")
    
    # Sample weather data
    weather_data = {
        "New York": {"temperature": 72, "conditions": "Partly cloudy", "humidity": 65},
        "London":   {"temperature": 62, "conditions": "Rainy",          "humidity": 80},
        "Tokyo":    {"temperature": 78, "conditions": "Sunny",          "humidity": 70},
        "Sydney":   {"temperature": 68, "conditions": "Clear",          "humidity": 60},
        "Cairo":    {"temperature": 90, "conditions": "Hot",            "humidity": 30},
    }
    
    location = args.get("location", "Unknown")
    result = weather_data.get(location, {"temperature": 75, "conditions": "Unknown", "humidity": 50})
    
    return result


async def calculator_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Perform a calculation."""
    operation = args.get("operation")
    a = float(args.get("a", 0))
    b = float(args.get("b", 0))
    
    print(f"🧮 Calculating: {a} {operation} {b}")
    
    result = 0
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b != 0:
            result = a / b
        else:
            return {"error": "Division by zero"}
    else:
        return {"error": f"Unknown operation: {operation}"}
    
    return {"result": result}


async def search_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulated search tool."""
    query = args.get("query", "")
    print(f"🔍 Searching for: {query}")
    
    # Simulated search results
    results = [
        {
            "title": f"Result for {query} - Example.com",
            "snippet": f"This is a top result for {query} with high relevance.",
            "url": f"https://example.com/search?q={query.replace(' ', '+')}"
        },
        {
            "title": f"{query} - Complete Guide - Tutorial Site",
            "snippet": f"Learn everything about {query} with our comprehensive guide.",
            "url": f"https://tutorial-site.com/{query.replace(' ', '-')}"
        },
        {
            "title": f"{query} Research - Academic Journal",
            "snippet": f"Recent research papers on {query} and related topics.",
            "url": f"https://academic-journal.org/research/{query.replace(' ', '_')}"
        }
    ]
    
    return {"results": results}


async def coffee_tool(args: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    """Generic coffee tool implementation."""
    print(f"☕ {tool_name}: {args}")
    
    coffee_actions = {
        "grind_beans": "Beans ground successfully to a medium-fine consistency",
        "boil_water": "Water boiled to 200°F, perfect for brewing coffee",
        "brew_coffee": "Coffee brewed perfectly, strong aroma and rich taste",
        "clean_station": "Coffee station cleaned and ready for next use"
    }
    
    message = coffee_actions.get(tool_name, f"Unknown coffee action: {tool_name}")
    return {"status": "success", "message": message}


# -------------------------------------------------------------------- LLM helpers
LLM_SYSTEM_MSG = (
    "You are an assistant that converts a natural-language task into a JSON "
    "plan. Return ONLY valid JSON!\n"
    "Schema:\n"
    "{\n"
    '  "title": str,\n'
    '  "steps": [              // ordered list\n'
    '    {"title": str, "tool": str, "args": {}, "depends_on": [indices]},\n'
    "    ...\n"
    "  ]\n"
    "}\n"
    "Indices start at 1 in the final output.\n"
    "Available tools: weather, calculator, search, grind_beans, boil_water, brew_coffee, clean_station\n"
    "The 'tool' field is required and must be one of the available tools.\n"
    "The 'args' field should contain the appropriate arguments for the tool."
)


async def call_llm_live(task: str) -> Dict[str, Any]:
    """Call the OpenAI API to generate a plan."""
    if not AsyncOpenAI:
        raise RuntimeError("openai package not installed")
    
    try:
        client = AsyncOpenAI()
        print("📡 Calling OpenAI API to generate plan...")
        
        # Detailed system message for better plan generation
        system_message = (
            "You are an assistant that converts a natural-language task into a JSON "
            "plan. Return ONLY valid JSON!\n\n"
            "Schema:\n"
            "{\n"
            '  "title": str,\n'
            '  "steps": [              // ordered list\n'
            '    {"title": str, "tool": str, "args": {}, "depends_on": [indices]},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "1. Indices start at 1 in the final output\n"
            "2. The 'depends_on' field should list step indices (numbers) that must complete before this step\n"
            "3. Make sure all JSON is valid with proper syntax and no trailing commas\n"
            "4. Include ONLY the JSON in your response, nothing else\n\n"
            "Available tools and their arguments:\n"
            "- weather: {\"location\": \"city name\"}\n"
            "- calculator: {\"operation\": \"add|subtract|multiply|divide\", \"a\": number, \"b\": number}\n"
            "- search: {\"query\": \"search query string\"}\n"
            "- grind_beans: {}\n"
            "- boil_water: {}\n"
            "- brew_coffee: {}\n"
            "- clean_station: {}\n"
        )
        
        # Call the API
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,  # Lower temperature for more structured output
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": task},
            ],
        )
        
        # Parse the response
        content = resp.choices[0].message.content.strip()
        
        # Try to extract JSON if there's any surrounding text
        try:
            import re
            json_match = re.search(r'```(?:json)?(.*?)```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
            
            # Another common pattern is JSON without code blocks
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
                
            return json.loads(content)
        except (json.JSONDecodeError, AttributeError):
            # If we can't extract JSON, try to parse the whole response
            return json.loads(content)
            
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {e}")
        print("📋 Falling back to simulated response...")
        # Fall back to simulation if API call fails
        return await call_llm_sim(task)


async def call_llm_sim(task: str) -> Dict[str, Any]:
    """
    Simulation that returns a fixed plan (no actual LLM call)
    """
    print("🤖 Simulating LLM response (no API call)...")
    print(f"   Task: {task}")
    print("   LLM system prompt:", LLM_SYSTEM_MSG[:100] + "...")
    
    return {
        "title": "Coffee, Weather, Calculation, and Search",
        "steps": [
            {
                "title": "Grind coffee beans",
                "tool": "grind_beans",
                "args": {},
                "depends_on": []
            },
            {
                "title": "Boil water",
                "tool": "boil_water",
                "args": {},
                "depends_on": []
            },
            {
                "title": "Brew coffee",
                "tool": "brew_coffee",
                "args": {},
                "depends_on": [1, 2]
            },
            {
                "title": "Check weather in New York",
                "tool": "weather",
                "args": {"location": "New York"},
                "depends_on": [3]
            },
            {
                "title": "Multiply 235.5 × 18.75",
                "tool": "calculator",
                "args": {"operation": "multiply", "a": 235.5, "b": 18.75},
                "depends_on": [3]
            },
            {
                "title": "Search for climate-adaptation info",
                "tool": "search",
                "args": {"query": "climate change adaptation"},
                "depends_on": [4, 5]
            },
            {
                "title": "Clean coffee station",
                "tool": "clean_station",
                "args": {},
                "depends_on": [3]
            }
        ]
    }


# -------------------------------------------------------------------- Plan conversion
def convert_to_universal_plan(llm_json: Dict[str, Any]) -> UniversalPlan:
    """Convert LLM-generated JSON to a UniversalPlan."""
    # Create a new universal plan
    plan = UniversalPlan(
        title=llm_json["title"],
        description="Generated from LLM input",
        tags=["llm-generated"]
    )
    
    # Add metadata about the source
    plan.add_metadata("source", "llm")
    plan.add_metadata("generation_time", str(uuid.uuid4()))
    
    # Create a mapping of LLM step index to Plan step ID
    step_ids = {}
    
    # First pass: Create all steps without tool links
    for i, step_data in enumerate(llm_json["steps"], 1):
        title = step_data["title"]
        step_index = plan.add_step(title, parent=None)
        
        # Get the step node
        step_id = None
        for node in plan._graph.nodes.values():
            if node.__class__.__name__ == "PlanStep" and node.data.get("index") == step_index:
                step_id = node.id
                break
        
        if step_id:
            step_ids[i] = step_id
    
    # Now add tool calls and dependencies
    for i, step_data in enumerate(llm_json["steps"], 1):
        step_id = step_ids.get(i)
        if not step_id:
            continue
            
        # Create tool call
        tool = step_data.get("tool")
        args = step_data.get("args", {})
        
        if tool:
            # Create and link tool call
            tool_call = ToolCall(data={"name": tool, "args": args})
            plan._graph.add_node(tool_call)
            plan._graph.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step_id, dst=tool_call.id))
            
            # Store result variable using a custom edge
            plan._graph.add_edge(GraphEdge(
                kind=EdgeKind.CUSTOM,
                src=step_id,
                dst=tool_call.id,
                data={"type": "result_variable", "variable": f"result_{i}"}
            ))
        
        # Add dependencies
        for dep_idx in step_data.get("depends_on", []):
            dep_id = step_ids.get(dep_idx)
            if dep_id:
                plan._graph.add_edge(GraphEdge(
                    kind=EdgeKind.STEP_ORDER,
                    src=dep_id,
                    dst=step_id
                ))
    
    # Save the plan
    plan.save()
    return plan


# -------------------------------------------------------------------- main flow
async def main(live: bool = False) -> None:
    print("🚀 Universal LLM-to-Execution Demo\n" + "=" * 45)
    
    # Allow the user to customize the task via command line
    parser = argparse.ArgumentParser(description="LLM to Universal Plan Demo")
    parser.add_argument("--task", type=str, 
                        default="I need a plan that first prepares coffee "
                                "then checks today's weather in New York, multiplies 235.5×18.75, "
                                "and finally searches for pages on climate-change adaptation.",
                        help="The natural language task to convert to a plan")
    parser.add_argument("--live", action="store_true",
                        help="Use real OpenAI API instead of simulation")
    args = parser.parse_args()
    
    if args.live and not os.getenv("OPENAI_API_KEY"):
        print("Error: To use --live option, set the OPENAI_API_KEY environment variable")
        exit(1)
    
    task = args.task

    # Step 1: Get LLM-generated plan
    print("\n📝 STEP 1: GENERATING PLAN FROM LLM...\n")
    if live:
        if not AsyncOpenAI:
            print("❌ Error: openai package not installed.")
            print("📋 Run: pip install openai")
            print("📋 Falling back to simulated response...")
            llm_json = await call_llm_sim(task)
        else:
            try:
                llm_json = await call_llm_live(task)
            except Exception as e:
                print(f"❌ Error with OpenAI API: {e}")
                print("📋 Falling back to simulated response...")
                llm_json = await call_llm_sim(task)
    else:
        llm_json = await call_llm_sim(task)
    
    print(f"\nLLM Response:")
    print(json.dumps(llm_json, indent=2))

    # Step 2: Convert to UniversalPlan
    print("\n🔄 STEP 2: CONVERTING TO UNIVERSAL PLAN...\n")
    try:
        plan = convert_to_universal_plan(llm_json)
        
        print("\nUniversal Plan Structure:")
        print(plan.outline())
        
        # Create a simplified version of the plan to display
        plan_summary = {
            "id": plan.id,
            "title": plan.title,
            "description": plan.description,
            "tags": plan.tags,
            "metadata": plan.metadata,
            "steps": []
        }
        
        # Get step information
        for node in plan._graph.nodes.values():
            if node.__class__.__name__ == "PlanStep":
                step_info = {
                    "index": node.data.get("index"),
                    "title": node.data.get("description"),
                    "tool_calls": []
                }
                
                # Find tool calls
                for edge in plan._graph.get_edges(src=node.id, kind=EdgeKind.PLAN_LINK):
                    tool_node = plan._graph.get_node(edge.dst)
                    if tool_node and tool_node.__class__.__name__ == "ToolCall":
                        step_info["tool_calls"].append({
                            "name": tool_node.data.get("name"),
                            "args": tool_node.data.get("args", {})
                        })
                
                # Find dependencies
                dependencies = []
                for edge in plan._graph.get_edges(dst=node.id, kind=EdgeKind.STEP_ORDER):
                    dep_node = plan._graph.get_node(edge.src)
                    if dep_node:
                        dependencies.append(dep_node.data.get("index"))
                
                if dependencies:
                    step_info["depends_on"] = dependencies
                
                plan_summary["steps"].append(step_info)
        
        print("\nPlan Details:")
        print(json.dumps(plan_summary, indent=2))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Error converting LLM output to Universal Plan: {e}")
        return

    # Step 3: Set up executor with tools
    print("\n⚙️ STEP 3: SETTING UP EXECUTOR...\n")
    
    # Create a new executor with the plan's graph store
    executor = UniversalExecutor(graph_store=plan._graph)
    
    # Register all the tools
    executor.register_tool("weather", weather_tool)
    executor.register_tool("calculator", calculator_tool)
    executor.register_tool("search", search_tool)
    
    # Register coffee tools
    for tool_name in ["grind_beans", "boil_water", "brew_coffee", "clean_station"]:
        # Create a closure to capture the tool name
        async def tool_fn(args, name=tool_name):
            return await coffee_tool(args, name)
        
        executor.register_tool(tool_name, tool_fn)
    
    print("Registered tools:")
    print("- weather: Get weather for a location")
    print("- calculator: Perform mathematical operations")
    print("- search: Search for information")
    print("- Coffee tools: grind_beans, boil_water, brew_coffee, clean_station")
    
    # Step 4: Execute the plan
    print("\n🏃 STEP 4: EXECUTING PLAN...\n")
    try:
        result = await executor.execute_plan(plan)
        
        if not result["success"]:
            print(f"\n❌ Plan execution failed: {result['error']}")
            return
        
        print("\n✅ Plan executed successfully!\n")
        
        # Show all variables
        print("Variables produced by plan execution:")
        for name, value in result["variables"].items():
            if name.startswith("result_"):
                print(f"\n--- Result for step {name[7:]} ---")
                pprint.pprint(value, width=100, sort_dicts=False)
        
        # Save results to file
        output_file = "llm_plan_results.json"
        with open(output_file, "w") as f:
            json.dump(result["variables"], f, indent=2, default=str)
        print(f"\n💾 Results saved to {output_file}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Error during execution: {e}")


# -------------------------------------------------------------------- entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM to Universal Plan Demo")
    parser.add_argument("--live", action="store_true",
                        help="Use real OpenAI API instead of simulation")
    parser.add_argument("--task", type=str, 
                        default="I need a plan that first prepares coffee "
                                "then checks today's weather in New York, multiplies 235.5×18.75, "
                                "and finally searches for pages on climate-change adaptation.",
                        help="The natural language task to convert to a plan")
    args = parser.parse_args()
    
    if args.live and not os.getenv("OPENAI_API_KEY"):
        print("Error: To use --live option, set the OPENAI_API_KEY environment variable")
        exit(1)
    
    asyncio.run(main(args.live))