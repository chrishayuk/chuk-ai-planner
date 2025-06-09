#!/usr/bin/env python
# examples/universal_llm_plan_demo.py
"""
universal_llm_plan_demo.py
==========================

Natural-language task → LLM-generated JSON plan → Universal Plan → executed plan

Demonstrates integrating the UniversalPlan class with the LLM-based plan generation flow.
"""

from __future__ import annotations
import argparse, asyncio, json, os, re, uuid
from typing import Dict, Any, List, Optional

# Import the official UniversalPlan implementation
from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor

# ── A2A plumbing -----------------------------------------------------
from chuk_ai_planner.models import ToolCall
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind
from chuk_ai_planner.utils.visualization import print_session_events, print_graph_structure

from dotenv import load_dotenv
load_dotenv()

# ── OpenAI (optional) -------------------------------------------------
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore

# ── LLM helpers -------------------------------------------------------
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
    if not AsyncOpenAI:
        raise RuntimeError("openai package not installed")
    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_MSG},
            {"role": "user", "content": task},
        ],
    )
    return json.loads(resp.choices[0].message.content)


async def call_llm_sim(task: str) -> Dict[str, Any]:
    """
    Simulation that returns a fixed plan for the coffee and weather task
    """
    print(f"🤖 Simulating LLM call for task: {task}\n")
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


# -------------------------------------------------------------------- Universal Plan conversion
def convert_to_universal_plan(llm_json: Dict[str, Any]) -> UniversalPlan:
    """Convert LLM-generated JSON to a UniversalPlan using the enhanced API"""
    # Create a new universal plan
    plan = UniversalPlan(
        title=llm_json["title"],
        description="Generated from LLM input",
        tags=["llm-generated"]
    )
    
    # Add metadata about the source
    plan.add_metadata("source", "llm")
    plan.add_metadata("generation_time", str(asyncio.get_running_loop().time()))
    
    # Create a mapping of LLM step index to Plan step ID
    step_ids = {}
    
    # Create all steps and their tool calls
    for i, step_data in enumerate(llm_json["steps"], 1):
        title = step_data["title"]
        tool = step_data.get("tool")
        args = step_data.get("args", {})
        
        if tool:
            # Use the enhanced API to add tool steps
            step_id = plan.add_tool_step(
                title=title,
                tool=tool,
                args=args,
                result_variable=f"result_{i}"
            )
            step_ids[i] = step_id
        else:
            # Fallback for steps without tools
            step_index = plan.add_step(title, parent=None)
            # Find the step ID (this is a bit complex with the current API)
            for node in plan._graph.nodes.values():
                if (node.__class__.__name__ == "PlanStep" and 
                    node.data.get("index") == step_index):
                    step_ids[i] = node.id
                    break
    
    # Add dependencies after all steps are created
    for i, step_data in enumerate(llm_json["steps"], 1):
        step_id = step_ids.get(i)
        if not step_id:
            continue
            
        # Add dependencies using step order edges
        for dep_idx in step_data.get("depends_on", []):
            dep_id = step_ids.get(dep_idx)
            if dep_id:
                plan._graph.add_edge(GraphEdge(
                    kind=EdgeKind.STEP_ORDER,
                    src=dep_id,
                    dst=step_id
                ))
    
    return plan


# -------------------------------------------------------------------- Tool implementations
def create_tool_implementations():
    """Create implementations for all the tools used in the demo"""
    
    async def grind_beans_tool(args):
        print("☕ Grinding coffee beans...")
        await asyncio.sleep(0.1)  # Simulate work
        return {"status": "Beans ground successfully", "grind_size": "medium"}
    
    async def boil_water_tool(args):
        print("🔥 Boiling water...")
        await asyncio.sleep(0.1)
        return {"status": "Water boiled to 200°F", "temperature": 200}
    
    async def brew_coffee_tool(args):
        print("☕ Brewing coffee...")
        await asyncio.sleep(0.1)
        return {"status": "Coffee brewed perfectly", "strength": "medium", "aroma": "excellent"}
    
    async def clean_station_tool(args):
        print("🧽 Cleaning coffee station...")
        await asyncio.sleep(0.1)
        return {"status": "Coffee station cleaned and sanitized"}
    
    async def calculator_tool(args):
        print(f"🧮 Calculating: {args}")
        if args.get("operation") == "multiply":
            a = float(args.get("a", 0))
            b = float(args.get("b", 0))
            result = a * b
            print(f"   {a} × {b} = {result}")
            return {"result": result, "operation": "multiply", "operands": [a, b]}
        return {"result": 0, "operation": args.get("operation", "unknown")}
    
    async def weather_tool(args):
        location = args.get("location", "Unknown")
        print(f"🌤️ Checking weather in {location}...")
        await asyncio.sleep(0.1)
        return {
            "temperature": 72, 
            "conditions": "Partly cloudy", 
            "location": location,
            "humidity": 65,
            "wind_speed": "8 mph"
        }
    
    async def search_tool(args):
        query = args.get("query", "")
        print(f"🔍 Searching for: {query}")
        await asyncio.sleep(0.1)
        return {
            "query": query,
            "results": [
                {"title": f"Climate Change Adaptation Strategies", "url": "https://example.com/climate-adaptation"},
                {"title": f"Best Practices for {query}", "url": "https://example.com/best-practices"},
                {"title": f"Research on {query}", "url": "https://example.com/research"}
            ],
            "result_count": 3
        }
    
    return {
        "grind_beans": grind_beans_tool,
        "boil_water": boil_water_tool,
        "brew_coffee": brew_coffee_tool,
        "clean_station": clean_station_tool,
        "calculator": calculator_tool,
        "weather": weather_tool,
        "search": search_tool
    }


# -------------------------------------------------------------------- main flow
async def main(live: bool) -> None:
    task = (
        "I need a short plan that first prepares coffee "
        "then checks today's weather in New York, multiplies 235.5×18.75, "
        "and finally searches for pages on climate-change adaptation."
    )

    # Get LLM-generated plan
    print("\n🤖 GENERATING PLAN FROM LLM...\n")
    llm_json = await (call_llm_live if live else call_llm_sim)(task)
    print(f"LLM Response: {json.dumps(llm_json, indent=2)}\n")

    # Convert to UniversalPlan
    print("\n🔄 CONVERTING TO UNIVERSAL PLAN...\n")
    plan = convert_to_universal_plan(llm_json)
    
    # Save the plan
    plan_id = plan.save()

    print("\n📋 UNIVERSAL PLAN STRUCTURE\n")
    print(plan.outline(), "\n")

    # Set up executor for execution
    print("\n🚀 EXECUTING PLAN...\n")
    
    # Create executor
    executor = UniversalExecutor(graph_store=plan.graph)
    
    # Register all tools
    tools = create_tool_implementations()
    for tool_name, tool_func in tools.items():
        executor.register_tool(tool_name, tool_func)
    
    print(f"Registered tools: {list(tools.keys())}\n")

    # Execute the plan
    try:
        result = await executor.execute_plan(plan)
        
        if result["success"]:
            print("\n✅ PLAN EXECUTED SUCCESSFULLY!\n")
            
            # Show results
            print("📊 EXECUTION RESULTS:\n")
            for var_name, var_value in result["variables"].items():
                if var_name.startswith("result_"):
                    step_num = var_name.split("_")[1]
                    print(f"Step {step_num} result:")
                    print(f"  {json.dumps(var_value, indent=2)}\n")
            
            print("🎯 KEY OUTCOMES:\n")
            # Extract key results
            for var_name, var_value in result["variables"].items():
                if var_name == "result_4":  # Weather result
                    weather = var_value
                    print(f"🌤️ Weather in {weather.get('location')}: {weather.get('temperature')}°F, {weather.get('conditions')}")
                elif var_name == "result_5":  # Calculator result
                    calc = var_value
                    print(f"🧮 Calculation: {calc.get('result')}")
                elif var_name == "result_6":  # Search result
                    search = var_value
                    print(f"🔍 Found {search.get('result_count')} search results for '{search.get('query')}'")
        else:
            print(f"\n❌ PLAN EXECUTION FAILED: {result.get('error')}\n")
            
    except Exception as e:
        print(f"\n💥 EXECUTION ERROR: {e}\n")


# -------------------------------------------------------------------- entry-point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Use real OpenAI rather than the simulator")
    args = parser.parse_args()

    if args.live and not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY or omit --live")

    asyncio.run(main(args.live))