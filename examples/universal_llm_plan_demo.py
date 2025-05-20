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

# ── demo tools (register on import) ──────────────────────────────────
from sample_tools import WeatherTool, CalculatorTool, SearchTool  # noqa: F401

# ── A2A plumbing -----------------------------------------------------
from chuk_session_manager.storage import InMemorySessionStore, SessionStoreProvider
from chuk_session_manager.models.session import Session
from chuk_ai_planner.models import ToolCall
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind
from chuk_ai_planner.processor import GraphAwareToolProcessor
from chuk_ai_planner.utils.visualization import print_session_events, print_graph_structure
from chuk_ai_planner.utils.registry_helpers import execute_tool

from dotenv import load_dotenv
load_dotenv()

# ── OpenAI (optional) -------------------------------------------------
from chuk_ai_planner.demo.llm_simulator import simulate_llm_call
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore

# ── quick registry name helper ---------------------------------------
from chuk_tool_processor.registry import default_registry
def registry_names():
    if hasattr(default_registry, "iter_tools"):
        for meta in default_registry.iter_tools():
            yield meta.name
    elif hasattr(default_registry, "tools"):
        yield from default_registry.tools.keys()             # type: ignore[attr-defined]
    else:
        yield from getattr(default_registry, "_tools", {}).keys()

# universal async adapter (same as earlier demo) ----------------------
async def adapter(tool_name: str, args: Dict[str, Any]) -> Any:
    tc = {"id": "call",
          "type": "function",
          "function": {"name": tool_name, "arguments": json.dumps(args)}}
    return await execute_tool(tc, _parent_event_id=None, _assistant_node_id=None)

def reg(proc: GraphAwareToolProcessor, name: str):
    proc.register_tool(name, lambda a, _n=name: adapter(_n, a))

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
    _ = await simulate_llm_call(task)  # just to show the prompt
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
    """Convert LLM-generated JSON to a UniversalPlan"""
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
    
    # First pass: Create all steps without tool links
    # This is different from the implementation in UniversalPlan
    # where we use the standard Chuk Plan API
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
            # This is a way to work around the EdgeKind limitation
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
    
    return plan


# -------------------------------------------------------------------- Tool registration helper
def create_mock_tool_executor(name):
    """Create a mock executor for a given tool name"""
    async def mock_executor(args):
        if name == "grind_beans":
            return {"status": "Beans ground successfully"}
        elif name == "boil_water":
            return {"status": "Water boiled to 200°F", "temperature": 200}
        elif name == "brew_coffee":
            return {"status": "Coffee brewed perfectly", "strength": "medium", "aroma": "excellent"}
        elif name == "clean_station":
            return {"status": "Coffee station cleaned"}
        elif name == "calculator":
            if args.get("operation") == "multiply":
                a = float(args.get("a", 0))
                b = float(args.get("b", 0))
                return {"result": a * b, "operation": "multiply"}
            return {"result": 0, "operation": args.get("operation", "unknown")}
        elif name == "weather":
            location = args.get("location", "Unknown")
            return {"temperature": 72, "conditions": "Partly cloudy", "location": location}
        elif name == "search":
            query = args.get("query", "")
            return {"results": [{"title": f"Result for {query}", "url": f"https://example.com/search?q={query}"}]}
        else:
            return {"status": f"Executed {name} with args: {args}"}
    return mock_executor


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
    
    # Create a simplified version of the plan to display
    plan_display = {
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
                "id": node.id,
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
            
            plan_display["steps"].append(step_info)
    
    print("\n📝 UNIVERSAL PLAN DETAILS\n")
    print(json.dumps(plan_display, indent=2), "\n")

    # Set up processor for execution
    print("\n🚀 EXECUTING PLAN...\n")
    SessionStoreProvider.set_store(InMemorySessionStore())
    session = Session(); SessionStoreProvider.get_store().save(session)

    proc = GraphAwareToolProcessor(session_id=session.id, graph_store=plan.graph)

    # Explicitly register all required tools with proper implementations
    required_tools = [
        "grind_beans", "boil_water", "brew_coffee", "clean_station", 
        "calculator", "weather", "search"
    ]
    
    # Register all the tools with mock executors
    for tool_name in required_tools:
        proc.register_tool(tool_name, create_mock_tool_executor(tool_name))
    
    print("Registered tools:", required_tools)

    # Run plan
    results = await proc.process_plan(
        plan_node_id=plan_id,
        assistant_node_id="assistant",
        llm_call_fn=lambda _: None,
    )

    print("\n✅ TOOL RESULTS\n")
    for r in results:
        print(f"• {r.tool}\n{json.dumps(r.result, indent=2)}\n")

    print_session_events(session)
    print_graph_structure(plan.graph)


# -------------------------------------------------------------------- entry-point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Use real OpenAI rather than the simulator")
    args = parser.parse_args()

    if args.live and not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY or omit --live")

    asyncio.run(main(args.live))