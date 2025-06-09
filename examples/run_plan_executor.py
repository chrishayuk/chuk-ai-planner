#!/usr/bin/env python
# examples/run_plan_executor.py
"""
Registry-driven PlanExecutor demo
================================

• Builds a three-step plan ("Daily helper")
• Executes it with PlanExecutor
• Uses the global tool registry – no ad-hoc tool code in the demo
• Pretty console logging (steps + tool calls)
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

# chuk_ai_planner imports
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.models import GraphNode, NodeKind
from chuk_ai_planner.models.edges import EdgeKind, GraphEdge, ParentChildEdge
from chuk_ai_planner.planner.plan_executor import PlanExecutor
from chuk_ai_planner.utils.pretty import clr, pretty_print_plan, PlanRunLogger

# ───────────────────── Tool implementations ────────────────────────
async def weather_tool(args):
    """Weather tool implementation"""
    location = args.get("location", "Unknown")
    print(f"🌤️ Getting weather for {location}...")
    
    # Mock weather data
    weather_data = {
        "New York": {"temperature": 72, "conditions": "Partly cloudy", "humidity": 65},
        "London": {"temperature": 62, "conditions": "Rainy", "humidity": 80},
        "Tokyo": {"temperature": 78, "conditions": "Sunny", "humidity": 70},
    }
    
    return weather_data.get(location, {"temperature": 75, "conditions": "Unknown", "humidity": 50})

async def calculator_tool(args):
    """Calculator tool implementation"""
    operation = args.get("operation")
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
        if b != 0:
            result = a / b
        else:
            return {"error": "Division by zero"}
    else:
        return {"error": f"Unknown operation: {operation}"}
    
    return {"result": result}

async def search_tool(args):
    """Search tool implementation"""
    query = args.get("query", "")
    print(f"🔍 Searching for: {query}")
    
    # Mock search results
    return {
        "query": query,
        "results": [
            {"title": f"Result for {query}", "url": f"https://example.com/search?q={query}"},
            {"title": f"Guide to {query}", "url": f"https://guide.com/{query}"},
            {"title": f"Research on {query}", "url": f"https://research.org/{query}"}
        ]
    }

# ───────────────────── Register tools with chuk_tool_processor ──────
async def register_tools():
    """Register tools with the chuk_tool_processor registry"""
    try:
        from chuk_tool_processor.registry import get_default_registry
        
        registry = await get_default_registry()
        
        # Register our tools
        registry.register("weather", weather_tool)
        registry.register("calculator", calculator_tool) 
        registry.register("search", search_tool)
        
        print("✅ Tools registered successfully:")
        print("   - weather")
        print("   - calculator")
        print("   - search")
        
    except ImportError:
        print("❌ chuk_tool_processor not available, using fallback")
        return False
    except Exception as e:
        print(f"❌ Error registering tools: {e}")
        return False
    
    return True

# ───────────────────── build tiny plan graph ───────────────────────
async def build_plan():
    print(clr("🟢  BUILD GRAPH\n", "1;32"))

    g = InMemoryGraphStore()
    plan = GraphNode(kind=NodeKind.PLAN,
                     data={"description": "Daily helper"})
    g.add_node(plan)

    def add_step(idx: str, desc: str) -> GraphNode:
        node = GraphNode(kind=NodeKind.PLAN_STEP,
                         data={"index": idx, "description": desc})
        g.add_node(node)
        g.add_edge(ParentChildEdge(src=plan.id, dst=node.id))
        return node

    s1 = add_step("1", "Check weather in New York")
    s2 = add_step("2", "Multiply 235.5 × 18.75")
    s3 = add_step("3", "Search climate-adaptation info")

    def link(step: GraphNode, name: str, args: dict) -> None:
        call = GraphNode(kind=NodeKind.TOOL_CALL,
                         data={"name": name, "args": args})
        g.add_node(call)
        g.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step.id, dst=call.id))

    link(s1, "weather", {"location": "New York"})
    link(s2, "calculator", {"operation": "multiply", "a": 235.5, "b": 18.75})
    link(s3, "search", {"query": "climate change adaptation"})

    pretty_print_plan(g, plan)
    print()
    
    return g, plan

# ───────────────────── Custom execute tool function ─────────────────
async def execute_tool_direct(
    tool_call: dict,
    _parent_event_id: str | None = None,
    _assistant_node_id: str | None = None,
) -> Any:
    """
    Direct tool execution without registry (fallback)
    """
    name = tool_call["function"]["name"]
    args_text = tool_call["function"].get("arguments", "{}")
    
    try:
        args = json.loads(args_text)
    except json.JSONDecodeError:
        args = {"raw_text": args_text}
    
    # Direct tool dispatch
    if name == "weather":
        return await weather_tool(args)
    elif name == "calculator":
        return await calculator_tool(args)
    elif name == "search":
        return await search_tool(args)
    else:
        raise RuntimeError(f"Unknown tool: {name}")

# ───────────────────── executor + logger ───────────────────────────
async def execute_plan(g, plan):
    logger = PlanRunLogger(g, plan.id)
    px = PlanExecutor(g)

    # Small semaphore so the demo doesn't hammer tools in parallel
    _sema = asyncio.Semaphore(3)

    async def guarded_execute_tool(
        tool_call: dict,
        _parent_event_id: str | None = None,
        _assistant_node_id: str | None = None,
    ) -> Any:
        """
        Guarded tool execution with semaphore
        """
        async with _sema:
            # Try registry first, fall back to direct execution
            try:
                from chuk_ai_planner.utils.registry_helpers import execute_tool
                return await execute_tool(tool_call, _parent_event_id, _assistant_node_id)
            except Exception as e:
                print(f"⚠️ Registry execution failed: {e}")
                print("🔄 Falling back to direct execution")
                return await execute_tool_direct(tool_call, _parent_event_id, _assistant_node_id)

    print(clr("🛠  EXECUTE", "1;34"))

    results: list[dict] = []
    steps = px.get_plan_steps(plan.id)
    batches = px.determine_execution_order(steps)

    for batch in batches:
        coroutines = [
            px.execute_step(
                step_id=sid,
                assistant_node_id="assistant",
                parent_event_id="root_evt",
                create_child_event=logger.evt,
                process_tool_call=lambda tc, e, a: logger.proc(
                    tc, e, a, guarded_execute_tool
                ),
            )
            for sid in batch
        ]
        for rlist in await asyncio.gather(*coroutines):
            results.extend(rlist)

    print(clr("\n🎉  RESULTS", "1;32"))
    for r in results:
        print(json.dumps(r, indent=2))

async def main() -> None:
    # Try to register tools first
    registry_available = await register_tools()
    
    if not registry_available:
        print("📝 Note: Using direct tool execution (no registry)")
    
    # Build and execute plan
    g, plan = await build_plan()
    await execute_plan(g, plan)

if __name__ == "__main__":
    asyncio.run(main())