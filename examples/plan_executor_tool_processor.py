#!/usr/bin/env python
# examples/plan_executor_tool_processor.py
"""
Registry-driven PlanExecutor
====================================================

• Uses proper chuk_tool_processor API based on the execution_strategies_demo
• Builds a three-step plan ("Daily helper")
• Executes it with PlanExecutor using proper tool registration
• Pretty console logging (steps + tool calls)
• Demonstrates both InProcess and Subprocess strategies
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

# chuk_ai_planner imports
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.models import GraphNode, NodeKind
from chuk_ai_planner.models.edges import EdgeKind, GraphEdge, ParentChildEdge
from chuk_ai_planner.planner.plan_executor import PlanExecutor
from chuk_ai_planner.utils.pretty import clr, pretty_print_plan, PlanRunLogger

# chuk_tool_processor imports (proper API)
from chuk_tool_processor.registry import register_tool, initialize
from chuk_tool_processor.models.tool_call import ToolCall
from chuk_tool_processor.execution.strategies.inprocess_strategy import InProcessStrategy
from chuk_tool_processor.execution.tool_executor import ToolExecutor

# ───────────────────── Tool implementations (using decorators) ──────
@register_tool(name="weather", namespace="demo")
class WeatherTool:
    """Weather tool implementation"""
    
    async def execute(self, location: str = "Unknown") -> Dict[str, Any]:
        """Get weather for a location"""
        print(f"🌤️ Getting weather for {location}...")
        
        # Mock weather data
        weather_data = {
            "New York": {"temperature": 72, "conditions": "Partly cloudy", "humidity": 65},
            "London": {"temperature": 62, "conditions": "Rainy", "humidity": 80},
            "Tokyo": {"temperature": 78, "conditions": "Sunny", "humidity": 70},
        }
        
        return weather_data.get(location, {"temperature": 75, "conditions": "Unknown", "humidity": 50})


@register_tool(name="calculator", namespace="demo") 
class CalculatorTool:
    """Calculator tool implementation"""
    
    async def execute(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        """Perform a calculation"""
        print(f"🧮 Calculating: {a} {operation} {b}")
        
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return {"result": result}


@register_tool(name="search", namespace="demo")
class SearchTool:
    """Search tool implementation"""
    
    async def execute(self, query: str = "") -> Dict[str, Any]:
        """Search for information"""
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

# ───────────────────── Tool Executor Wrapper ───────────────────────
class ToolProcessorWrapper:
    """Wrapper to integrate chuk_tool_processor with PlanExecutor"""
    
    def __init__(self, registry):
        self.registry = registry
        self.strategy = InProcessStrategy(registry, default_timeout=10.0, max_concurrency=4)
        self.executor = ToolExecutor(registry=registry, strategy=self.strategy)
    
    async def execute_tool_call(self, tool_call: Dict[str, Any], parent_event_id: str = None, assistant_node_id: str = None) -> Any:
        """Execute a tool call using chuk_tool_processor"""
        tool_name = tool_call["function"]["name"]
        args_text = tool_call["function"].get("arguments", "{}")
        
        try:
            args = json.loads(args_text)
        except json.JSONDecodeError:
            args = {"raw_text": args_text}
        
        # Create ToolCall object
        call = ToolCall(
            tool=tool_name,
            namespace="demo",
            arguments=args
        )
        
        # Execute using the tool processor
        results = await self.executor.execute([call])
        
        if results and len(results) > 0:
            result = results[0]
            if result.error:
                raise RuntimeError(f"Tool execution failed: {result.error}")
            return result.result
        else:
            raise RuntimeError("No results returned from tool execution")
    
    async def shutdown(self):
        """Shutdown the executor"""
        await self.executor.strategy.shutdown()

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

# ───────────────────── executor + logger ───────────────────────────
async def execute_plan(g, plan, tool_wrapper):
    logger = PlanRunLogger(g, plan.id)
    px = PlanExecutor(g)

    # Small semaphore so the demo doesn't hammer tools in parallel
    _sema = asyncio.Semaphore(3)

    async def guarded_execute_tool(
        tool_call: dict,
        parent_event_id: str = None,
        assistant_node_id: str = None,
    ) -> Any:
        """Execute tool with semaphore and proper error handling"""
        async with _sema:
            return await tool_wrapper.execute_tool_call(tool_call, parent_event_id, assistant_node_id)

    print(clr("🛠  EXECUTE", "1;34"))

    results: list[dict] = []
    steps = px.get_plan_steps(plan.id)
    batches = px.determine_execution_order(steps)

    print(f"📊 Plan analysis:")
    print(f"   - {len(steps)} steps found")
    print(f"   - {len(batches)} execution batches")
    
    for i, batch in enumerate(batches, 1):
        print(f"   - Batch {i}: {len(batch)} steps")

    for batch_num, batch in enumerate(batches, 1):
        print(f"\n--- Executing Batch {batch_num} ---")
        
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
        
        try:
            batch_results = await asyncio.gather(*coroutines)
            for rlist in batch_results:
                results.extend(rlist)
            print(f"✅ Batch {batch_num} completed successfully")
        except Exception as e:
            print(f"❌ Batch {batch_num} failed: {e}")
            # Continue with next batch

    print(clr("\n🎉  RESULTS", "1;32"))
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(json.dumps(r, indent=2))
    
    return results

# ───────────────────── Demo with strategy comparison ──────────────────
async def demo_with_strategy_comparison(g, plan):
    """Demonstrate plan execution with different strategies"""
    print(clr("\n🔄 STRATEGY COMPARISON", "1;35"))
    
    # Initialize registry
    registry = await initialize()
    
    # Test InProcess strategy
    print("\n=== InProcess Strategy ===")
    inprocess_wrapper = ToolProcessorWrapper(registry)
    inprocess_wrapper.strategy = InProcessStrategy(registry, default_timeout=10.0, max_concurrency=4)
    inprocess_wrapper.executor = ToolExecutor(registry=registry, strategy=inprocess_wrapper.strategy)
    
    try:
        inprocess_results = await execute_plan(g, plan, inprocess_wrapper)
        print(f"InProcess strategy: {len(inprocess_results)} results")
    finally:
        await inprocess_wrapper.shutdown()
    
    # Test Subprocess strategy (if available)
    try:
        from chuk_tool_processor.execution.strategies.subprocess_strategy import SubprocessStrategy
        
        print("\n=== Subprocess Strategy ===")
        subprocess_wrapper = ToolProcessorWrapper(registry)
        subprocess_wrapper.strategy = SubprocessStrategy(registry, max_workers=4, default_timeout=10.0)
        subprocess_wrapper.executor = ToolExecutor(registry=registry, strategy=subprocess_wrapper.strategy)
        
        try:
            subprocess_results = await execute_plan(g, plan, subprocess_wrapper)
            print(f"Subprocess strategy: {len(subprocess_results)} results")
        finally:
            await subprocess_wrapper.shutdown()
            
    except ImportError:
        print("\n⚠️ Subprocess strategy not available")
    except Exception as e:
        print(f"\n❌ Subprocess strategy failed: {e}")

# ───────────────────── Main function ───────────────────────────────
async def main() -> None:
    print(clr("🚀 Enhanced Plan Executor with chuk_tool_processor", "1;36"))
    print("=" * 60)
    
    # Initialize the registry first
    print("🔧 Initializing chuk_tool_processor registry...")
    registry = await initialize()
    print("✅ Registry initialized with tools:")
    print("   - weather (demo namespace)")
    print("   - calculator (demo namespace)") 
    print("   - search (demo namespace)")
    
    # Build plan
    g, plan = await build_plan()
    
    # Create tool wrapper
    tool_wrapper = ToolProcessorWrapper(registry)
    
    try:
        # Execute plan with default strategy
        results = await execute_plan(g, plan, tool_wrapper)
        
        # Summary
        print(clr(f"\n📈 SUMMARY", "1;33"))
        print(f"   - Plan execution completed")
        print(f"   - {len(results)} total results")
        print(f"   - Tools executed via chuk_tool_processor")
        
        if results:
            print(f"   - All tools executed successfully ✅")
        else:
            print(f"   - No results generated ⚠️")
        
        # Run strategy comparison if requested
        print(f"\n🔄 Running strategy comparison...")
        await demo_with_strategy_comparison(g, plan)
        
    finally:
        # Clean up
        await tool_wrapper.shutdown()

if __name__ == "__main__":
    asyncio.run(main())