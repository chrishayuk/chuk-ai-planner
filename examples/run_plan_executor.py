#!/usr/bin/env python
# examples/run_plan_executor.py
"""
Registry-driven PlanExecutor demo
================================

• Builds a three-step plan (“Daily helper”)
• Executes it with PlanExecutor
• Uses the global tool registry – no ad-hoc tool code in the demo
• Pretty console logging (steps + tool calls)
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

# register tools
from sample_tools import WeatherTool, CalculatorTool, SearchTool  # noqa: F401

# chuk_ai_planner imports
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.models import GraphNode, NodeKind
from chuk_ai_planner.models.edges import EdgeKind, GraphEdge, ParentChildEdge
from chuk_ai_planner.planner.plan_executor import PlanExecutor
from chuk_ai_planner.utils.pretty import clr, pretty_print_plan, PlanRunLogger
from chuk_ai_planner.utils.registry_helpers import execute_tool          # <── central helper

# ───────────────────── build tiny plan graph ───────────────────────
print(clr("🟢  BUILD GRAPH\n", "1;32"))

g     = InMemoryGraphStore()
plan  = GraphNode(kind=NodeKind.PLAN,
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


link(s1, "weather",    {"location": "New York"})
link(s2, "calculator", {"operation": "multiply", "a": 235.5, "b": 18.75})
link(s3, "search",     {"query": "climate change adaptation"})

pretty_print_plan(g, plan)
print()

# ───────────────────── executor + logger ───────────────────────────
logger = PlanRunLogger(g, plan.id)
px     = PlanExecutor(g)

# ── small semaphore so the demo doesn’t hammer the registry in parallel
_sema = asyncio.Semaphore(3)

async def guarded_execute_tool(
    tool_call: dict,
    _parent_event_id: str | None = None,
    _assistant_node_id: str | None = None,
) -> Any:
    """
    Thin async wrapper that forwards the *exact* signature PlanExecutor
    passes (`tc, parent_event_id, assistant_node_id`) to `execute_tool`
    while ensuring only a handful run concurrently.
    """
    async with _sema:
        return await execute_tool(tool_call, _parent_event_id, _assistant_node_id)


async def main() -> None:
    print(clr("🛠  EXECUTE", "1;34"))

    results: list[dict] = []
    steps   = px.get_plan_steps(plan.id)
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


if __name__ == "__main__":
    asyncio.run(main())
