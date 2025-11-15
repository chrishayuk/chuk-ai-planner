#!/usr/bin/env python
"""
demo_planner_demo.py
====================

Showcase:  Plan DSL  →  persisted graph  →  executed plan

Steps
-----
1. Check weather in New York
2. Multiply 235.5 × 18.75
3. Search climate-adaptation info
"""

from __future__ import annotations
import asyncio
import json
from typing import Any, Dict

# ── demo tools self-register on import ───────────────────────────────
from sample_tools import WeatherTool, CalculatorTool, SearchTool  # noqa: F401

# ── core graph / executor helpers ────────────────────────────────────
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.planner import Plan
from chuk_ai_planner.graph import ToolCall
from chuk_ai_planner.graph import GraphEdge, EdgeType
from chuk_ai_planner.planner.plan_executor import PlanExecutor
from chuk_ai_planner.utils.pretty import clr, PlanRunLogger
from chuk_ai_planner.utils.registry_helpers import execute_tool

# ──────────────────────── build plan via DSL ─────────────────────────
g    = InMemoryGraphStore()
plan = (
    Plan("Weather → Calc → Search", graph=g)
        .step("Check weather in New York").up()
        .step("Multiply 235.5 × 18.75").up()
        .step("Search climate-adaptation info")
)
plan_id = plan.save()

print(clr("\n📋  PLAN OUTLINE\n", "1;33"))
print(plan.outline(), "\n")

# ───────────── attach ToolCalls to the saved PlanSteps ───────────────
idx2id = {n.data["index"]: n.id
          for n in g.nodes.values()
          if n.__class__.__name__ == "PlanStep"}

def link(idx: str, name: str, args: Dict):
    tc = ToolCall(data={"name": name, "args": args})
    g.add_node(tc)
    g.add_edge(GraphEdge(kind=EdgeType.PLAN_LINK,
                         src=idx2id[idx], dst=tc.id))

link("1", "weather",    {"location": "New York"})
link("2", "calculator", {"operation": "multiply",
                         "a": 235.5, "b": 18.75})
link("3", "search",     {"query": "climate change adaptation"})

# ───────────────────── executor + console logger ─────────────────────
logger = PlanRunLogger(g, plan_id)
px     = PlanExecutor(g)
_sema  = asyncio.Semaphore(3)

async def guarded_execute_tool(
    tc: Dict,
    _parent_event_id: str | None = None,
    _assistant_node_id: str | None = None,
) -> Any:
    async with _sema:
        return await execute_tool(tc, _parent_event_id, _assistant_node_id)

# ─────────────────────────── run it! ─────────────────────────────────
async def main() -> None:
    print(clr("🛠  EXECUTE", "1;34"))

    results: list[Dict] = []
    steps   = px.get_plan_steps(plan_id)
    batches = px.determine_execution_order(steps)

    for batch in batches:
        coros = [
            px.execute_step(
                sid,
                assistant_node_id="assistant",
                parent_event_id="root_evt",
                create_child_event=logger.evt,
                process_tool_call=lambda tc, e, a: logger.proc(
                    tc, e, a, guarded_execute_tool
                ),
            )
            for sid in batch
        ]
        for rlist in await asyncio.gather(*coros):
            results.extend(rlist)

    print(clr("\n🎉  RESULTS", "1;32"))
    for r in results:
        print(json.dumps(r, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
