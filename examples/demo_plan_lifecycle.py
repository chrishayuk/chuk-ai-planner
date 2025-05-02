#!/usr/bin/env python
"""
demo_plan_lifecycle.py
======================

One script that shows the *entire* life-cycle:

    Plan DSL  ➜  persisted graph  ➜  GraphAwareToolProcessor  ➜  results
"""

from __future__ import annotations
import asyncio, json, inspect
from typing import Dict, Any, Callable, Iterable

# ── demo real tools (weather / calculator / search) ───────────────────
from sample_tools import WeatherTool, CalculatorTool, SearchTool  # noqa: F401

# ── tiny stub tools so we can demo nested steps & runtime addition ────
async def grind_beans(args: Dict):   return {"action": "beans ground"}
async def boil_water(args: Dict):    return {"temperature": "100 °C"}
async def brew_coffee(args: Dict):   return {"coffee": "espresso ☕"}
async def clean_station(args: Dict): return {"status": "sparkling ✨"}

# ── A2A + session imports ─────────────────────────────────────────────
from a2a_session_manager.storage import InMemorySessionStore, SessionStoreProvider
from a2a_session_manager.models.session import Session
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.planner import Plan
from chuk_ai_planner.models import ToolCall
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind
from chuk_ai_planner.processor import GraphAwareToolProcessor
from chuk_ai_planner.utils.visualization import (
    print_session_events,
    print_graph_structure,
)
from chuk_ai_planner.utils.registry_helpers import execute_tool

# ── chuk registry helper (works on all versions) ──────────────────────
from chuk_tool_processor.registry import default_registry
def registry_names() -> Iterable[str]:
    if hasattr(default_registry, "iter_tools"):
        for meta in default_registry.iter_tools():
            yield meta.name
    elif hasattr(default_registry, "tools"):
        yield from default_registry.tools.keys()             # type: ignore[attr-defined]
    else:
        yield from getattr(default_registry, "_tools", {}).keys()

# ────────────────────────── universal async adapter ───────────────────
import json as _json
async def adapter(tool_name: str, args: Dict[str, Any]) -> Any:
    """
    Always dispatch `tool_name(args)` via the proven `execute_tool` helper
    → works for ValidatedTool classes/instances & plain callables.
    """
    tc = {
        "id": "call",
        "type": "function",
        "function": {"name": tool_name, "arguments": _json.dumps(args)},
    }
    return await execute_tool(tc, _parent_event_id=None, _assistant_node_id=None)

# convenience registration
def reg(proc: GraphAwareToolProcessor, name: str, obj: Any):
    proc.register_tool(name, lambda a, _n=name: adapter(_n, a))

# ─────────────────────────────────  main  ─────────────────────────────
async def main() -> None:
    # 1) session & graph ------------------------------------------------
    SessionStoreProvider.set_store(InMemorySessionStore())
    graph   = InMemoryGraphStore()
    session = Session(); SessionStoreProvider.get_store().save(session)

    # 2) author plan (nested + dependencies) ---------------------------
    plan = (
        Plan("Make coffee & check day", graph=graph)
          .step("Prepare coffee")
              .step("Grind beans").up()
              .step("Boil water").up()
              .step("Brew coffee", after=["1", "2"]).up()
          .up()                         # back to root
          .step("Check day")
              .step("Check weather in New York").up()
              .step("Multiply 235.5 × 18.75").up()
              .step("Search climate-adaptation info").up()
    )
    plan_id = plan.save()

    # runtime addition: cleaning
    plan.add_step("Clean coffee station", parent="1", after=["1.3"])

    print("\n📋  PLAN OUTLINE\n")
    print(plan.outline(), "\n")

    # 3) attach ToolCalls ---------------------------------------------
    idx2id = {n.data["index"]: n.id
              for n in graph.nodes.values()
              if n.__class__.__name__ == "PlanStep"}

    def link(idx: str, name: str, args: Dict[str, Any] | None = None) -> None:
        tc = ToolCall(data={"name": name, "args": args or {}})
        graph.add_node(tc)
        graph.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK,
                                 src=idx2id[idx], dst=tc.id))

    # coffee branch
    link("1.1", "grind_beans")
    link("1.2", "boil_water")
    link("1.3", "brew_coffee")
    link("1.4", "clean_station")

    # day-info branch (real demo tools)
    link("2.1", "weather",    {"location": "New York"})
    link("2.2", "calculator", {"operation": "multiply", "a": 235.5, "b": 18.75})
    link("2.3", "search",     {"query": "climate change adaptation"})

    # 4) processor & tool registry ------------------------------------
    proc = GraphAwareToolProcessor(session_id=session.id, graph_store=graph)

    # registry tools first
    for name in registry_names():
        reg(proc, name, default_registry.get_tool(name))

    # explicit demo + stub tools
    reg(proc, "weather",       WeatherTool)
    reg(proc, "calculator",    CalculatorTool)
    reg(proc, "search",        SearchTool)
    proc.register_tool("grind_beans",  grind_beans)
    proc.register_tool("boil_water",   boil_water)
    proc.register_tool("brew_coffee",  brew_coffee)
    proc.register_tool("clean_station",clean_station)

    # 5) execute plan --------------------------------------------------
    results = await proc.process_plan(
        plan_node_id      = plan_id,
        assistant_node_id = "assistant",
        llm_call_fn       = lambda _: None,
    )

    # 6) output --------------------------------------------------------
    print("✅  TOOL RESULTS\n")
    for r in results:
        print(f"• {r.tool}\n{json.dumps(r.result, indent=2)}\n")

    print_session_events(session)
    print_graph_structure(graph)

if __name__ == "__main__":
    asyncio.run(main())
