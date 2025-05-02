#!/usr/bin/env python
"""
demo_graph_processor.py – 3-step plan with GraphAwareToolProcessor
"""

from __future__ import annotations
import asyncio, json
from typing import Dict, Any, Iterable

from sample_tools import WeatherTool, CalculatorTool, SearchTool  # noqa: F401

from a2a_session_manager.storage import InMemorySessionStore, SessionStoreProvider
from a2a_session_manager.models.session import Session
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.planner import Plan
from chuk_ai_planner.models import ToolCall
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind
from chuk_ai_planner.processor import GraphAwareToolProcessor
from chuk_ai_planner.utils.visualization import print_session_events, print_graph_structure
from chuk_ai_planner.utils.registry_helpers import execute_tool

from chuk_tool_processor.registry import default_registry

def registry_names() -> Iterable[str]:
    if hasattr(default_registry, "iter_tools"):
        for meta in default_registry.iter_tools():
            yield meta.name
    elif hasattr(default_registry, "tools"):
        yield from default_registry.tools.keys()             # type: ignore[attr-defined]
    else:
        yield from getattr(default_registry, "_tools", {}).keys()

# universal adapter ---------------------------------------------------
async def adapter(tool_name: str, args: Dict[str, Any]) -> Any:
    """Always runs the *real* registry tool via execute_tool."""
    tc = {
        "id": "call",
        "type": "function",
        "function": {"name": tool_name, "arguments": json.dumps(args)},
    }
    return await execute_tool(tc, _parent_event_id=None, _assistant_node_id=None)

# main ----------------------------------------------------------------
async def main() -> None:
    SessionStoreProvider.set_store(InMemorySessionStore())
    graph   = InMemoryGraphStore()
    session = Session(); SessionStoreProvider.get_store().save(session)

    plan = (
        Plan("Weather → Calc → Search", graph=graph)
          .step("Check weather in New York").up()
          .step("Multiply 235.5 × 18.75").up()
          .step("Search climate-adaptation info")
    )
    plan_id = plan.save()
    print("\n", plan.outline(), "\n")

    idx2id = {n.data["index"]: n.id
              for n in graph.nodes.values()
              if n.__class__.__name__ == "PlanStep"}

    def link(idx: str, name: str, args: Dict[str, Any]) -> None:
        tc = ToolCall(data={"name": name, "args": args})
        graph.add_node(tc)
        graph.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK,
                                 src=idx2id[idx], dst=tc.id))

    link("1", "weather",    {"location": "New York"})
    link("2", "calculator", {"operation": "multiply", "a": 235.5, "b": 18.75})
    link("3", "search",     {"query": "climate change adaptation"})

    proc = GraphAwareToolProcessor(session_id=session.id, graph_store=graph)

    # generic tools
    for name in registry_names():
        proc.register_tool(name, lambda a, _n=name: adapter(_n, a))

    # ensure demo tools override broken entries
    proc.register_tool("weather",    lambda a: adapter("weather", a))
    proc.register_tool("calculator", lambda a: adapter("calculator", a))
    proc.register_tool("search",     lambda a: adapter("search", a))

    results = await proc.process_plan(
        plan_node_id      = plan_id,
        assistant_node_id = "assistant",
        llm_call_fn       = lambda _: None,
    )

    print("✅  TOOL RESULTS\n")
    for r in results:
        print(f"• {r.tool}\n{json.dumps(r.result, indent=2)}\n")

    print_session_events(session)
    print_graph_structure(graph)

if __name__ == "__main__":
    asyncio.run(main())
