#!/usr/bin/env python
"""
demo_llm_plan_autotools.py – “weather-only” strict demo + reasoning pass
=======================================================================

Task ➜ LLM JSON plan ➜ validated (only 3 weather calls) ➜ executed ➜
LLM summarises which city is hottest.

Run:
    uv run examples/demo_llm_plan_autotools.py          # offline stub
    uv run examples/demo_llm_plan_autotools.py --live   # real OpenAI
"""

from __future__ import annotations
import argparse, asyncio, json, os, textwrap
from typing import Any, Dict, Iterable, List, Tuple

from dotenv import load_dotenv

load_dotenv()            # needed for --live (OPENAI_API_KEY)

# ── demo tools (self-register)
from sample_tools import WeatherTool  # noqa: F401  (calculator/search stay registered but won’t be used)

# ── A2A plumbing
from a2a_session_manager.storage import InMemorySessionStore, SessionStoreProvider
from a2a_session_manager.models.session import Session
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.planner import Plan
from chuk_ai_planner.models import ToolCall
from chuk_ai_planner.models.edges import EdgeKind, GraphEdge
from chuk_ai_planner.processor import GraphAwareToolProcessor
from chuk_ai_planner.utils.visualization import print_session_events, print_graph_structure
from chuk_ai_planner.utils.registry_helpers import execute_tool

from chuk_ai_planner.demo.llm_simulator import simulate_llm_call as sim_llm

try:
    from openai import AsyncOpenAI  # type: ignore
except ImportError:
    AsyncOpenAI = None  # type: ignore

# ───────────────────────────────── allowed tool(s)
ALLOWED_TOOLS = {"weather"}           # <- single-tool demo

TOOL_SPECS: dict[str, dict[str, callable[[Any], bool]]] = {
    "weather": {"location": lambda v: isinstance(v, str) and v.strip()},
}

# ───────────────────────────────── validation helpers
def _step_valid(step: Dict[str, Any]) -> Tuple[bool, str]:
    tool = step.get("tool")
    if tool not in ALLOWED_TOOLS:
        return False, f"tool {tool!r} not allowed in this demo"
    spec, args = TOOL_SPECS[tool], step.get("args", {})
    missing = [k for k in spec if k not in args]
    extra   = [k for k in args if k not in spec]
    bad     = [k for k, fn in spec.items() if k in args and not fn(args[k])]
    if missing: return False, f"{tool}: missing {missing}"
    if extra:   return False, f"{tool}: extra {extra}"
    if bad:     return False, f"{tool}: invalid {bad}"
    return True, ""

# ───────────────────────────────── registry adapter
async def _adapter(name: str, args: Dict[str, Any]) -> Any:
    tc = {"id": "x", "type": "function",
          "function": {"name": name, "arguments": json.dumps(args)}}
    return await execute_tool(tc, None, None)

def _reg_weather(proc: GraphAwareToolProcessor) -> None:
    proc.register_tool("weather", lambda a: _adapter("weather", a))

# ───────────────────────────────── LLM helpers
SYS_MSG = textwrap.dedent("""\
    You write a JSON execution plan that uses ONLY the `weather` tool.
    Required args: { "location": str }

    Return ONLY:

    {
      "title": str,
      "steps": [
        { "title": str, "tool": "weather", "args": { "location": str }, "depends_on": [] }
      ]
    }
""")

async def _chat_live(messages: List[Dict[str, str]]) -> str:
    client = AsyncOpenAI()
    rsp = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=messages,
    )
    return rsp.choices[0].message.content

async def _chat_sim(msgs: List[Dict[str, str]]) -> str:
    return sim_llm(msgs[-1]["content"])

async def _plan_live(prompt: str) -> Dict[str, Any]:
    txt = await _chat_live([{"role": "system", "content": SYS_MSG},
                            {"role": "user", "content": prompt}])
    return json.loads(txt)

async def _plan_sim(_: str) -> Dict[str, Any]:
    return {
        "title": "Weather in three cities",
        "steps": [
            {"title": "Check weather in Paris",  "tool": "weather", "args": {"location": "Paris"},  "depends_on": []},
            {"title": "Check weather in Berlin", "tool": "weather", "args": {"location": "Berlin"}, "depends_on": []},
            {"title": "Check weather in New York", "tool": "weather", "args": {"location": "New York"}, "depends_on": []},
        ],
    }

async def _reason_live(weather_json: str) -> str:
    prompt = (
        "Here is JSON with three cities and their temperatures:\n\n"
        f"{weather_json}\n\n"
        "Which city is hottest?  Reply with one short sentence."
    )
    return await _chat_live([{"role": "user", "content": prompt}])

async def _reason_sim(_: str) -> str:
    return "All three cities share the same temperature (22.5 °C)."

# ───────────────────────────────── plan w/ retry
async def get_plan(prompt: str, live: bool, tries: int = 3) -> Dict[str, Any]:
    caller = _plan_live if live else _plan_sim
    for attempt in range(1, tries + 1):
        plan = await caller(prompt)
        errors = [e for ok, e in (_step_valid(s) for s in plan["steps"]) if not ok]
        if not errors:
            return plan
        if attempt == tries:
            raise RuntimeError("Invalid after retries:\n• " + "\n• ".join(errors))
        prompt = (
            "Your JSON is invalid:\n"
            + "\n".join(f"- {e}" for e in errors)
            + "\nRemember: only the weather tool is allowed with a {location} string."
        )

# ───────────────────────────────── main driver
async def run(live: bool) -> None:
    user_task = (
        "Check the weather in Paris, Berlin and New York and tell me which city will be hottest."
    )

    plan_json = await get_plan(user_task, live)

    # build Plan
    plan = Plan(plan_json["title"])
    for s in plan_json["steps"]:
        plan.step(s["title"]).up()
    plan_id = plan.save()

    print("\n📋  LLM-GENERATED PLAN (validated)\n")
    print(plan.outline(), "\n")

    # link ToolCalls
    idx2id = {n.data["index"]: n.id
              for n in plan.graph.nodes.values()
              if n.__class__.__name__ == "PlanStep"}
    for i, s in enumerate(plan_json["steps"], 1):
        tc = ToolCall(data={"name": "weather", "args": s["args"]})
        plan.graph.add_node(tc)
        plan.graph.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=idx2id[str(i)], dst=tc.id))

    # session + processor
    SessionStoreProvider.set_store(InMemorySessionStore())
    session = Session(); SessionStoreProvider.get_store().save(session)
    proc = GraphAwareToolProcessor(session.id, plan.graph)
    _reg_weather(proc)

    results = await proc.process_plan(plan_id, assistant_node_id="assistant",
                                      llm_call_fn=lambda _: None)

    print("✅  TOOL RESULTS\n")
    for r in results:
        print(f"• {r.tool}\n{json.dumps(r.result, indent=2)}\n")

    # reasoning pass
    weather_json = json.dumps([r.result for r in results], indent=2)
    reasoning = await (_reason_live if live else _reason_sim)(weather_json)
    print("🤔  LLM REASONING\n")
    print(reasoning, "\n")

    print_session_events(session)
    print_graph_structure(plan.graph)

# ───────────────────────────────── CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use OpenAI instead of simulator")
    opts = ap.parse_args()

    if opts.live and (AsyncOpenAI is None or not os.getenv("OPENAI_API_KEY")):
        ap.error("Install `openai` and define OPENAI_API_KEY, or omit --live")

    asyncio.run(run(opts.live))
