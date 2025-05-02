# chuk_ai_planner/utils/pretty.py
# src/chuk_ai_planner/utils/pretty.py
"""
Console helpers: colour, plan outline, and a tidy PlanRunLogger.
"""
from __future__ import annotations
import json, os
from typing import Dict, List, Any, Callable, Awaitable

from chuk_ai_planner.models import GraphNode, NodeKind
from chuk_ai_planner.models.edges import EdgeKind
from chuk_ai_planner.store.base import GraphStore
from a2a_session_manager.models.event_type import EventType


# ─────────────────────────── colour helper ──────────────────────────────
def clr(txt: str, code: str) -> str:
    return txt if os.getenv("NO_COLOR") else f"\033[{code}m{txt}\033[0m"


# ───────────────────────── plan outline print ───────────────────────────
def pretty_print_plan(graph: GraphStore, plan_node: GraphNode) -> None:
    if plan_node.kind != NodeKind.PLAN:
        raise ValueError("expected a PlanNode")

    def key(n: GraphNode) -> List[int]:
        return [int(p) for p in n.data["index"].split(".")]

    def dfs(pid: str, depth: int = 0):
        children = [
            graph.get_node(e.dst)
            for e in graph.get_edges(src=pid, kind=EdgeKind.PARENT_CHILD)
            if (n := graph.get_node(e.dst))
        ]
        children.sort(key=key)
        for ch in children:
            if ch.kind != NodeKind.PLAN_STEP:
                continue
            idx = ch.data["index"]
            indent = "  " * (depth + 1)
            print(f"{indent}{idx:<5} {ch.data['description']}")
            dfs(ch.id, depth + 1)

    print(clr(plan_node.data.get("description", "Plan"), "1;33"))
    dfs(plan_node.id)


# ─────────────────────────── run-time logger ────────────────────────────
class PlanRunLogger:
    """
    Human-readable live log.

        [tool] 1.1 Grind beans → echo({...}) ✓
    """

    def __init__(self, graph: GraphStore, plan_id: str):
        self.label: Dict[str, str] = {}

        # walk tree and build id → "1 Grind beans"
        def walk(step_id: str):
            step = graph.get_node(step_id)
            if not step or step.kind != NodeKind.PLAN_STEP:
                return
            lab = f"{step.data['index']} {step.data['description']}"
            self.label[step.id] = lab

            # map all tools of this step
            for e in graph.get_edges(src=step.id, kind=EdgeKind.PLAN_LINK):
                t = graph.get_node(e.dst)
                if t and t.kind == NodeKind.TOOL_CALL:
                    self.label[t.id] = lab

            # recurse into sub-steps
            for e in graph.get_edges(src=step.id, kind=EdgeKind.PARENT_CHILD):
                walk(e.dst)

        for e in graph.get_edges(src=plan_id, kind=EdgeKind.PARENT_CHILD):
            walk(e.dst)

        # widest label width for alignment
        self._w = max((len(l) for l in self.label.values()), default=1) + 2

    # ------- step summary wrapper -------------------------------------
    def evt(self, typ: EventType, msg: Dict[str, Any], _parent: str):
        if typ is EventType.SUMMARY and "status" in msg and "step_id" in msg:
            lab = self.label.get(msg["step_id"], "<?>")
            print(clr("[step]", "35"), f"{lab:<{self._w}} {msg['status']}")
        return type("Evt", (), {"id": "evt"})()

    # ------- tool-call wrapper ----------------------------------------
    async def proc(
        self,
        tc: Dict[str, Any],
        start_evt_id: str,
        assistant_id: str,
        real_proc: Callable[[Dict[str, Any], str, str], Awaitable[Any]],
    ):
        lab  = self.label.get(tc["id"], "<?>")
        name = tc["function"]["name"]
        args = tc["function"]["arguments"]
        print(
            clr("[tool]", "36"),
            f"{lab:<{self._w}} → {name}({args}) {clr('✓', '32')}"
        )
        return await real_proc(tc, start_evt_id, assistant_id)
