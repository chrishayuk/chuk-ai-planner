# chuk_ai_planner/utils/pretty.py
"""
Console helpers: colour, plan outline, and a tidy PlanRunLogger.
"""

from __future__ import annotations
import os
from typing import Dict, List, Any, Callable, Awaitable

from chuk_ai_planner.core.graph import GraphNode, NodeType, PlanNode, PlanStep
from chuk_ai_planner.core.graph import EdgeType
from chuk_ai_planner.core.store.base import GraphStore
from chuk_session_manager.models.event_type import EventType  # type: ignore


# ─────────────────────────── colour helper ──────────────────────────────
def clr(txt: str, code: str) -> str:
    return txt if os.getenv("NO_COLOR") else f"\033[{code}m{txt}\033[0m"


# ───────────────────────── plan outline print ───────────────────────────
async def pretty_print_plan(graph: GraphStore, plan_node: GraphNode) -> None:
    """Print a plan tree (async-native!)."""
    if plan_node.kind != NodeType.PLAN:
        raise ValueError("expected a PlanNode")

    # Type-safe cast since we checked kind
    typed_plan = plan_node if isinstance(plan_node, PlanNode) else None
    if not typed_plan:
        raise ValueError("expected a PlanNode instance")

    def key(n: GraphNode) -> List[int]:
        # Type-safe access to index field
        if isinstance(n, PlanStep) and n.index:
            return [int(p) for p in n.index.split(".")]
        return [0]

    async def dfs(pid: str, depth: int = 0):
        # Async-native: await GraphStore calls!
        edges = await graph.get_edges(src=pid, kind=EdgeType.PARENT_CHILD)
        children = [await graph.get_node(e.dst) for e in edges]
        # Filter out None values
        valid_children = [ch for ch in children if ch is not None]
        valid_children.sort(key=key)
        for ch in valid_children:
            if ch.kind != NodeType.PLAN_STEP or not isinstance(ch, PlanStep):
                continue
            idx = ch.index or ""
            indent = "  " * (depth + 1)
            print(f"{indent}{idx:<5} {ch.description}")
            await dfs(ch.id, depth + 1)

    print(clr(typed_plan.title or "Plan", "1;33"))
    await dfs(plan_node.id)


# ─────────────────────────── run-time logger ────────────────────────────
class PlanRunLogger:
    """
    Human-readable live log.

        [tool] 1.1 Grind beans → echo({...}) ✓
    """

    def __init__(self) -> None:
        self.label: Dict[str, str] = {}
        self._w = 1

    @classmethod
    async def create(cls, graph: GraphStore, plan_id: str) -> "PlanRunLogger":
        """
        Async factory method to create a PlanRunLogger.

        Parameters
        ----------
        graph : GraphStore
            The graph store to read from
        plan_id : str
            The plan node ID

        Returns
        -------
        PlanRunLogger
            The initialized logger
        """
        logger = cls()

        # walk tree and build id → "1 Grind beans"
        async def walk(step_id: str):
            step = await graph.get_node(step_id)
            if not step or step.kind != NodeType.PLAN_STEP:
                return

            # Type-safe access to PlanStep fields
            if isinstance(step, PlanStep):
                lab = f"{step.index or ''} {step.description}"
                logger.label[step.id] = lab

                # map all tools of this step
                for e in await graph.get_edges(src=step.id, kind=EdgeType.PLAN_LINK):
                    t = await graph.get_node(e.dst)
                    if t and t.kind == NodeType.TOOL_CALL:
                        logger.label[t.id] = lab

                # recurse into sub-steps
                for e in await graph.get_edges(src=step.id, kind=EdgeType.PARENT_CHILD):
                    await walk(e.dst)

        for e in await graph.get_edges(src=plan_id, kind=EdgeType.PARENT_CHILD):
            await walk(e.dst)

        # widest label width for alignment
        logger._w = max((len(lbl) for lbl in logger.label.values()), default=1) + 2

        return logger

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
        start_evt_id: str | None,
        assistant_id: str | None,
        real_proc: Callable[[Dict[str, Any], str | None, str | None], Awaitable[Any]],
    ):
        lab = self.label.get(tc["id"], "<?>")
        name = tc["function"]["name"]
        args = tc["function"]["arguments"]
        print(
            clr("[tool]", "36"), f"{lab:<{self._w}} → {name}({args}) {clr('✓', '32')}"
        )
        return await real_proc(tc, start_evt_id, assistant_id)
