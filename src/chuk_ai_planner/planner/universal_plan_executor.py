"""
Enhanced Universal Plan Executor
================================

Drop‑in replacement for ``src/chuk_ai_planner/planner/universal_plan_executor.py``
that fixes

1. **GraphStore mismatch** – option to pass an existing ``GraphStore`` so
   the executor sees the same nodes the plan saved.
2. **Robust result handling** – safely unwraps any *ToolResult* /
   dataclass / nested dict so variables like ``report`` contain the
   expected plain payload (with keys such as ``summary``).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from a2a_session_manager.models.session import Session
from a2a_session_manager.storage import InMemorySessionStore, SessionStoreProvider

from chuk_ai_planner.models.edges import EdgeKind
from chuk_ai_planner.processor import GraphAwareToolProcessor
from chuk_ai_planner.store.base import GraphStore
from chuk_ai_planner.store.memory import InMemoryGraphStore

from .plan_executor import PlanExecutor
from .universal_plan import UniversalPlan

__all__ = ["UniversalExecutor"]


class UniversalExecutor:
    """Execute :class:`~chuk_ai_planner.planner.universal_plan.UniversalPlan`."""

    # ------------------------------------------------------------------ init
    def __init__(self, graph_store: GraphStore | None = None):
        # ensure there is a session store
        try:
            SessionStoreProvider.get_store()
        except Exception:
            SessionStoreProvider.set_store(InMemorySessionStore())

        self.session: Session = Session()
        SessionStoreProvider.get_store().save(self.session)

        # allow caller‑provided graph store (avoids step‑not‑found issue)
        self.graph_store: GraphStore = graph_store or InMemoryGraphStore()

        self.processor = GraphAwareToolProcessor(
            self.session.id,
            self.graph_store,
            enable_caching=True,
            enable_retries=True,
        )
        self.plan_executor = PlanExecutor(self.graph_store)
        self.assistant_node_id: str = str(uuid.uuid4())

        self.tool_registry: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self.function_registry: Dict[str, Callable[..., Any]] = {}

    # ----------------------------------------------------------- registry
    def register_tool(self, name: str, fn: Callable[..., Awaitable[Any]]) -> None:
        self.tool_registry[name] = fn
        self.processor.register_tool(name, fn)

    def register_function(self, name: str, fn: Callable[..., Any]) -> None:
        self.function_registry[name] = fn

        async def wrapper(args: Dict[str, Any]):
            fn_name = args.get("function")
            fn_args = args.get("args", {})
            target = self.function_registry.get(fn_name)
            if target is None:
                raise ValueError(f"Unknown function {fn_name!r}")
            if asyncio.iscoroutinefunction(target):
                return await target(**fn_args)
            return target(**fn_args)

        self.processor.register_tool("function", wrapper)

    # ----------------------------------------------------------- execution
    async def execute_plan(self, plan: UniversalPlan, variables: Optional[Dict[str, Any]] = None):
        # copy plan graph into our store if necessary
        if plan.graph is not self.graph_store:
            for node in plan.graph.nodes.values():
                self.graph_store.add_node(node)
            for edge in plan.graph.edges:
                self.graph_store.add_edge(edge)

        if not plan._indexed:
            plan.save()

        ctx: Dict[str, Any] = {
            "variables": {**plan.variables, **(variables or {})},
            "results": {},
        }

        async def noop_llm(_):
            return {"content": "noop", "tool_calls": []}

        try:
            await self.processor.process_plan(
                plan.id,
                self.assistant_node_id,
                noop_llm,
                on_step=lambda sid, res: self._process_step_results(sid, res, ctx),
            )
            return {"success": True, **ctx}
        except Exception as exc:
            return {"success": False, "error": str(exc), **ctx}

    # ----------------------------------------------------------- helpers
    # ----------------------------------------------------------- helpers
    def _extract_value(self, obj: Any) -> Any:
        """Return a plain payload regardless of how deeply it’s wrapped."""
        # --- 0. None ------------------------------------------------------
        if obj is None:
            return None

        # --- 1. single‑element list --------------------------------------
        if isinstance(obj, list):
            if len(obj) == 1:
                return self._extract_value(obj[0])
            return [self._extract_value(x) for x in obj]

        # --- 2. dicts -----------------------------------------------------
        if isinstance(obj, dict):
            val = obj
            # peel layers of {"result": …}, {"data": …}, {"payload": …}
            while (
                isinstance(val, dict)
                and len(val) == 1
                and next(iter(val)) in ("result", "payload", "data")
            ):
                val = next(iter(val.values()))
            return val

        # --- 3. objects with common attributes ---------------------------
        for attr in ("result", "payload", "data"):
            if hasattr(obj, attr):
                inner = getattr(obj, attr)
                if inner is not None:
                    return self._extract_value(inner)

        # --- 4. dataclass -------------------------------------------------
        if is_dataclass(obj):
            return asdict(obj)

        # --- 5. fallback --------------------------------------------------
        return getattr(obj, "__dict__", obj)

    def _process_step_results(self, step_id: str, results: Any, context: Dict[str, Any]):
        context["results"][step_id] = results

        first = results[0] if isinstance(results, list) else results
        value = self._extract_value(first)

        for edge in self.graph_store.get_edges(src=step_id, kind=EdgeKind.CUSTOM):
            if edge.data.get("type") == "result_variable":
                context["variables"][edge.data["variable"]] = value
        return True

    # ----------------------------------------------------------- convenience
    async def execute_plan_by_id(self, plan_id: str, variables: Optional[Dict[str, Any]] = None):
        node = self.graph_store.get_node(plan_id)
        if node is None:
            raise ValueError(f"Plan {plan_id} not found")
        plan = UniversalPlan(title=node.data.get("title", "Plan"), id=plan_id, graph=self.graph_store)
        return await self.execute_plan(plan, variables)
