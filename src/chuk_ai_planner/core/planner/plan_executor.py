# chuk_ai_planner/planner/plan_executor.py
"""
PlanExecutor
============

Utility class that lives *below* the high-level Plan DSL and *above* the
GraphAwareToolProcessor.  It provides three things:

1. ``get_plan_steps(plan_id)``
   - Returns **all** ``PLAN_STEP`` nodes under a ``PlanNode`` — including
   nested sub-steps such as *1.2.3* — by doing a DFS over ``PARENT_CHILD``
   edges.

2. ``determine_execution_order(steps)``
   - Topologically batches the steps so that every step whose
   dependencies are met can run in parallel.
   - An edge ``STEP_ORDER(src → dst)`` means *dst depends on src*.

3. ``execute_step(...)``
   - Runs the tool-calls explicitly linked to one step, emits start /
   completion session-events, and returns a list of ``ToolResult``-like
   payloads.

Nothing in here is author-facing; it is an internal helper that lets the
*Processor* stay slim.
"""

from __future__ import annotations
import json
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from pydantic import BaseModel

# graph imports
from chuk_ai_planner.core.graph import PlanStep, ToolCall, GraphNode, NodeType
from chuk_ai_planner.core.graph import EdgeType, StepStatus
from chuk_ai_planner.core.store.base import GraphStore
from chuk_session_manager.models.event_type import EventType  # type: ignore[import-untyped]

# Import serialization utilities
from chuk_ai_planner.utils.serialization import unfreeze_data


# --------------------------------------------------------------------------- Constants
STATUS_STARTED = "started"
TOOL_TYPE_FUNCTION = "function"


# --------------------------------------------------------------------------- Pydantic models
class FunctionCall(BaseModel):
    """Function call details in a tool call"""

    name: str
    arguments: str  # JSON string


class ToolCallPayload(BaseModel):
    """Tool call structure for event callbacks"""

    id: str
    type: str  # e.g., "function"
    function: FunctionCall


class StepEventPayload(BaseModel):
    """Event payload for step start/completion"""

    step_id: str
    status: str
    description: Optional[str] = None
    tools_executed: Optional[int] = None


# --------------------------------------------------------------------------- helpers
def _hier_key(idx: str) -> tuple[int, ...]:
    """
    Convert a dotted index string ``\"1.2.10\"`` into a sortable tuple
    ``(1, 2, 10)``.  Non-numeric parts are ignored.
    """
    return tuple(int(p) for p in idx.split(".") if p.isdigit())


# --------------------------------------------------------------------------- main class
class PlanExecutor:
    """
    Extracts plan steps, figures out safe parallel batches, and executes a
    single step by dispatching its linked tool-calls.
    """

    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    # ...................................................................... steps
    async def get_plan_steps(self, plan_id: str) -> List[GraphNode]:
        """
        Depth-first collect **all** ``PLAN_STEP`` nodes that are (directly or
        indirectly) children of the given ``PlanNode``.
        """
        out: List[GraphNode] = []
        stack: List[str] = [plan_id]

        while stack:
            parent = stack.pop()
            for edge in await self.graph_store.get_edges(
                src=parent, kind=EdgeType.PARENT_CHILD
            ):
                node = await self.graph_store.get_node(edge.dst)
                if not node:
                    continue

                if node.kind == NodeType.PLAN_STEP:
                    out.append(node)

                # even a PLAN_STEP can own sub-steps
                stack.append(node.id)

        # stable sort by hierarchical index, e.g. 1 < 1.2 < 1.10 < 2
        return sorted(
            out,
            key=lambda n: _hier_key(str(n.index)) if isinstance(n, PlanStep) else [0],
        )

    # ...................................................................... batching
    async def determine_execution_order(
        self, steps: List[GraphNode]
    ) -> List[List[str]]:
        """
        Return ``[[batch1-ids], [batch2-ids], …]`` where every batch can be
        executed in parallel and all explicit ``STEP_ORDER`` dependencies are
        respected.
        """
        dependencies: Dict[str, Set[str]] = {s.id: set() for s in steps}
        dependents: Dict[str, Set[str]] = {s.id: set() for s in steps}

        for step in steps:
            for edge in await self.graph_store.get_edges(
                src=step.id, kind=EdgeType.STEP_ORDER
            ):
                # edge.src (=step.id) must run *before* edge.dst
                dependencies[edge.dst].add(step.id)
                dependents[step.id].add(edge.dst)

        ready = [sid for sid, deps in dependencies.items() if not deps]
        if not ready and steps:
            # cycle or missing edges – fallback: pick the first by sort-order
            ready = [steps[0].id]

        batches: List[List[str]] = []
        processed: Set[str] = set()

        while ready:
            # Filter out already processed steps to prevent infinite loops
            ready = [sid for sid in ready if sid not in processed]
            if not ready:  # pragma: no cover - defensive check, shouldn't be reached
                break

            batches.append(ready)
            processed.update(ready)
            next_ready: List[str] = []

            for sid in ready:
                for dep in list(dependents[sid]):
                    dependencies[dep].discard(sid)
                    if not dependencies[dep] and dep not in processed:
                        next_ready.append(dep)

            ready = next_ready

        return batches

    # ...................................................................... execute one
    async def execute_step(
        self,
        step_id: str,
        assistant_node_id: str,
        parent_event_id: str,
        create_child_event: Callable[[EventType, Dict[str, Any], str], Any],
        process_tool_call: Callable[
            [Dict[str, Any], str, Optional[str]], Awaitable[Any]
        ],
    ) -> List[Any]:
        """
        1. Emit \"started\" summary event.
        2. For each linked ``PLAN_LINK`` → ``ToolCall`` execute the tool via
           *process_tool_call*.
        3. Emit \"completed\" summary event.
        4. Return list of tool results.
        """
        step_node = await self.graph_store.get_node(step_id)
        if not step_node or step_node.kind != NodeType.PLAN_STEP:
            raise ValueError(f"Invalid plan step {step_id!r}")

        # Type-safe access to PlanStep (guaranteed by check above)
        assert isinstance(step_node, PlanStep), (
            "step_node must be PlanStep after kind check"
        )
        description = step_node.description

        start_event_payload = StepEventPayload(
            step_id=step_id, description=description, status=STATUS_STARTED
        )
        start_evt = create_child_event(
            EventType.SUMMARY,
            start_event_payload.model_dump(),
            parent_event_id,
        )

        results: List[Any] = []
        for edge in await self.graph_store.get_edges(
            src=step_id, kind=EdgeType.PLAN_LINK
        ):
            tool_node = await self.graph_store.get_node(edge.dst)
            if not tool_node or tool_node.kind != NodeType.TOOL_CALL:
                continue

            # Get tool data and unfreeze for JSON serialization
            if isinstance(tool_node, ToolCall):
                tool_name = tool_node.name
                tool_args = tool_node.args

            # Unfreeze the args for JSON serialization
            unfrozen_args = unfreeze_data(tool_args)

            tool_call_payload = ToolCallPayload(
                id=tool_node.id,
                type=TOOL_TYPE_FUNCTION,
                function=FunctionCall(
                    name=tool_name, arguments=json.dumps(unfrozen_args)
                ),
            )
            res = await process_tool_call(
                tool_call_payload.model_dump(), start_evt.id, assistant_node_id
            )
            results.append(res)

        completed_event_payload = StepEventPayload(
            step_id=step_id,
            status=StepStatus.COMPLETED.value,
            tools_executed=len(results),
        )
        create_child_event(
            EventType.SUMMARY,
            completed_event_payload.model_dump(),
            parent_event_id,
        )
        return results
