# src/chuk_ai_planner/planner/plan_executor.py
"""
PlanExecutor
============

Utility class that lives *below* the high–level Plan DSL and *above* the
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


# graph imports
from chuk_ai_planner.models import GraphNode, NodeKind
from chuk_ai_planner.models.edges import EdgeKind
from chuk_ai_planner.store.base import GraphStore
from a2a_session_manager.models.event_type import EventType




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
    def get_plan_steps(self, plan_id: str) -> List[GraphNode]:
        """
        Depth-first collect **all** ``PLAN_STEP`` nodes that are (directly or
        indirectly) children of the given ``PlanNode``.
        """
        out: List[GraphNode] = []
        stack: List[str] = [plan_id]

        while stack:
            parent = stack.pop()
            for edge in self.graph_store.get_edges(src=parent, kind=EdgeKind.PARENT_CHILD):
                node = self.graph_store.get_node(edge.dst)
                if not node:
                    continue

                if node.kind == NodeKind.PLAN_STEP:
                    out.append(node)

                # even a PLAN_STEP can own sub-steps
                stack.append(node.id)

        # stable sort by hierarchical index, e.g. 1 < 1.2 < 1.10 < 2
        return sorted(out, key=lambda n: _hier_key(str(n.data.get("index", ""))))

    # ...................................................................... batching
    def determine_execution_order(self, steps: List[GraphNode]) -> List[List[str]]:
        """
        Return ``[[batch1-ids], [batch2-ids], …]`` where every batch can be
        executed in parallel and all explicit ``STEP_ORDER`` dependencies are
        respected.
        """
        dependencies: Dict[str, Set[str]] = {s.id: set() for s in steps}
        dependents:   Dict[str, Set[str]] = {s.id: set() for s in steps}

        for step in steps:
            for edge in self.graph_store.get_edges(src=step.id, kind=EdgeKind.STEP_ORDER):
                # edge.src (=step.id) must run *before* edge.dst
                dependencies[edge.dst].add(step.id)
                dependents[step.id].add(edge.dst)

        ready = [sid for sid, deps in dependencies.items() if not deps]
        if not ready and steps:
            # cycle or missing edges – fallback: pick the first by sort-order
            ready = [steps[0].id]

        batches: List[List[str]] = []
        while ready:
            batches.append(ready)
            next_ready: List[str] = []

            for sid in ready:
                for dep in list(dependents[sid]):
                    dependencies[dep].discard(sid)
                    if not dependencies[dep]:
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
        process_tool_call: Callable[[Dict[str, Any], str, Optional[str]], Awaitable[Any]],
    ) -> List[Any]:
        """
        1. Emit \"started\" summary event.  
        2. For each linked ``PLAN_LINK`` → ``ToolCall`` execute the tool via
           *process_tool_call*.  
        3. Emit \"completed\" summary event.  
        4. Return list of tool results.
        """
        step_node = self.graph_store.get_node(step_id)
        if not step_node or step_node.kind != NodeKind.PLAN_STEP:
            raise ValueError(f"Invalid plan step {step_id!r}")

        start_evt = create_child_event(
            EventType.SUMMARY,
            {
                "step_id": step_id,
                "description": step_node.data.get("description", "Unknown step"),
                "status": "started",
            },
            parent_event_id,
        )

        results: List[Any] = []
        for edge in self.graph_store.get_edges(src=step_id, kind=EdgeKind.PLAN_LINK):
            tool_node = self.graph_store.get_node(edge.dst)
            if not tool_node or tool_node.kind != NodeKind.TOOL_CALL:
                continue

            tool_call = {
                "id": tool_node.id,
                "type": "function",
                "function": {
                    "name": tool_node.data.get("name"),
                    "arguments": json.dumps(tool_node.data.get("args", {})),
                },
            }
            res = await process_tool_call(tool_call, start_evt.id, assistant_node_id)
            results.append(res)

        create_child_event(
            EventType.SUMMARY,
            {
                "step_id": step_id,
                "status": "completed",
                "tools_executed": len(results),
            },
            parent_event_id,
        )
        return results
