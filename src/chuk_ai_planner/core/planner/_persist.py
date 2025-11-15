# chuk_ai_planner/planner/_persist.py
"""
planner._persist
================

Low-level helpers that take an *in-memory* step tree and write it into a
GraphStore.  They are intentionally kept tiny and free of Plan-DSL
details so they can be re-used elsewhere.

Public API
----------

* `persist_full_plan(plan_node, step_tree, graph)`
* `persist_single_step(step_obj, parent_step, graph)`

Both helpers are imported by `plan.py`.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from chuk_ai_planner.core.planner.plan import _Step

from chuk_ai_planner.core.graph import PlanNode, PlanStep
from chuk_ai_planner.core.graph import ParentChildEdge, StepEdge
from chuk_ai_planner.core.store.base import GraphStore


# --------------------------------------------------------------------- helpers
async def _dump_steps(
    plan_node: PlanNode,
    step_map: Dict[str, "_Step"],
    graph: GraphStore,
) -> None:
    """Write PlanStep nodes + PARENT_CHILD edges."""
    for step in step_map.values():
        # Use typed fields instead of data dict
        ps = PlanStep(id=step.id, description=step.title, index=step.index)
        await graph.add_node(ps)

        # link plan → step
        await graph.add_edge(ParentChildEdge(src=plan_node.id, dst=ps.id))

        # link parent-step → child-step (skip root container)
        parent = step.parent
        if parent and parent.title != "[ROOT]":
            await graph.add_edge(ParentChildEdge(src=parent.id, dst=ps.id))


async def _dump_dependencies(
    step_map: Dict[str, "_Step"],
    graph: GraphStore,
) -> None:
    """Write STEP_ORDER(src→dst) edges based on `after` lists."""
    for st in step_map.values():
        for dep_idx in st.after:
            src = step_map.get(dep_idx)  # the prerequisite step
            if src:
                await graph.add_edge(StepEdge(src=src.id, dst=st.id))


# --------------------------------------------------------------------- public API
async def persist_full_plan(
    plan_node: PlanNode,
    step_map: Dict[str, "_Step"],
    graph: GraphStore,
) -> None:
    """
    Write the complete Plan *once* — used by `Plan.save()`.

    * `plan_node`   - already added to the graph by the caller
    * `step_map`    - { "1.2": _Step(...), ... } after numbering
    * `graph`       - any GraphStore implementation
    """
    await _dump_steps(plan_node, step_map, graph)
    await _dump_dependencies(step_map, graph)


async def persist_single_step(
    step_obj: "_Step",
    parent_step: "_Step | None",
    graph: GraphStore,
    plan_id: str,
) -> None:
    """
    Persist *one* newly added step after the plan was already saved.

    Called by `Plan.add_step()`.
    """
    # Use typed fields instead of data dict
    new_node = PlanStep(
        id=step_obj.id, description=step_obj.title, index=step_obj.index
    )
    await graph.add_node(new_node)

    # link plan → step
    await graph.add_edge(ParentChildEdge(src=plan_id, dst=new_node.id))

    # link parent-step → child-step if not root
    if parent_step and parent_step.title != "[ROOT]":
        await graph.add_edge(ParentChildEdge(src=parent_step.id, dst=new_node.id))

    # dependency edges for the new step
    for dep_idx in step_obj.after:
        if parent_step:
            dep = (
                parent_step._index_map if hasattr(parent_step, "_index_map") else {}
            ).get(dep_idx)  # type: ignore[attr-defined]
            if dep:
                await graph.add_edge(StepEdge(src=dep.id, dst=step_obj.id))
