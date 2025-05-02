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
from typing import Dict, Iterable

from chuk_ai_planner.models import PlanNode, PlanStep
from chuk_ai_planner.models.edges import ParentChildEdge, StepEdge
from chuk_ai_planner.store.base import GraphStore


# --------------------------------------------------------------------- helpers
def _dump_steps(
    plan_node: PlanNode,
    step_map: Dict[str, "planner.plan._Step"],
    graph: GraphStore,
) -> None:
    """Write PlanStep nodes + PARENT_CHILD edges."""
    for step in step_map.values():
        ps = PlanStep(
            id=step.id,
            data={"description": step.title, "index": step.index},
        )
        graph.add_node(ps)

        # link plan → step
        graph.add_edge(ParentChildEdge(src=plan_node.id, dst=ps.id))

        # link parent-step → child-step (skip root container)
        parent = step.parent
        if parent and parent.title != "[ROOT]":
            graph.add_edge(ParentChildEdge(src=parent.id, dst=ps.id))


def _dump_dependencies(
    step_map: Dict[str, "planner.plan._Step"],
    graph: GraphStore,
) -> None:
    """Write STEP_ORDER(src→dst) edges based on `after` lists."""
    for st in step_map.values():
        for dep_idx in st.after:
            src = step_map.get(dep_idx)        # the prerequisite step
            if src:
                graph.add_edge(StepEdge(src=src.id, dst=st.id))


# --------------------------------------------------------------------- public API
def persist_full_plan(
    plan_node: PlanNode,
    step_map: Dict[str, "planner.plan._Step"],
    graph: GraphStore,
) -> None:
    """
    Write the complete Plan *once* — used by `Plan.save()`.

    * `plan_node`   – already added to the graph by the caller
    * `step_map`    – { "1.2": _Step(...), ... } after numbering
    * `graph`       – any GraphStore implementation
    """
    _dump_steps(plan_node, step_map, graph)
    _dump_dependencies(step_map, graph)


def persist_single_step(
    step_obj: "planner.plan._Step",
    parent_step: "planner.plan._Step | None",
    graph: GraphStore,
    plan_id: str,
) -> None:
    """
    Persist *one* newly added step after the plan was already saved.

    Called by `Plan.add_step()`.
    """
    new_node = PlanStep(
        id=step_obj.id,
        data={"description": step_obj.title, "index": step_obj.index},
    )
    graph.add_node(new_node)

    # link plan → step
    graph.add_edge(ParentChildEdge(src=plan_id, dst=new_node.id))

    # link parent-step → child-step if not root
    if parent_step and parent_step.title != "[ROOT]":
        graph.add_edge(ParentChildEdge(src=parent_step.id, dst=new_node.id))

    # dependency edges for the new step
    for dep_idx in step_obj.after:
        dep = (parent_step._index_map if hasattr(parent_step, "_index_map") else {}).get(dep_idx)  # type: ignore[attr-defined]
        if dep:
            graph.add_edge(StepEdge(src=dep.id, dst=step_obj.id))
