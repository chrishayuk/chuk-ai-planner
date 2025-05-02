"""
tests/test_plan.py
==================

Unit-tests for chuk_ai_planner.planner.plan.Plan
"""

import re
from chuk_ai_planner.planner import Plan
from chuk_ai_planner.models import NodeKind, PlanStep


# --------------------------------------------------------------------- helpers
def _step_nodes(plan: Plan):
    """Return list of (<index>, PlanStep-node) tuples sorted by index."""
    nodes = [
        (n.data["index"], n)
        for n in plan.graph.nodes.values()
        if isinstance(n, PlanStep) and n.kind == NodeKind.PLAN_STEP
    ]
    return sorted(nodes, key=lambda t: tuple(int(p) for p in t[0].split(".")))


# --------------------------------------------------------------------- tests
def test_simple_hierarchy_and_outline():
    plan = (
        Plan("Demo")
          .step("Gather requirements").up()
          .step("Draft design").up()
          .step("Write code", after=["1", "2"])
    )
    out = plan.outline()

    # indices & order in outline
    assert re.search(r"^\s*1\s+Gather requirements", out, re.M)
    assert re.search(r"^\s*2\s+Draft design", out, re.M)
    assert re.search(r"^\s*3\s+Write code.*depends on \['1', '2'\]", out, re.M)

    # plan.save() should persist three PLAN_STEP nodes
    plan_id = plan.save()
    steps = _step_nodes(plan)
    assert len(steps) == 3
    assert {idx for idx, _ in steps} == {"1", "2", "3"}
    assert plan_id == plan.id


def test_add_step_runtime_and_persistence():
    plan = (Plan("Nested")
              .step("Prepare").step("Step-A").up().up()
              .step("Finish"))

    plan_id = plan.save()

    # add new sub-step under "1" (Prepare)
    idx = plan.add_step("Step-B", parent="1", after=["1.1"])
    assert idx == "1.2"                       # correct hierarchical index

    # graph now has 4 PlanStep nodes
    steps = _step_nodes(plan)
    assert len(steps) == 4
    assert any(idx == "1.2" and node.data["description"] == "Step-B"
               for idx, node in steps)

    # dependency persisted?
    step_b = next(node for i, node in steps if i == "1.2")
    assert step_b.data.get("index") == "1.2"


def test_after_dependencies_are_stored():
    plan = (Plan("Deps")
              .step("First").up()
              .step("Second").up()
              .step("Third", after=["1", "2"]))
    plan.save()
    third = next(node for idx, node in _step_nodes(plan) if idx == "3")
    # "after" list is kept in node data
    assert third.data["index"] == "3"
    assert third.data.get("description") == "Third"
