import asyncio
import json
from chuk_session_manager.models.event_type import EventType
import pytest
from collections import defaultdict

# imports
from chuk_ai_planner.planner.plan_executor import PlanExecutor
from chuk_ai_planner.models import GraphNode, NodeKind
from chuk_ai_planner.models.edges import EdgeKind, GraphEdge
from chuk_ai_planner.store.memory import InMemoryGraphStore


# --------------------------------------------------------------------- helpers
def _mk_step(i: str, desc: str) -> GraphNode:
    """Create a bare PlanStep node with dotted index."""
    return GraphNode(
        kind=NodeKind.PLAN_STEP,
        data={"description": desc, "index": i}
    )


@pytest.fixture
def graph():
    return InMemoryGraphStore()


@pytest.fixture
def executor(graph):
    return PlanExecutor(graph)



# --------------------------------------------------------------------- tests
def test_get_plan_steps_collects_depth(graph, executor):
    """
    plan
      ├─ 1
      │   └─ 1.1
      └─ 2
    """
    plan = GraphNode(kind=NodeKind.PLAN, data={})
    s1   = _mk_step("1", "A")
    s11  = _mk_step("1.1", "A.1")
    s2   = _mk_step("2", "B")

    for n in (plan, s1, s11, s2):
        graph.add_node(n)

    # hierarchy
    graph.add_edge(GraphEdge(kind=EdgeKind.PARENT_CHILD, src=plan.id, dst=s1.id))
    graph.add_edge(GraphEdge(kind=EdgeKind.PARENT_CHILD, src=s1.id,   dst=s11.id))
    graph.add_edge(GraphEdge(kind=EdgeKind.PARENT_CHILD, src=plan.id, dst=s2.id))

    steps = executor.get_plan_steps(plan.id)
    assert [n.data["index"] for n in steps] == ["1", "1.1", "2"]


def test_determine_execution_batches(graph, executor):
    """
    1  -> 3
    2  -> 3          →  batches: [1,2] then [3]
    """
    s1 = _mk_step("1", "A"); graph.add_node(s1)
    s2 = _mk_step("2", "B"); graph.add_node(s2)
    s3 = _mk_step("3", "C"); graph.add_node(s3)

    # deps
    graph.add_edge(GraphEdge(kind=EdgeKind.STEP_ORDER, src=s1.id, dst=s3.id))
    graph.add_edge(GraphEdge(kind=EdgeKind.STEP_ORDER, src=s2.id, dst=s3.id))

    batches = executor.determine_execution_order([s1, s2, s3])
    assert batches == [[s1.id, s2.id], [s3.id]]


@pytest.mark.asyncio
async def test_execute_step_runs_tool_calls(graph, executor):
    """
    A single step linked to two ToolCall nodes should invoke process_tool_call
    twice and emit 'started'/'completed' events via create_child_event.
    """
    step  = _mk_step("1", "Run tools"); graph.add_node(step)

    tool1 = GraphNode(kind=NodeKind.TOOL_CALL,
                      data={"name": "dummy", "args": {"x": 1}})
    tool2 = GraphNode(kind=NodeKind.TOOL_CALL,
                      data={"name": "dummy", "args": {"x": 2}})
    graph.add_node(tool1); graph.add_node(tool2)

    graph.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step.id, dst=tool1.id))
    graph.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step.id, dst=tool2.id))

    calls = []

    async def _proc_tool_call(tc, parent_evt_id, _assistant):
        calls.append(json.loads(tc["function"]["arguments"]))
        return {"ok": True}

    events = defaultdict(int)

    def _create_evt(et, msg, parent):
        idx = events[et] = events[et] + 1          # increment + keep count
        return type("Evt", (), {"id": f"evt{idx}"})()   # tiny mock with .id

    results = await executor.execute_step(
        step_id=step.id,
        assistant_node_id="assistant",
        parent_event_id="root_evt",
        create_child_event=_create_evt,
        process_tool_call=_proc_tool_call
    )

    assert [c["x"] for c in calls] == [1, 2]
    assert events[EventType.SUMMARY] == 2          # started + completed
    assert len(results) == 2
