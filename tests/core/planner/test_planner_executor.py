import json
from chuk_session_manager.models.event_type import EventType
import pytest
from collections import defaultdict

# imports
from chuk_ai_planner.core.planner.plan_executor import PlanExecutor
from chuk_ai_planner.core.graph import (
    PlanNode,
    PlanStep,
    ToolCall,
    ParentChildEdge,
    StepEdge,
)
from chuk_ai_planner.core.store.memory import InMemoryGraphStore


# --------------------------------------------------------------------- helpers
def _mk_step(i: str, desc: str) -> PlanStep:
    """Create a bare PlanStep node with dotted index."""
    return PlanStep(description=desc, index=i)


@pytest.fixture
def graph():
    return InMemoryGraphStore()


@pytest.fixture
def executor(graph):
    return PlanExecutor(graph)


# --------------------------------------------------------------------- tests
@pytest.mark.asyncio
async def test_get_plan_steps_collects_depth(graph, executor):
    """
    plan
      ├─ 1
      │   └─ 1.1
      └─ 2
    """
    plan = PlanNode(title="Test Plan")
    s1 = _mk_step("1", "A")
    s11 = _mk_step("1.1", "A.1")
    s2 = _mk_step("2", "B")

    for n in (plan, s1, s11, s2):
        await graph.add_node(n)

    # hierarchy
    await graph.add_edge(ParentChildEdge(src=plan.id, dst=s1.id))
    await graph.add_edge(ParentChildEdge(src=s1.id, dst=s11.id))
    await graph.add_edge(ParentChildEdge(src=plan.id, dst=s2.id))

    steps = await executor.get_plan_steps(plan.id)
    assert [n.index for n in steps] == ["1", "1.1", "2"]


@pytest.mark.asyncio
async def test_determine_execution_batches(graph, executor):
    """
    1  -> 3
    2  -> 3          →  batches: [1,2] then [3]
    """
    s1 = _mk_step("1", "A")
    await graph.add_node(s1)
    s2 = _mk_step("2", "B")
    await graph.add_node(s2)
    s3 = _mk_step("3", "C")
    await graph.add_node(s3)

    # deps
    await graph.add_edge(StepEdge(src=s1.id, dst=s3.id))
    await graph.add_edge(StepEdge(src=s2.id, dst=s3.id))

    batches = await executor.determine_execution_order([s1, s2, s3])
    assert batches == [[s1.id, s2.id], [s3.id]]


@pytest.mark.asyncio
async def test_execute_step_runs_tool_calls(graph, executor):
    """
    A single step linked to two ToolCall nodes should invoke process_tool_call
    twice and emit 'started'/'completed' events via create_child_event.
    """
    step = _mk_step("1", "Run tools")
    await graph.add_node(step)

    tool1 = ToolCall(name="dummy", args={"x": 1})
    tool2 = ToolCall(name="dummy", args={"x": 2})
    await graph.add_node(tool1)
    await graph.add_node(tool2)

    from chuk_ai_planner.core.graph import PlanLinkEdge

    await graph.add_edge(PlanLinkEdge(src=step.id, dst=tool1.id))
    await graph.add_edge(PlanLinkEdge(src=step.id, dst=tool2.id))

    calls = []

    async def _proc_tool_call(tc, parent_evt_id, _assistant):
        calls.append(json.loads(tc["function"]["arguments"]))
        return {"ok": True}

    events = defaultdict(int)

    def _create_evt(et, msg, parent):
        idx = events[et] = events[et] + 1  # increment + keep count
        return type("Evt", (), {"id": f"evt{idx}"})()  # tiny mock with .id

    results = await executor.execute_step(
        step_id=step.id,
        assistant_node_id="assistant",
        parent_event_id="root_evt",
        create_child_event=_create_evt,
        process_tool_call=_proc_tool_call,
    )

    assert [c["x"] for c in calls] == [1, 2]
    assert events[EventType.SUMMARY] == 2  # started + completed
    assert len(results) == 2


@pytest.mark.asyncio
async def test_get_plan_steps_handles_missing_node(graph, executor):
    """Test that get_plan_steps skips edges pointing to non-existent nodes."""
    plan = PlanNode(title="Test Plan")
    s1 = _mk_step("1", "A")

    await graph.add_node(plan)
    await graph.add_node(s1)

    # Create edge to existing step
    await graph.add_edge(ParentChildEdge(src=plan.id, dst=s1.id))

    # Create edge to non-existent node (simulates deleted node)
    await graph.add_edge(ParentChildEdge(src=plan.id, dst="nonexistent-node-id"))

    steps = await executor.get_plan_steps(plan.id)
    # Should only return the valid step, skipping the missing one
    assert len(steps) == 1
    assert steps[0].index == "1"


@pytest.mark.asyncio
async def test_determine_execution_order_handles_cycle(graph, executor):
    """Test that determine_execution_order handles cycles by picking first step."""
    s1 = _mk_step("1", "A")
    s2 = _mk_step("2", "B")
    s3 = _mk_step("3", "C")

    await graph.add_node(s1)
    await graph.add_node(s2)
    await graph.add_node(s3)

    # Create a cycle: 1 -> 2 -> 3 -> 1
    await graph.add_edge(StepEdge(src=s1.id, dst=s2.id))
    await graph.add_edge(StepEdge(src=s2.id, dst=s3.id))
    await graph.add_edge(StepEdge(src=s3.id, dst=s1.id))

    batches = await executor.determine_execution_order([s1, s2, s3])
    # Should use fallback and pick the first step by sort order
    assert len(batches) > 0
    assert s1.id in batches[0]


@pytest.mark.asyncio
async def test_execute_step_with_invalid_step_id(graph, executor):
    """Test that execute_step raises ValueError for invalid step_id."""
    events = defaultdict(int)

    def _create_evt(et, msg, parent):
        idx = events[et] = events[et] + 1
        return type("Evt", (), {"id": f"evt{idx}"})()

    async def _proc_tool_call(tc, parent_evt_id, _assistant):
        return {"ok": True}

    with pytest.raises(ValueError, match="Invalid plan step"):
        await executor.execute_step(
            step_id="nonexistent-step-id",
            assistant_node_id="assistant",
            parent_event_id="root_evt",
            create_child_event=_create_evt,
            process_tool_call=_proc_tool_call,
        )


@pytest.mark.asyncio
async def test_execute_step_with_non_planstep_node(graph, executor):
    """Test that execute_step handles non-PlanStep nodes gracefully."""
    # Create a non-PlanStep node (e.g., a PlanNode)
    plan_node = PlanNode(title="Not a step")
    await graph.add_node(plan_node)

    events = defaultdict(int)

    def _create_evt(et, msg, parent):
        idx = events[et] = events[et] + 1
        return type("Evt", (), {"id": f"evt{idx}"})()

    async def _proc_tool_call(tc, parent_evt_id, _assistant):
        return {"ok": True}

    # This should raise ValueError because it's not a PLAN_STEP
    with pytest.raises(ValueError, match="Invalid plan step"):
        await executor.execute_step(
            step_id=plan_node.id,
            assistant_node_id="assistant",
            parent_event_id="root_evt",
            create_child_event=_create_evt,
            process_tool_call=_proc_tool_call,
        )


@pytest.mark.asyncio
async def test_execute_step_skips_invalid_tool_nodes(graph, executor):
    """Test that execute_step skips tool edges pointing to non-ToolCall nodes."""
    step = _mk_step("1", "Run tools")
    await graph.add_node(step)

    # Add a valid tool call
    tool1 = ToolCall(name="dummy", args={"x": 1})
    await graph.add_node(tool1)

    # Add a non-ToolCall node that step incorrectly links to
    other_node = PlanNode(title="Not a tool")
    await graph.add_node(other_node)

    from chuk_ai_planner.core.graph import PlanLinkEdge

    await graph.add_edge(PlanLinkEdge(src=step.id, dst=tool1.id))
    await graph.add_edge(PlanLinkEdge(src=step.id, dst=other_node.id))

    calls = []

    async def _proc_tool_call(tc, parent_evt_id, _assistant):
        calls.append(json.loads(tc["function"]["arguments"]))
        return {"ok": True}

    events = defaultdict(int)

    def _create_evt(et, msg, parent):
        idx = events[et] = events[et] + 1
        return type("Evt", (), {"id": f"evt{idx}"})()

    results = await executor.execute_step(
        step_id=step.id,
        assistant_node_id="assistant",
        parent_event_id="root_evt",
        create_child_event=_create_evt,
        process_tool_call=_proc_tool_call,
    )

    # Should only execute the valid tool call, skipping the non-ToolCall node
    assert len(calls) == 1
    assert calls[0]["x"] == 1
    assert len(results) == 1


@pytest.mark.asyncio
async def test_determine_execution_order_all_steps_processed(graph, executor):
    """Test determine_execution_order processes all steps and exits cleanly."""
    # Create a simple linear chain: 1 -> 2 -> 3
    s1 = _mk_step("1", "A")
    s2 = _mk_step("2", "B")
    s3 = _mk_step("3", "C")

    await graph.add_node(s1)
    await graph.add_node(s2)
    await graph.add_node(s3)

    # Create dependencies: 1 -> 2 -> 3
    await graph.add_edge(StepEdge(src=s1.id, dst=s2.id))
    await graph.add_edge(StepEdge(src=s2.id, dst=s3.id))

    batches = await executor.determine_execution_order([s1, s2, s3])

    # Should have 3 batches (linear execution)
    assert len(batches) == 3
    assert batches[0] == [s1.id]
    assert batches[1] == [s2.id]
    assert batches[2] == [s3.id]


@pytest.mark.asyncio
async def test_determine_execution_order_complex_cycle_safety(graph, executor):
    """Test that the safety break (line 123) prevents infinite loops in edge cases."""
    # Create a more complex scenario with multiple cycles
    s1 = _mk_step("1", "A")
    s2 = _mk_step("2", "B")
    s3 = _mk_step("3", "C")
    s4 = _mk_step("4", "D")

    await graph.add_node(s1)
    await graph.add_node(s2)
    await graph.add_node(s3)
    await graph.add_node(s4)

    # Create a complex dependency graph with cycles
    # 1 -> 2 -> 3 -> 1 (cycle)
    # 4 depends on 2
    await graph.add_edge(StepEdge(src=s1.id, dst=s2.id))
    await graph.add_edge(StepEdge(src=s2.id, dst=s3.id))
    await graph.add_edge(StepEdge(src=s3.id, dst=s1.id))  # Creates cycle
    await graph.add_edge(StepEdge(src=s2.id, dst=s4.id))

    batches = await executor.determine_execution_order([s1, s2, s3, s4])

    # Should complete without infinite loop
    # Will use fallback to pick first step, then process what it can
    assert len(batches) > 0
    assert len(batches) <= 4  # At most 4 batches (one per step)
