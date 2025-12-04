# tests/utils/test_visualization.py
import pytest
from chuk_ai_planner.utils.visualization import (
    print_session_events,
    print_graph_structure,
    _print_plan_structure,
    _print_assistant_structure,
)

from chuk_session_manager.models.session import Session, SessionEvent
from chuk_session_manager.models.event_type import EventType
from chuk_session_manager.models.event_source import EventSource

from chuk_ai_planner.core.store.memory import InMemoryGraphStore


@pytest.mark.asyncio
async def test_print_session_events_nested_and_types(capsys):
    # Create a session with a root MESSAGE event and a child TOOL_CALL event
    session = Session()
    root = SessionEvent(
        message={"content": "User says hello"},
        type=EventType.MESSAGE,
        source=EventSource.USER,
    )
    session.events.append(root)
    child = SessionEvent(
        message={"tool": "weather", "error": None},
        type=EventType.TOOL_CALL,
        source=EventSource.LLM,
        metadata={"parent_event_id": root.id},
    )
    session.events.append(child)

    print_session_events(session)
    captured = capsys.readouterr().out

    # Check header and counts
    assert "==== SESSION EVENTS (2) ====" in captured
    # Check root event printed
    assert f"• {root.type.value}" in captured
    assert f"id={root.id}" in captured
    # Check child event printed with indentation and tool info
    assert "  • tool_call" in captured
    assert "⇒ weather    error=None" in captured


@pytest.mark.asyncio
async def test_print_graph_structure_basic(capsys):
    # Setup in-memory graph store with a session and a plan
    from chuk_ai_planner.core.graph import SessionNode, PlanNode, ParentChildEdge

    store = InMemoryGraphStore()
    session_node = SessionNode(id="s1", name="Test Session")
    plan_node = PlanNode(id="p1", title="Test Plan")
    await store.add_node(session_node)
    await store.add_node(plan_node)
    # Connect session -> plan
    edge = ParentChildEdge(src="s1", dst="p1")
    await store.add_edge(edge)

    await print_graph_structure(store)
    captured = capsys.readouterr().out

    # Check summary lines
    assert "==== GRAPH STRUCTURE ====" in captured
    assert "Total nodes: 2" in captured
    assert "Total edges: 1" in captured
    # Check node counts by type
    assert "session: 1" in captured
    assert "plan: 1" in captured
    # Check session and plan hierarchy
    assert f"Session: {session_node!r}" in captured
    # plan child printed under session
    assert "└── plan:" in captured
    assert f"{plan_node!r}" in captured


@pytest.mark.asyncio
async def test_print_session_events_with_summary_description(capsys):
    """Test printing SUMMARY events with description"""
    session = Session()
    summary_event = SessionEvent(
        message={"description": "Test description"},
        type=EventType.SUMMARY,
        source=EventSource.SYSTEM,
    )
    session.events.append(summary_event)

    print_session_events(session)
    captured = capsys.readouterr().out

    assert "Test description" in captured


@pytest.mark.asyncio
async def test_print_session_events_with_summary_note(capsys):
    """Test printing SUMMARY events with note"""
    session = Session()
    summary_event = SessionEvent(
        message={"note": "Important note"},
        type=EventType.SUMMARY,
        source=EventSource.SYSTEM,
    )
    session.events.append(summary_event)

    print_session_events(session)
    captured = capsys.readouterr().out

    assert "Note: Important note" in captured


@pytest.mark.asyncio
async def test_print_session_events_with_summary_step_id(capsys):
    """Test printing SUMMARY events with step_id"""
    session = Session()
    summary_event = SessionEvent(
        message={"step_id": "step_123", "status": "completed"},
        type=EventType.SUMMARY,
        source=EventSource.SYSTEM,
    )
    session.events.append(summary_event)

    print_session_events(session)
    captured = capsys.readouterr().out

    assert "Step step_123: completed" in captured


@pytest.mark.asyncio
async def test_print_session_events_with_error(capsys):
    """Test printing events with error messages"""
    session = Session()
    error_event = SessionEvent(
        message={"error": "Something went wrong"},
        type=EventType.MESSAGE,
        source=EventSource.SYSTEM,
    )
    session.events.append(error_event)

    print_session_events(session)
    captured = capsys.readouterr().out

    assert "Error: Something went wrong" in captured


@pytest.mark.asyncio
async def test_print_graph_structure_with_plan_steps(capsys):
    """Test printing graph structure with plan steps and tools"""
    from chuk_ai_planner.core.graph import (
        SessionNode,
        PlanNode,
        PlanStep,
        ToolCall,
        ParentChildEdge,
        PlanLinkEdge,
    )

    store = InMemoryGraphStore()
    session_node = SessionNode(id="s1", name="Test Session")
    plan_node = PlanNode(id="p1", title="Test Plan")
    step_node = PlanStep(id="step1", description="Test Step", index="1")
    tool_node = ToolCall(id="tool1", name="test_tool", args={"arg": "value"})

    await store.add_node(session_node)
    await store.add_node(plan_node)
    await store.add_node(step_node)
    await store.add_node(tool_node)

    # Connect session -> plan
    await store.add_edge(ParentChildEdge(src="s1", dst="p1"))
    # Connect plan -> step
    await store.add_edge(ParentChildEdge(src="p1", dst="step1"))
    # Connect step -> tool
    await store.add_edge(PlanLinkEdge(src="step1", dst="tool1"))

    await print_graph_structure(store)
    captured = capsys.readouterr().out

    assert "Step 1: Test Step" in captured
    assert "test_tool" in captured


@pytest.mark.asyncio
async def test_print_graph_structure_no_session_nodes(capsys):
    """Test printing graph structure when no session nodes exist"""
    from chuk_ai_planner.core.graph import PlanNode

    store = InMemoryGraphStore()
    plan_node = PlanNode(id="p1", title="Orphan Plan")
    await store.add_node(plan_node)

    await print_graph_structure(store)
    captured = capsys.readouterr().out

    assert "Total nodes: 1" in captured
    assert "plan: 1" in captured
    # Should not print session hierarchy since there are no sessions


@pytest.mark.asyncio
async def test_print_plan_structure_directly(capsys):
    """Test _print_plan_structure function directly"""
    from chuk_ai_planner.core.graph import (
        PlanNode,
        PlanStep,
        ToolCall,
        ParentChildEdge,
        PlanLinkEdge,
    )

    store = InMemoryGraphStore()
    plan_node = PlanNode(id="p1", title="Direct Plan Test")
    step_node = PlanStep(id="step1", description="Direct Step", index="1")
    tool_node = ToolCall(id="tool1", name="direct_tool", args={})

    await store.add_node(plan_node)
    await store.add_node(step_node)
    await store.add_node(tool_node)

    await store.add_edge(ParentChildEdge(src="p1", dst="step1"))
    await store.add_edge(PlanLinkEdge(src="step1", dst="tool1"))

    # Get nodes and edges
    nodes = list(store.nodes.values())
    edges = store.edges

    _print_plan_structure(store, plan_node, nodes, edges, "  ")
    captured = capsys.readouterr().out

    assert "Step 1: Direct Step" in captured
    assert "direct_tool" in captured


@pytest.mark.asyncio
async def test_print_assistant_structure_directly(capsys):
    """Test _print_assistant_structure function directly"""
    from chuk_ai_planner.core.graph import ToolCall, TaskRun, TaskStatus

    store = InMemoryGraphStore()

    # Create a mock assistant node (using a generic GraphNode since AssistantMessage is in extensions)
    from chuk_ai_planner.core.graph.nodes.base import GraphNode

    assistant_node = GraphNode(id="asst1", kind="assistant_message", data={})
    tool_node = ToolCall(id="tool1", name="assistant_tool", args={})
    task_node = TaskRun(
        id="task1",
        tool_call_id="tool1",
        status=TaskStatus.SUCCESS,
        result={"output": "success"},
    )

    await store.add_node(assistant_node)
    await store.add_node(tool_node)
    await store.add_node(task_node)

    # Connect assistant -> tool using CustomEdge
    from chuk_ai_planner.core.graph import CustomEdge

    await store.add_edge(
        CustomEdge(src="asst1", dst="tool1", custom_type="assistant_tool_link")
    )
    # Connect tool -> task using CustomEdge
    await store.add_edge(
        CustomEdge(src="tool1", dst="task1", custom_type="tool_task_link")
    )

    # Get nodes and edges
    nodes = list(store.nodes.values())
    edges = store.edges

    _print_assistant_structure(store, assistant_node, nodes, edges, "  ")
    captured = capsys.readouterr().out

    assert "Tool: assistant_tool" in captured
    assert "Task: ✓" in captured
