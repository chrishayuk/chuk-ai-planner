# tests/utils/test_visualization.py
import pytest
from chuk_ai_planner.utils.visualization import print_session_events, print_graph_structure

from a2a_session_manager.models.session import Session, SessionEvent
from a2a_session_manager.models.event_type import EventType
from a2a_session_manager.models.event_source import EventSource

from chuk_ai_planner.models import NodeKind
from chuk_ai_planner.models.edges import EdgeKind, GraphEdge
from chuk_ai_planner.models.base import GraphNode
from chuk_ai_planner.store.memory import InMemoryGraphStore


def test_print_session_events_nested_and_types(capsys):
    # Create a session with a root MESSAGE event and a child TOOL_CALL event
    session = Session()
    root = SessionEvent(
        message={"content": "User says hello"},
        type=EventType.MESSAGE,
        source=EventSource.USER
    )
    session.events.append(root)
    child = SessionEvent(
        message={"tool": "weather", "error": None},
        type=EventType.TOOL_CALL,
        source=EventSource.LLM,
        metadata={"parent_event_id": root.id}
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


def test_print_graph_structure_basic(capsys):
    # Setup in-memory graph store with a session and a plan
    store = InMemoryGraphStore()
    session_node = GraphNode(id="s1", kind=NodeKind.SESSION, data={})
    plan_node = GraphNode(id="p1", kind=NodeKind.PLAN, data={})
    store.add_node(session_node)
    store.add_node(plan_node)
    # Connect session -> plan
    edge = GraphEdge(src="s1", dst="p1", kind=EdgeKind.PARENT_CHILD)
    store.add_edge(edge)

    print_graph_structure(store)
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
