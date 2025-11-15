# tests/store/test_memory.py
import pytest

from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.graph import (
    GraphNode,
    NodeType,
    SessionNode,
    PlanNode,
    PlanStep,
    ParentChildEdge,
    NextEdge,
    CustomEdge,
    EdgeType,
)


@pytest.fixture
def store():
    return InMemoryGraphStore()


@pytest.fixture
def node_factory():
    """Create nodes using specific typed classes."""

    def _create(node_id: str, kind: NodeType, **attrs) -> GraphNode:
        # Map node types to their specific classes
        if kind == NodeType.SESSION:
            return SessionNode(id=node_id, name=attrs.get("name", "Test Session"))
        elif kind == NodeType.PLAN:
            return PlanNode(id=node_id, title=attrs.get("title", "Test Plan"))
        elif kind == NodeType.PLAN_STEP:
            return PlanStep(
                id=node_id, description=attrs.get("description", "Test Step")
            )
        else:
            # For other types, create a SessionNode as fallback
            return SessionNode(id=node_id, name=attrs.get("name", "Test"))

    return _create


@pytest.fixture
def edge_factory():
    """Create edges using specific typed classes."""

    def _create(src: str, dst: str, kind: EdgeType, **attrs) -> ParentChildEdge:
        if kind == EdgeType.PARENT_CHILD:
            return ParentChildEdge(src=src, dst=dst)
        elif kind == EdgeType.NEXT:
            return NextEdge(src=src, dst=dst)
        elif kind == EdgeType.CUSTOM:
            return CustomEdge(
                src=src, dst=dst, custom_type=attrs.get("custom_type", "test")
            )
        else:
            return ParentChildEdge(src=src, dst=dst)

    return _create


# Node operations


def test_add_and_get_node(store, node_factory):
    node = node_factory("n1", NodeType.SESSION, name="Test Session")
    assert store.get_node("n1") is None
    store.add_node(node)
    assert store.get_node("n1") is node
    assert store.get_node("n1").name == "Test Session"


def test_update_existing_node(store, node_factory):
    node = node_factory("n2", NodeType.PLAN_STEP, description="Original")
    store.add_node(node)
    # Use metadata dict for custom data (no .data dict anymore!)
    updated = node.model_copy(update={"metadata": {"value": 99}})
    store.update_node(updated)
    result = store.get_node("n2")
    assert result.metadata["value"] == 99


def test_update_nonexistent_node_noop(store, node_factory):
    node = node_factory("n3", NodeType.SESSION, name="Test")
    # Should not raise
    store.update_node(node)
    assert store.get_node("n3") is None


# Edge operations


def test_add_and_get_edges(store, edge_factory):
    e1 = edge_factory("a", "b", EdgeType.PARENT_CHILD)
    e2 = edge_factory("b", "c", EdgeType.NEXT)
    e3 = edge_factory("a", "c", EdgeType.PARENT_CHILD)
    store.add_edge(e1)
    store.add_edge(e2)
    store.add_edge(e3)
    all_edges = store.get_edges()
    assert set(all_edges) == {e1, e2, e3}

    assert set(store.get_edges(src="a")) == {e1, e3}
    assert set(store.get_edges(dst="c")) == {e2, e3}
    assert set(store.get_edges(kind=EdgeType.PARENT_CHILD)) == {e1, e3}
    assert store.get_edges(src="a", dst="c", kind=EdgeType.PARENT_CHILD) == [e3]


# Nodes by kind


def test_get_nodes_by_kind(store, node_factory):
    n1 = node_factory("x", NodeType.SESSION)
    n2 = node_factory("y", NodeType.PLAN_STEP)
    store.add_node(n1)
    store.add_node(n2)
    assert store.get_nodes_by_kind(NodeType.SESSION) == [n1]
    assert store.get_nodes_by_kind(NodeType.PLAN_STEP) == [n2]
    assert store.get_nodes_by_kind(NodeType.PLAN) == []  # Changed from USER_MSG


# Clear store


def test_clear(store, node_factory, edge_factory):
    n = node_factory("n", NodeType.SESSION)
    store.add_node(n)
    e = edge_factory("n", "n", EdgeType.CUSTOM)
    store.add_edge(e)
    assert store.get_node("n") is not None
    assert store.get_edges() != []
    store.clear()
    assert store.get_node("n") is None
    assert store.get_edges() == []
