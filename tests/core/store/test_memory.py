# tests/store/test_memory.py
import pytest

from chuk_ai_planner.core.store.memory import InMemoryGraphStore
from chuk_ai_planner.core.graph import (
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


@pytest.mark.asyncio
async def test_add_and_get_node(store, node_factory):
    node = node_factory("n1", NodeType.SESSION, name="Test Session")
    assert await store.get_node("n1") is None
    await store.add_node(node)
    retrieved = await store.get_node("n1")
    assert retrieved is node
    assert retrieved.name == "Test Session"


@pytest.mark.asyncio
async def test_update_existing_node(store, node_factory):
    node = node_factory("n2", NodeType.PLAN_STEP, description="Original")
    await store.add_node(node)
    # Use metadata dict for custom data (no .data dict anymore!)
    updated = node.model_copy(update={"metadata": {"value": 99}})
    await store.update_node(updated)
    result = await store.get_node("n2")
    assert result.metadata["value"] == 99


@pytest.mark.asyncio
async def test_update_nonexistent_node_noop(store, node_factory):
    node = node_factory("n3", NodeType.SESSION, name="Test")
    # Should not raise
    await store.update_node(node)
    assert await store.get_node("n3") is None


# Edge operations


@pytest.mark.asyncio
async def test_add_and_get_edges(store, edge_factory):
    e1 = edge_factory("a", "b", EdgeType.PARENT_CHILD)
    e2 = edge_factory("b", "c", EdgeType.NEXT)
    e3 = edge_factory("a", "c", EdgeType.PARENT_CHILD)
    await store.add_edge(e1)
    await store.add_edge(e2)
    await store.add_edge(e3)
    all_edges = await store.get_edges()
    assert set(all_edges) == {e1, e2, e3}

    assert set(await store.get_edges(src="a")) == {e1, e3}
    assert set(await store.get_edges(dst="c")) == {e2, e3}
    assert set(await store.get_edges(kind=EdgeType.PARENT_CHILD)) == {e1, e3}
    assert await store.get_edges(src="a", dst="c", kind=EdgeType.PARENT_CHILD) == [e3]


# Nodes by kind


@pytest.mark.asyncio
async def test_get_nodes_by_kind(store, node_factory):
    n1 = node_factory("x", NodeType.SESSION)
    n2 = node_factory("y", NodeType.PLAN_STEP)
    await store.add_node(n1)
    await store.add_node(n2)
    assert await store.get_nodes_by_kind(NodeType.SESSION) == [n1]
    assert await store.get_nodes_by_kind(NodeType.PLAN_STEP) == [n2]
    assert await store.get_nodes_by_kind(NodeType.PLAN) == []  # Changed from USER_MSG


# List nodes


@pytest.mark.asyncio
async def test_list_nodes_all(store, node_factory):
    """Test listing all nodes."""
    n1 = node_factory("a", NodeType.SESSION)
    n2 = node_factory("b", NodeType.PLAN)
    await store.add_node(n1)
    await store.add_node(n2)

    all_nodes = await store.list_nodes()
    assert len(all_nodes) == 2
    assert set(all_nodes) == {n1, n2}


@pytest.mark.asyncio
async def test_list_nodes_by_kind(store, node_factory):
    """Test listing nodes filtered by kind."""
    n1 = node_factory("a", NodeType.SESSION)
    n2 = node_factory("b", NodeType.PLAN)
    n3 = node_factory("c", NodeType.SESSION)
    await store.add_node(n1)
    await store.add_node(n2)
    await store.add_node(n3)

    session_nodes = await store.list_nodes(kind=NodeType.SESSION.value)
    assert len(session_nodes) == 2
    assert set(session_nodes) == {n1, n3}

    plan_nodes = await store.list_nodes(kind=NodeType.PLAN.value)
    assert len(plan_nodes) == 1
    assert plan_nodes[0] == n2


# Clear store


@pytest.mark.asyncio
async def test_clear(store, node_factory, edge_factory):
    n = node_factory("n", NodeType.SESSION)
    await store.add_node(n)
    e = edge_factory("n", "n", EdgeType.CUSTOM)
    await store.add_edge(e)
    assert await store.get_node("n") is not None
    assert await store.get_edges() != []
    await store.clear()
    assert await store.get_node("n") is None
    assert await store.get_edges() == []
