# tests/store/test_base.py
import pytest

from chuk_ai_planner.core.store.base import GraphStore
from chuk_ai_planner.core.graph import (
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
    SessionNode,
    PlanNode,
    PlanStep,
    ParentChildEdge,
    NextEdge,
)


class DummyGraphStore(GraphStore):
    """Async-native dummy graph store for testing."""

    def __init__(self):
        self._nodes = {}
        self._edges = []

    async def add_node(self, node: GraphNode) -> None:
        self._nodes[node.id] = node

    async def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    async def update_node(self, node: GraphNode) -> None:
        if node.id not in self._nodes:
            raise KeyError(f"Node {node.id} not found")
        self._nodes[node.id] = node

    async def add_edge(self, edge: GraphEdge) -> None:
        self._edges.append(edge)

    async def get_edges(
        self,
        src: str | None = None,
        dst: str | None = None,
        kind: EdgeType | None = None,
    ) -> list[GraphEdge]:
        results = self._edges
        if src is not None:
            results = [e for e in results if e.src == src]
        if dst is not None:
            results = [e for e in results if e.dst == dst]
        if kind is not None:
            results = [e for e in results if e.kind == kind]
        return results

    # Override get_nodes_by_kind to return nodes of matching kind
    async def get_nodes_by_kind(self, kind: NodeType) -> list[GraphNode]:
        return [n for n in self._nodes.values() if n.kind == kind]


@pytest.fixture
def store():
    return DummyGraphStore()


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

    def _create(src: str, dst: str, kind: EdgeType, **attrs):
        if kind == EdgeType.PARENT_CHILD:
            return ParentChildEdge(src=src, dst=dst)
        elif kind == EdgeType.NEXT:
            return NextEdge(src=src, dst=dst)
        else:
            return ParentChildEdge(src=src, dst=dst)

    return _create


# Node tests


@pytest.mark.asyncio
async def test_add_and_get_node(store, node_factory):
    node = node_factory("n1", NodeType.SESSION, name="Test Session")
    assert await store.get_node("n1") is None
    await store.add_node(node)
    retrieved = await store.get_node("n1")
    assert retrieved is node
    assert retrieved.name == "Test Session"


@pytest.mark.asyncio
async def test_update_node(store, node_factory):
    node = node_factory("n2", NodeType.PLAN_STEP, description="Original")
    await store.add_node(node)
    # Use metadata dict for custom data (no .data dict anymore!)
    updated = node.model_copy(update={"metadata": {"value": 42}})
    await store.update_node(updated)
    retrieved = await store.get_node("n2")
    assert retrieved.metadata["value"] == 42


@pytest.mark.asyncio
async def test_update_nonexistent_node_raises(store, node_factory):
    node = node_factory("n3", NodeType.SESSION, name="Test")
    with pytest.raises(KeyError):
        await store.update_node(node)


# Edge tests


@pytest.mark.asyncio
async def test_add_and_get_edges(store, edge_factory):
    e1 = edge_factory("n1", "n2", EdgeType.PARENT_CHILD)
    e2 = edge_factory("n2", "n3", EdgeType.NEXT)
    e3 = edge_factory("n1", "n3", EdgeType.PARENT_CHILD)
    await store.add_edge(e1)
    await store.add_edge(e2)
    await store.add_edge(e3)
    all_edges = await store.get_edges()
    assert set(all_edges) == {e1, e2, e3}

    src_edges = await store.get_edges(src="n1")
    assert set(src_edges) == {e1, e3}

    dst_edges = await store.get_edges(dst="n3")
    assert set(dst_edges) == {e2, e3}

    kind_edges = await store.get_edges(kind=EdgeType.PARENT_CHILD)
    assert set(kind_edges) == {e1, e3}

    combined = await store.get_edges(src="n1", dst="n3", kind=EdgeType.PARENT_CHILD)
    assert combined == [e3]


# get_nodes_by_kind tests


@pytest.mark.asyncio
async def test_get_nodes_by_kind_default_raises():
    # use a subclass that doesn't override get_nodes_by_kind
    class BaseDummy(GraphStore):
        async def add_node(self, node):
            pass

        async def get_node(self, node_id):
            return None

        async def update_node(self, node):
            pass

        async def add_edge(self, edge):
            pass

        async def get_edges(self, src=None, dst=None, kind=None):
            return []

    base = BaseDummy()
    with pytest.raises(NotImplementedError):
        await base.get_nodes_by_kind(NodeType.SESSION)


@pytest.mark.asyncio
async def test_get_nodes_by_kind_override(store, node_factory):
    n1 = node_factory("a", NodeType.SESSION)
    n2 = node_factory("b", NodeType.PLAN_STEP)
    await store.add_node(n1)
    await store.add_node(n2)
    sessions = await store.get_nodes_by_kind(NodeType.SESSION)
    assert sessions == [n1]
    steps = await store.get_nodes_by_kind(NodeType.PLAN_STEP)
    assert steps == [n2]


# get_edges_by_src test


@pytest.mark.asyncio
async def test_get_edges_by_src(store, edge_factory):
    """Test the get_edges_by_src convenience method."""
    e1 = edge_factory("n1", "n2", EdgeType.PARENT_CHILD)
    e2 = edge_factory("n1", "n3", EdgeType.NEXT)
    e3 = edge_factory("n2", "n3", EdgeType.PARENT_CHILD)
    await store.add_edge(e1)
    await store.add_edge(e2)
    await store.add_edge(e3)

    outgoing = await store.get_edges_by_src("n1")
    assert set(outgoing) == {e1, e2}

    outgoing_filtered = await store.get_edges_by_src("n1", kind=EdgeType.PARENT_CHILD)
    assert outgoing_filtered == [e1]
