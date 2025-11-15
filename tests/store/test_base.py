# tests/store/test_base.py
import pytest

from chuk_ai_planner.store.base import GraphStore
from chuk_ai_planner.graph import (
    GraphNode, GraphEdge, NodeType, EdgeType,
    SessionNode, PlanNode, PlanStep,
    ParentChildEdge, NextEdge
)

class DummyGraphStore(GraphStore):
    def __init__(self):
        self._nodes = {}
        self._edges = []

    def add_node(self, node: GraphNode) -> None:
        self._nodes[node.id] = node

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def update_node(self, node: GraphNode) -> None:
        if node.id not in self._nodes:
            raise KeyError(f"Node {node.id} not found")
        self._nodes[node.id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        self._edges.append(edge)

    def get_edges(
        self, 
        src: str | None = None, 
        dst: str | None = None,
        kind: EdgeType | None = None
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
    def get_nodes_by_kind(self, kind: NodeType) -> list[GraphNode]:
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
            return SessionNode(id=node_id, name=attrs.get('name', 'Test Session'))
        elif kind == NodeType.PLAN:
            return PlanNode(id=node_id, title=attrs.get('title', 'Test Plan'))
        elif kind == NodeType.PLAN_STEP:
            return PlanStep(id=node_id, description=attrs.get('description', 'Test Step'))
        else:
            # For other types, create a SessionNode as fallback
            return SessionNode(id=node_id, name=attrs.get('name', 'Test'))
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

def test_add_and_get_node(store, node_factory):
    node = node_factory("n1", NodeType.SESSION, name="Test Session")
    assert store.get_node("n1") is None
    store.add_node(node)
    retrieved = store.get_node("n1")
    assert retrieved is node
    assert retrieved.name == "Test Session"


def test_update_node(store, node_factory):
    node = node_factory("n2", NodeType.PLAN_STEP, description="Original")
    store.add_node(node)
    # Use metadata dict for custom data (no .data dict anymore!)
    updated = node.model_copy(update={"metadata": {"value": 42}})
    store.update_node(updated)
    retrieved = store.get_node("n2")
    assert retrieved.metadata["value"] == 42


def test_update_nonexistent_node_raises(store, node_factory):
    node = node_factory("n3", NodeType.SESSION, name="Test")
    with pytest.raises(KeyError):
        store.update_node(node)

# Edge tests

def test_add_and_get_edges(store, edge_factory):
    e1 = edge_factory("n1", "n2", EdgeType.PARENT_CHILD)
    e2 = edge_factory("n2", "n3", EdgeType.NEXT)
    e3 = edge_factory("n1", "n3", EdgeType.PARENT_CHILD)
    store.add_edge(e1)
    store.add_edge(e2)
    store.add_edge(e3)
    all_edges = store.get_edges()
    assert set(all_edges) == {e1, e2, e3}

    src_edges = store.get_edges(src="n1")
    assert set(src_edges) == {e1, e3}

    dst_edges = store.get_edges(dst="n3")
    assert set(dst_edges) == {e2, e3}

    kind_edges = store.get_edges(kind=EdgeType.PARENT_CHILD)
    assert set(kind_edges) == {e1, e3}

    combined = store.get_edges(src="n1", dst="n3", kind=EdgeType.PARENT_CHILD)
    assert combined == [e3]

# get_nodes_by_kind tests

def test_get_nodes_by_kind_default_raises():
    # use a subclass that doesn't override get_nodes_by_kind
    class BaseDummy(GraphStore):
        def add_node(self, node): pass
        def get_node(self, node_id): return None
        def update_node(self, node): pass
        def add_edge(self, edge): pass
        def get_edges(self, src=None, dst=None, kind=None): return []

    base = BaseDummy()
    with pytest.raises(NotImplementedError):
        base.get_nodes_by_kind(NodeType.SESSION)


def test_get_nodes_by_kind_override(store, node_factory):
    n1 = node_factory("a", NodeType.SESSION)
    n2 = node_factory("b", NodeType.PLAN_STEP)
    store.add_node(n1)
    store.add_node(n2)
    sessions = store.get_nodes_by_kind(NodeType.SESSION)
    assert sessions == [n1]
    steps = store.get_nodes_by_kind(NodeType.PLAN_STEP)
    assert steps == [n2]
