# tests/store/test_base.py
import pytest

from chuk_ai_planner.store.base import GraphStore
from chuk_ai_planner.models import GraphNode, NodeKind
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind

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
        kind: EdgeKind | None = None
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
    def get_nodes_by_kind(self, kind: NodeKind) -> list[GraphNode]:
        return [n for n in self._nodes.values() if n.kind == kind]


@pytest.fixture

def store():
    return DummyGraphStore()


@pytest.fixture

def node_factory():
    def _create(node_id: str, kind: NodeKind, **attrs) -> GraphNode:
        return GraphNode(id=node_id, kind=kind, data=attrs)
    return _create


@pytest.fixture

def edge_factory():
    def _create(src: str, dst: str, kind: EdgeKind, **attrs) -> GraphEdge:
        return GraphEdge(src=src, dst=dst, kind=kind, data=attrs)
    return _create


# Node tests

def test_add_and_get_node(store, node_factory):
    node = node_factory("n1", NodeKind.SESSION, foo="bar")
    assert store.get_node("n1") is None
    store.add_node(node)
    retrieved = store.get_node("n1")
    assert retrieved is node
    assert retrieved.data["foo"] == "bar"


def test_update_node(store, node_factory):
    node = node_factory("n2", NodeKind.PLAN_STEP, value=1)
    store.add_node(node)
    # Use model_copy() instead of deprecated copy()
    updated = node.model_copy()
    updated_data = dict(updated.data)
    updated_data["value"] = 42
    updated = updated.model_copy(update={"data": updated_data})
    store.update_node(updated)
    retrieved = store.get_node("n2")
    assert retrieved.data["value"] == 42


def test_update_nonexistent_node_raises(store, node_factory):
    node = node_factory("n3", NodeKind.USER_MSG)
    with pytest.raises(KeyError):
        store.update_node(node)

# Edge tests

def test_add_and_get_edges(store, edge_factory):
    e1 = edge_factory("n1", "n2", EdgeKind.PARENT_CHILD)
    e2 = edge_factory("n2", "n3", EdgeKind.NEXT)
    e3 = edge_factory("n1", "n3", EdgeKind.PARENT_CHILD)
    store.add_edge(e1)
    store.add_edge(e2)
    store.add_edge(e3)
    all_edges = store.get_edges()
    assert set(all_edges) == {e1, e2, e3}

    src_edges = store.get_edges(src="n1")
    assert set(src_edges) == {e1, e3}

    dst_edges = store.get_edges(dst="n3")
    assert set(dst_edges) == {e2, e3}

    kind_edges = store.get_edges(kind=EdgeKind.PARENT_CHILD)
    assert set(kind_edges) == {e1, e3}

    combined = store.get_edges(src="n1", dst="n3", kind=EdgeKind.PARENT_CHILD)
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
        base.get_nodes_by_kind(NodeKind.SESSION)


def test_get_nodes_by_kind_override(store, node_factory):
    n1 = node_factory("a", NodeKind.SESSION)
    n2 = node_factory("b", NodeKind.PLAN_STEP)
    store.add_node(n1)
    store.add_node(n2)
    sessions = store.get_nodes_by_kind(NodeKind.SESSION)
    assert sessions == [n1]
    steps = store.get_nodes_by_kind(NodeKind.PLAN_STEP)
    assert steps == [n2]
