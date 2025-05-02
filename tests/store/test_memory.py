# tests/store/test_memory.py
import pytest

from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.models import GraphNode, NodeKind
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind

@pytest.fixture

def store():
    return InMemoryGraphStore()

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

# Node operations

def test_add_and_get_node(store, node_factory):
    node = node_factory("n1", NodeKind.SESSION, foo="bar")
    assert store.get_node("n1") is None
    store.add_node(node)
    assert store.get_node("n1") is node
    assert store.get_node("n1").data["foo"] == "bar"


def test_update_existing_node(store, node_factory):
    node = node_factory("n2", NodeKind.PLAN_STEP, value=1)
    store.add_node(node)
    updated = node.model_copy()
    upd_data = dict(updated.data)
    upd_data["value"] = 99
    updated = updated.model_copy(update={"data": upd_data})
    store.update_node(updated)
    result = store.get_node("n2")
    assert result.data["value"] == 99


def test_update_nonexistent_node_noop(store, node_factory):
    node = node_factory("n3", NodeKind.USER_MSG)
    # Should not raise
    store.update_node(node)
    assert store.get_node("n3") is None

# Edge operations

def test_add_and_get_edges(store, edge_factory):
    e1 = edge_factory("a", "b", EdgeKind.PARENT_CHILD)
    e2 = edge_factory("b", "c", EdgeKind.NEXT)
    e3 = edge_factory("a", "c", EdgeKind.PARENT_CHILD)
    store.add_edge(e1)
    store.add_edge(e2)
    store.add_edge(e3)
    all_edges = store.get_edges()
    assert set(all_edges) == {e1, e2, e3}

    assert set(store.get_edges(src="a")) == {e1, e3}
    assert set(store.get_edges(dst="c")) == {e2, e3}
    assert set(store.get_edges(kind=EdgeKind.PARENT_CHILD)) == {e1, e3}
    assert store.get_edges(src="a", dst="c", kind=EdgeKind.PARENT_CHILD) == [e3]

# Nodes by kind

def test_get_nodes_by_kind(store, node_factory):
    n1 = node_factory("x", NodeKind.SESSION)
    n2 = node_factory("y", NodeKind.PLAN_STEP)
    store.add_node(n1)
    store.add_node(n2)
    assert store.get_nodes_by_kind(NodeKind.SESSION) == [n1]
    assert store.get_nodes_by_kind(NodeKind.PLAN_STEP) == [n2]
    assert store.get_nodes_by_kind(NodeKind.USER_MSG) == []

# Clear store

def test_clear(store, node_factory, edge_factory):
    n = node_factory("n", NodeKind.SESSION)
    store.add_node(n)
    e = edge_factory("n", "n", EdgeKind.CUSTOM)
    store.add_edge(e)
    assert store.get_node("n") is not None
    assert store.get_edges() != []
    store.clear()
    assert store.get_node("n") is None
    assert store.get_edges() == []
