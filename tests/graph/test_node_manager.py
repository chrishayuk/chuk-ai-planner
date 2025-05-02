# tests/graph/test_node_manager.py
import pytest
from datetime import datetime, timezone

from chuk_ai_planner.graph.node_manager import GraphNodeManager
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.models import AssistantMessage, ToolCall, TaskRun, Summary, NodeKind, GraphNode
from chuk_ai_planner.models.edges import EdgeKind, GraphEdge, ParentChildEdge

@pytest.fixture
def store():
    return InMemoryGraphStore()

@pytest.fixture
def manager(store):
    return GraphNodeManager(store)

@pytest.fixture
def assistant_node(store):
    # Create and add an AssistantMessage node
    node = AssistantMessage(data={"content": "orig", "tool_calls": []})
    store.add_node(node)
    return node

class DummyMessage:
    # For update_assistant_node, mimic incoming assistant message dict
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls
    def get(self, key, default=None):
        return getattr(self, key, default)


def test_update_assistant_node_missing(store, manager):
    # No such node
    result = manager.update_assistant_node("no-id", {"content": "new"})
    assert result is None


def test_update_assistant_node_wrong_kind(store, manager):
    # Add a non-assistant node
    wrong = Summary(data={"content": "sum"})
    store.add_node(wrong)
    result = manager.update_assistant_node(wrong.id, {"content": "new"})
    assert result is None


def test_update_assistant_node_success(store, manager, assistant_node):
    # Update content and tool_calls
    new_data = {"content": "updated text", "tool_calls": [1, 2, 3]}
    updated = manager.update_assistant_node(assistant_node.id, new_data)
    assert isinstance(updated, AssistantMessage)
    # Verify fields
    assert updated.id == assistant_node.id
    assert updated.data["content"] == new_data["content"]
    assert updated.data["tool_calls"] == new_data["tool_calls"]
    assert "updated_at" in updated.data
    # Check store has updated node
    stored = store.get_node(assistant_node.id)
    assert stored is updated


def test_create_tool_call_node(store, manager, assistant_node):
    tool_name = "weather"
    args = {"loc": "X"}
    result = {"value": 10}
    tool_node = manager.create_tool_call_node(tool_name, args, result, assistant_node.id, error=None, is_cached=True)
    # Validate node
    assert isinstance(tool_node, ToolCall)
    assert tool_node.data["name"] == tool_name
    assert tool_node.data["args"] == args
    assert tool_node.data["result"] == result
    assert tool_node.data["error"] is None
    assert tool_node.data["cached"] is True
    assert "timestamp" in tool_node.data
    # Node stored
    assert store.get_node(tool_node.id) is tool_node
    # Edge exists
    edges = store.get_edges(src=assistant_node.id)
    assert len(edges) == 1
    edge = edges[0]
    assert isinstance(edge, ParentChildEdge)
    assert edge.dst == tool_node.id


def test_create_task_run_node(store, manager):
    # Create tool call node first
    tool_node = ToolCall(data={})
    store.add_node(tool_node)
    # Now create task run
    task_node = manager.create_task_run_node(tool_node.id, success=False, error="fail")
    assert isinstance(task_node, TaskRun)
    assert task_node.data["success"] is False
    assert task_node.data["error"] == "fail"
    assert "timestamp" in task_node.data
    # Node stored
    assert store.get_node(task_node.id) is task_node
    # Edge exists
    edges = store.get_edges(src=tool_node.id)
    assert len(edges) == 1
    edge = edges[0]
    assert isinstance(edge, ParentChildEdge)
    assert edge.dst == task_node.id


def test_create_summary_node(store, manager):
    # Create parent node
    parent = ToolCall(data={})
    store.add_node(parent)
    content = "a summary"
    summary_node = manager.create_summary_node(content, parent.id)
    assert isinstance(summary_node, Summary)
    assert summary_node.data["content"] == content
    assert "timestamp" in summary_node.data
    # Node stored
    assert store.get_node(summary_node.id) is summary_node
    # Edge exists
    edges = store.get_edges(src=parent.id)
    assert len(edges) == 1
    edge = edges[0]
    assert isinstance(edge, ParentChildEdge)
    assert edge.dst == summary_node.id
