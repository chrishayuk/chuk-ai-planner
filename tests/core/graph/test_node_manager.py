"""
Tests for GraphNodeManager.

Tests for creating and managing nodes in the graph store.
"""

import pytest

from chuk_ai_planner.core.graph import (
    ToolCall,
    TaskRun,
    SummaryNode,
    EdgeType,
)
from chuk_ai_planner.core.graph.node_manager import GraphNodeManager
from chuk_ai_planner.core.store.memory import InMemoryGraphStore


class TestGraphNodeManager:
    """Test GraphNodeManager initialization."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Should initialize with a graph store."""
        graph = InMemoryGraphStore()
        manager = GraphNodeManager(graph)

        assert manager.graph_store is graph


class TestCreateToolCallNode:
    """Test creating tool call nodes."""

    @pytest.mark.asyncio
    async def test_create_tool_call_basic(self):
        """Should create a tool call node with required fields."""
        graph = InMemoryGraphStore()
        manager = GraphNodeManager(graph)

        # Create a parent node
        from chuk_ai_planner.core.graph import SessionNode

        parent = SessionNode(name="Test Session")
        await graph.add_node(parent)

        # Create tool call
        tool = await manager.create_tool_call_node(
            tool_name="get_weather",
            args={"city": "New York"},
            result=None,  # Ignored
            assistant_node_id=parent.id,
        )

        # Verify tool node created
        assert isinstance(tool, ToolCall)
        assert tool.name == "get_weather"
        assert tool.args == {"city": "New York"}

        # Verify node added to graph
        retrieved = await graph.get_node(tool.id)
        assert retrieved == tool

        # Verify edge created
        edges = await graph.get_edges(src=parent.id)
        assert len(edges) == 1
        assert edges[0].kind == EdgeType.PARENT_CHILD
        assert edges[0].dst == tool.id

    @pytest.mark.asyncio
    async def test_create_tool_call_with_ignored_params(self):
        """Should ignore result, error, and is_cached params."""
        graph = InMemoryGraphStore()
        manager = GraphNodeManager(graph)

        from chuk_ai_planner.core.graph import SessionNode

        parent = SessionNode(name="Test")
        await graph.add_node(parent)

        # These params are ignored
        tool = await manager.create_tool_call_node(
            tool_name="my_tool",
            args={},
            result="some result",  # Ignored
            assistant_node_id=parent.id,
            error="some error",  # Ignored
            is_cached=True,  # Ignored
        )

        # Tool node should not have result/error fields
        assert tool.name == "my_tool"
        assert not hasattr(tool, "result")
        assert not hasattr(tool, "error")


class TestCreateTaskRunNode:
    """Test creating task run nodes."""

    @pytest.mark.asyncio
    async def test_create_task_run_success(self):
        """Should create a successful task run."""
        graph = InMemoryGraphStore()
        manager = GraphNodeManager(graph)

        # Create a tool call node first
        from chuk_ai_planner.core.graph import SessionNode

        parent = SessionNode(name="Test")
        await graph.add_node(parent)

        tool = await manager.create_tool_call_node(
            tool_name="test_tool", args={}, result=None, assistant_node_id=parent.id
        )

        # Create successful task run
        task = await manager.create_task_run_node(
            tool_node_id=tool.id, success=True, result={"data": "value"}
        )

        # Verify task node
        assert isinstance(task, TaskRun)
        assert task.tool_call_id == tool.id
        assert task.status == "success"
        assert task.result == {"data": "value"}
        assert task.error is None
        assert task.started_at is not None
        assert task.completed_at is not None

        # Verify node added to graph
        retrieved = await graph.get_node(task.id)
        assert retrieved == task

        # Verify edge created
        edges = await graph.get_edges(src=tool.id)
        assert len(edges) == 1
        assert edges[0].kind == EdgeType.PARENT_CHILD
        assert edges[0].dst == task.id

    @pytest.mark.asyncio
    async def test_create_task_run_failure(self):
        """Should create a failed task run."""
        graph = InMemoryGraphStore()
        manager = GraphNodeManager(graph)

        from chuk_ai_planner.core.graph import SessionNode

        parent = SessionNode(name="Test")
        await graph.add_node(parent)

        tool = await manager.create_tool_call_node(
            tool_name="test_tool", args={}, result=None, assistant_node_id=parent.id
        )

        # Create failed task run
        task = await manager.create_task_run_node(
            tool_node_id=tool.id, success=False, error="Tool execution failed"
        )

        # Verify task node
        assert task.status == "failure"
        assert task.error == "Tool execution failed"
        assert task.result is None

    @pytest.mark.asyncio
    async def test_create_task_run_with_result_and_error(self):
        """Should handle both result and error being set."""
        graph = InMemoryGraphStore()
        manager = GraphNodeManager(graph)

        from chuk_ai_planner.core.graph import SessionNode

        parent = SessionNode(name="Test")
        await graph.add_node(parent)

        tool = await manager.create_tool_call_node(
            tool_name="test_tool", args={}, result=None, assistant_node_id=parent.id
        )

        # Create with both
        task = await manager.create_task_run_node(
            tool_node_id=tool.id,
            success=False,  # Failed
            result={"partial": "data"},
            error="Partial failure",
        )

        assert task.status == "failure"
        assert task.result == {"partial": "data"}
        assert task.error == "Partial failure"


class TestCreateSummaryNode:
    """Test creating summary nodes."""

    @pytest.mark.asyncio
    async def test_create_summary_basic(self):
        """Should create a summary node with required fields."""
        graph = InMemoryGraphStore()
        manager = GraphNodeManager(graph)

        from chuk_ai_planner.core.graph import SessionNode

        parent = SessionNode(name="Test")
        await graph.add_node(parent)

        # Create summary
        summary = await manager.create_summary_node(
            content="This is a test summary", parent_node_id=parent.id
        )

        # Verify summary node
        assert isinstance(summary, SummaryNode)
        assert summary.content == "This is a test summary"
        assert summary.title == "This is a test summary"[:50]
        assert summary.summary_type == "checkpoint"

        # Verify node added to graph
        retrieved = await graph.get_node(summary.id)
        assert retrieved == summary

        # Verify edge created
        edges = await graph.get_edges(src=parent.id)
        assert len(edges) == 1
        assert edges[0].kind == EdgeType.PARENT_CHILD
        assert edges[0].dst == summary.id

    @pytest.mark.asyncio
    async def test_create_summary_with_title(self):
        """Should use provided title."""
        graph = InMemoryGraphStore()
        manager = GraphNodeManager(graph)

        from chuk_ai_planner.core.graph import SessionNode

        parent = SessionNode(name="Test")
        await graph.add_node(parent)

        summary = await manager.create_summary_node(
            content="Long summary content here",
            parent_node_id=parent.id,
            title="Custom Title",
        )

        assert summary.title == "Custom Title"
        assert summary.content == "Long summary content here"

    @pytest.mark.asyncio
    async def test_create_summary_types(self):
        """Should support different summary types."""
        graph = InMemoryGraphStore()
        manager = GraphNodeManager(graph)

        from chuk_ai_planner.core.graph import SessionNode

        parent = SessionNode(name="Test")
        await graph.add_node(parent)

        # Test different types
        for summary_type in ["checkpoint", "completion", "error", "milestone"]:
            summary = await manager.create_summary_node(
                content=f"Test {summary_type}",
                parent_node_id=parent.id,
                summary_type=summary_type,
            )
            assert summary.summary_type == summary_type

    @pytest.mark.asyncio
    async def test_create_summary_long_content_default_title(self):
        """Should truncate content for default title."""
        graph = InMemoryGraphStore()
        manager = GraphNodeManager(graph)

        from chuk_ai_planner.core.graph import SessionNode

        parent = SessionNode(name="Test")
        await graph.add_node(parent)

        long_content = "A" * 100  # 100 characters
        summary = await manager.create_summary_node(
            content=long_content, parent_node_id=parent.id
        )

        # Title should be first 50 chars
        assert summary.title == "A" * 50
        assert summary.content == long_content


class TestIntegration:
    """Test node manager integration scenarios."""

    @pytest.mark.asyncio
    async def test_tool_call_with_task_run(self):
        """Should create tool call and task run together."""
        graph = InMemoryGraphStore()
        manager = GraphNodeManager(graph)

        from chuk_ai_planner.core.graph import SessionNode

        parent = SessionNode(name="Test")
        await graph.add_node(parent)

        # Create tool call
        tool = await manager.create_tool_call_node(
            tool_name="get_data",
            args={"param": "value"},
            result=None,
            assistant_node_id=parent.id,
        )

        # Create task run for the tool
        task = await manager.create_task_run_node(
            tool_node_id=tool.id, success=True, result={"data": "retrieved"}
        )

        # Verify graph structure
        parent_edges = await graph.get_edges(src=parent.id)
        assert len(parent_edges) == 1
        assert parent_edges[0].dst == tool.id

        tool_edges = await graph.get_edges(src=tool.id)
        assert len(tool_edges) == 1
        assert tool_edges[0].dst == task.id

    @pytest.mark.asyncio
    async def test_multiple_summaries(self):
        """Should create multiple summaries for same parent."""
        graph = InMemoryGraphStore()
        manager = GraphNodeManager(graph)

        from chuk_ai_planner.core.graph import SessionNode

        parent = SessionNode(name="Test")
        await graph.add_node(parent)

        # Create multiple summaries
        summary1 = await manager.create_summary_node(
            content="First checkpoint",
            parent_node_id=parent.id,
            summary_type="checkpoint",
        )

        summary2 = await manager.create_summary_node(
            content="Second checkpoint",
            parent_node_id=parent.id,
            summary_type="checkpoint",
        )

        summary3 = await manager.create_summary_node(
            content="Completion", parent_node_id=parent.id, summary_type="completion"
        )

        # All should be children of parent
        edges = await graph.get_edges(src=parent.id)
        assert len(edges) == 3

        summary_ids = {e.dst for e in edges}
        assert summary1.id in summary_ids
        assert summary2.id in summary_ids
        assert summary3.id in summary_ids
