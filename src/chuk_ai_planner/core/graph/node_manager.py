# chuk_ai_planner/graph/node_manager.py
"""
Graph node management component.

This module handles creating and updating graph nodes, including
tool calls, tasks, summaries, etc.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from chuk_ai_planner.core.graph import ToolCall, TaskRun, SummaryNode
from chuk_ai_planner.core.graph import ParentChildEdge
from chuk_ai_planner.core.graph.types import TaskStatus, SummaryType

from ..store.base import GraphStore

_log = logging.getLogger(__name__)


class GraphNodeManager:
    """
    Handles managing nodes in the graph.

    This class provides methods for creating and updating various types
    of nodes in the graph, as well as creating edges between them.
    """

    def __init__(self, graph_store: GraphStore):
        """
        Initialize the graph node manager.

        Parameters
        ----------
        graph_store : GraphStore
            The graph store to use for storing nodes and edges
        """
        self.graph_store = graph_store

    async def create_tool_call_node(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,  # Ignored - stored in TaskRun instead
        assistant_node_id: str,
        error: Optional[str] = None,  # Ignored - stored in TaskRun instead
        is_cached: bool = False,  # Ignored - can be stored in TaskRun metadata
    ) -> ToolCall:
        """
        Create a tool call node and connect it to the parent node.

        Note: result, error, and is_cached parameters are ignored.
        They should be stored in the TaskRun node instead.

        Parameters
        ----------
        tool_name : str
            Name of the tool
        args : Dict[str, Any]
            Arguments passed to the tool
        result : Any
            (Ignored) Result is stored in TaskRun node
        assistant_node_id : str
            ID of the parent node (usually assistant or session)
        error : Optional[str]
            (Ignored) Error is stored in TaskRun node
        is_cached : bool
            (Ignored) Cache info can be stored in TaskRun metadata

        Returns
        -------
        ToolCall
            The created tool call node
        """
        # Create tool call node with typed fields (no data dict!)
        tool_node = ToolCall(name=tool_name, args=args)
        await self.graph_store.add_node(tool_node)

        # Create edge from parent to tool call
        edge = ParentChildEdge(src=assistant_node_id, dst=tool_node.id)
        await self.graph_store.add_edge(edge)

        return tool_node

    async def create_task_run_node(
        self,
        tool_node_id: str,
        success: bool,
        error: Optional[str] = None,
        result: Optional[Any] = None,
    ) -> TaskRun:
        """
        Create a task run node and connect it to the tool call node.

        Parameters
        ----------
        tool_node_id : str
            ID of the tool call node
        success : bool
            Whether the task was successful
        error : Optional[str]
            Error message, if any
        result : Optional[Any]
            The result data from the tool execution

        Returns
        -------
        TaskRun
            The created task run node
        """
        # Convert boolean success to status enum
        status = TaskStatus.SUCCESS if success else TaskStatus.FAILURE

        # Create task run node with typed fields (no data dict!)
        now = datetime.now(timezone.utc)
        task_node = TaskRun(
            tool_call_id=tool_node_id,
            status=status,
            result=result,
            error=error,
            started_at=now,  # We don't have separate start time, use now
            completed_at=now,
        )
        await self.graph_store.add_node(task_node)

        # Create edge from tool call to task run
        edge = ParentChildEdge(src=tool_node_id, dst=task_node.id)
        await self.graph_store.add_edge(edge)

        return task_node

    async def create_summary_node(
        self,
        content: str,
        parent_node_id: str,
        title: Optional[str] = None,
        summary_type: str = "checkpoint",
    ) -> SummaryNode:
        """
        Create a summary node and connect it to the parent node.

        Parameters
        ----------
        content : str
            Summary content
        parent_node_id : str
            ID of the parent node
        title : Optional[str]
            Summary title (defaults to truncated content)
        summary_type : str
            Type of summary: checkpoint, completion, error, or milestone

        Returns
        -------
        SummaryNode
            The created summary node
        """
        # Convert string to SummaryType enum
        summary_enum = SummaryType(summary_type)

        # Create summary node with typed fields (no data dict!)
        summary_node = SummaryNode(
            title=title or content[:50],  # Use first 50 chars if no title
            content=content,
            summary_type=summary_enum,
        )
        await self.graph_store.add_node(summary_node)

        # Create edge from parent to summary
        edge = ParentChildEdge(src=parent_node_id, dst=summary_node.id)
        await self.graph_store.add_edge(edge)

        return summary_node
