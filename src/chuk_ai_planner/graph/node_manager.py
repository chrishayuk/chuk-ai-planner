# chuk_ai_planner/graph/node_manager.py
"""
Graph node management component.

This module handles creating and updating graph nodes, including
tool calls, tasks, assistant messages, etc.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from chuk_ai_planner.models import (
    NodeKind,
    AssistantMessage,
    ToolCall,
    TaskRun,
    Summary
)
from chuk_ai_planner.models.edges import ParentChildEdge

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
    
    def update_assistant_node(
        self,
        node_id: str,
        assistant_msg: Dict[str, Any]
    ) -> Optional[AssistantMessage]:
        """
        Update an assistant message node with new content.
        
        Parameters
        ----------
        node_id : str
            ID of the node to update
        assistant_msg : Dict[str, Any]
            New content for the node
            
        Returns
        -------
        Optional[AssistantMessage]
            The updated node, or None if node not found or not an assistant message
        """
        # Get the existing node
        node = self.graph_store.get_node(node_id)
        if not node or node.kind != NodeKind.ASSIST_MSG:
            _log.warning(f"Node {node_id} not found or not an assistant message")
            return None
        
        # Create updated node
        updated_node = AssistantMessage(
            id=node_id,
            data={
                **node.data,
                "content": assistant_msg.get("content"),
                "tool_calls": assistant_msg.get("tool_calls", []),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Update in store
        self.graph_store.update_node(updated_node)
        
        return updated_node
    
    def create_tool_call_node(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        assistant_node_id: str,
        error: Optional[str] = None,
        is_cached: bool = False
    ) -> ToolCall:
        """
        Create a tool call node and connect it to the assistant node.
        
        Parameters
        ----------
        tool_name : str
            Name of the tool
        args : Dict[str, Any]
            Arguments passed to the tool
        result : Any
            Result returned by the tool
        assistant_node_id : str
            ID of the assistant node that initiated the tool call
        error : Optional[str]
            Error message, if any
        is_cached : bool
            Whether the result was retrieved from cache
            
        Returns
        -------
        ToolCall
            The created tool call node
        """
        # Create tool call node
        tool_node = ToolCall(
            data={
                "name": tool_name,
                "args": args,
                "result": result,
                "error": error,
                "cached": is_cached,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        self.graph_store.add_node(tool_node)
        
        # Create edge from assistant to tool call
        edge = ParentChildEdge(
            src=assistant_node_id,
            dst=tool_node.id
        )
        self.graph_store.add_edge(edge)
        
        return tool_node
    
    def create_task_run_node(
        self,
        tool_node_id: str,
        success: bool,
        error: Optional[str] = None
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
            
        Returns
        -------
        TaskRun
            The created task run node
        """
        # Create task run node
        task_node = TaskRun(
            data={
                "success": success,
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        self.graph_store.add_node(task_node)
        
        # Create edge from tool call to task run
        edge = ParentChildEdge(
            src=tool_node_id,
            dst=task_node.id
        )
        self.graph_store.add_edge(edge)
        
        return task_node
    
    def create_summary_node(
        self,
        content: str,
        parent_node_id: str
    ) -> Summary:
        """
        Create a summary node and connect it to the parent node.
        
        Parameters
        ----------
        content : str
            Summary content
        parent_node_id : str
            ID of the parent node
            
        Returns
        -------
        Summary
            The created summary node
        """
        # Create summary node
        summary_node = Summary(
            data={
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        self.graph_store.add_node(summary_node)
        
        # Create edge from parent to summary
        edge = ParentChildEdge(
            src=parent_node_id,
            dst=summary_node.id
        )
        self.graph_store.add_edge(edge)
        
        return summary_node