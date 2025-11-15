# chuk_ai_planner/graph/nodes/execution.py
"""
Execution node types.

Nodes related to tool invocation and execution results.
Pure Pydantic with typed fields.
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import Field

from chuk_ai_planner.core.graph.nodes.base import GraphNode
from chuk_ai_planner.core.graph.types import NodeType, TaskStatus

__all__ = ["ToolCall", "TaskRun"]


class ToolCall(GraphNode):
    """
    A tool invocation request.

    Represents a call to an MCP tool or external function.
    Contains the tool name and arguments needed for execution.
    """

    kind: Literal[NodeType.TOOL_CALL] = NodeType.TOOL_CALL

    # Tool identification
    name: str

    # Arguments to pass to the tool
    args: dict[str, Any] = Field(default_factory=dict)

    # Where to store the result
    result_variable: Optional[str] = None


class TaskRun(GraphNode):
    """
    Result of a tool execution.

    Captures the outcome of executing a ToolCall, including
    success/failure status, result data, and timing information.
    """

    kind: Literal[NodeType.TASK_RUN] = NodeType.TASK_RUN

    # Reference to the tool call that was executed
    tool_call_id: str

    # Execution status
    status: TaskStatus

    # Result data (on success)
    result: Optional[Any] = None

    # Error message (on failure)
    error: Optional[str] = None

    # Execution timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Retry tracking
    attempt_number: int = 1  # Which attempt this is (1 = first try)
    max_attempts: int = 1  # Max attempts configured
    retry_delay_seconds: Optional[float] = None  # Delay before retry

    # Cost & performance tracking
    cost: Optional[float] = None  # Actual cost incurred
    tokens_used: Optional[int] = None  # For LLM calls
    model_used: Optional[str] = None  # Which model was used

    @property
    def duration_seconds(self) -> Optional[float]:
        """
        Calculate execution duration in seconds.

        Returns
        -------
        Optional[float]
            Duration in seconds, or None if timing info incomplete
        """
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds()
        return None
