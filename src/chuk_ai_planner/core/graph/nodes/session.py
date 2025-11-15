# chuk_ai_planner/graph/nodes/session.py
"""
Session and summary node types.

Top-level organizational nodes for tracking execution sessions.
Pure Pydantic with typed fields.
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import Field

from chuk_ai_planner.core.graph.nodes.base import GraphNode
from chuk_ai_planner.core.graph.types import NodeType, SummaryType

__all__ = ["SessionNode", "SummaryNode"]


class SessionNode(GraphNode):
    """
    An execution session.

    Represents a single execution context that contains plans,
    messages, and execution history. Used for tracking and persistence.
    """

    kind: Literal[NodeType.SESSION] = NodeType.SESSION

    # Session name/title
    name: str

    # Session description
    description: Optional[str] = None

    # User identifier (for multi-tenancy)
    user_id: Optional[str] = None

    # Session-level variables/context
    context: dict[str, Any] = Field(default_factory=dict)

    # Session status
    status: Literal["active", "completed", "failed", "cancelled"] = "active"

    # Session timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class SummaryNode(GraphNode):
    """
    A summary or checkpoint in the execution.

    Captures the state of execution at a point in time.
    Used for checkpointing, debugging, and audit trails.
    """

    kind: Literal[NodeType.SUMMARY] = NodeType.SUMMARY

    # Summary title
    title: str

    # Summary content
    content: str

    # What was summarized
    summary_type: SummaryType = SummaryType.CHECKPOINT

    # Execution state at this point
    execution_state: dict[str, Any] = Field(default_factory=dict)
