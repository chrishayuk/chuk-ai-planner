# chuk_ai_planner/graph/nodes/workflow.py
"""
Workflow-specific graph nodes.

Nodes for human-in-the-loop, approvals, and workflow control.
These enable resilient, interactive planning workflows.
"""

from datetime import datetime
from typing import Literal, Optional

from chuk_ai_planner.core.graph.nodes.base import GraphNode
from chuk_ai_planner.core.graph.types import NodeType, ApprovalStatus

__all__ = ["ApprovalNode"]


class ApprovalNode(GraphNode):
    """
    Human-in-the-loop approval gate.

    Pauses execution until a human approves or rejects.
    Supports timeout, escalation, and approval tracking.

    Examples
    --------
    >>> approval = ApprovalNode(
    ...     prompt="Approve this blog post for publication?",
    ...     approval_type="human",
    ...     timeout_seconds=3600
    ... )
    >>> approval.status
    <ApprovalStatus.PENDING: 'pending'>
    """

    kind: Literal[NodeType.APPROVAL] = NodeType.APPROVAL

    # What needs approval
    approval_type: str  # "human", "system", "policy"
    prompt: str  # What to show the approver

    # State
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None

    # Behavior
    timeout_seconds: Optional[int] = None
    auto_approve_after: Optional[int] = None  # Auto-approve after N seconds
    escalate_to: Optional[str] = None  # User/role to escalate to on timeout
    escalation_timeout: Optional[int] = None  # Escalation timeout in seconds
