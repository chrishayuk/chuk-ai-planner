"""
Tests for workflow-specific nodes.

Tests for ApprovalNode and other workflow control nodes.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from chuk_ai_planner.graph.nodes.workflow import ApprovalNode
from chuk_ai_planner.graph.types import NodeType, ApprovalStatus


class TestApprovalNode:
    """Test ApprovalNode for human-in-the-loop workflows."""

    def test_create_basic_approval(self):
        """Should create approval node with required fields."""
        approval = ApprovalNode(
            approval_type="human",
            prompt="Approve this content?"
        )

        assert approval.kind == NodeType.APPROVAL
        assert approval.approval_type == "human"
        assert approval.prompt == "Approve this content?"
        assert approval.status == ApprovalStatus.PENDING
        assert approval.approved_by is None
        assert approval.approved_at is None
        assert approval.rejected_at is None
        assert approval.rejection_reason is None

    def test_approval_with_timeout(self):
        """Should create approval with timeout settings."""
        approval = ApprovalNode(
            approval_type="human",
            prompt="Quick approval needed",
            timeout_seconds=300,
            auto_approve_after=600
        )

        assert approval.timeout_seconds == 300
        assert approval.auto_approve_after == 600

    def test_approval_with_escalation(self):
        """Should create approval with escalation settings."""
        approval = ApprovalNode(
            approval_type="human",
            prompt="Important decision",
            escalate_to="manager@company.com",
            escalation_timeout=1800
        )

        assert approval.escalate_to == "manager@company.com"
        assert approval.escalation_timeout == 1800

    def test_approval_status_defaults_to_pending(self):
        """Should default status to PENDING."""
        approval = ApprovalNode(
            approval_type="human",
            prompt="Test"
        )

        assert approval.status == ApprovalStatus.PENDING

    def test_approval_with_approval_tracking(self):
        """Should track who approved and when."""
        now = datetime.now(timezone.utc)
        approval = ApprovalNode(
            approval_type="human",
            prompt="Test",
            status=ApprovalStatus.APPROVED,
            approved_by="user123",
            approved_at=now
        )

        assert approval.status == ApprovalStatus.APPROVED
        assert approval.approved_by == "user123"
        assert approval.approved_at == now

    def test_approval_with_rejection_tracking(self):
        """Should track rejection reason and time."""
        now = datetime.now(timezone.utc)
        approval = ApprovalNode(
            approval_type="human",
            prompt="Test",
            status=ApprovalStatus.REJECTED,
            rejected_at=now,
            rejection_reason="Content doesn't meet standards"
        )

        assert approval.status == ApprovalStatus.REJECTED
        assert approval.rejected_at == now
        assert approval.rejection_reason == "Content doesn't meet standards"

    def test_approval_types(self):
        """Should support different approval types."""
        for approval_type in ["human", "system", "policy", "automated"]:
            approval = ApprovalNode(
                approval_type=approval_type,
                prompt="Test"
            )
            assert approval.approval_type == approval_type

    def test_approval_immutable(self):
        """Should be immutable (frozen)."""
        approval = ApprovalNode(
            approval_type="human",
            prompt="Test"
        )

        with pytest.raises(ValidationError):
            approval.status = ApprovalStatus.APPROVED

    def test_approval_has_graph_node_fields(self):
        """Should have all GraphNode base fields."""
        approval = ApprovalNode(
            approval_type="human",
            prompt="Test"
        )

        assert hasattr(approval, "id")
        assert hasattr(approval, "kind")
        assert hasattr(approval, "ts")
        assert hasattr(approval, "metadata")

    def test_approval_model_copy(self):
        """Should support model_copy for updates."""
        approval = ApprovalNode(
            approval_type="human",
            prompt="Test",
            status=ApprovalStatus.PENDING
        )

        now = datetime.now(timezone.utc)
        approved = approval.model_copy(update={
            "status": ApprovalStatus.APPROVED,
            "approved_by": "user456",
            "approved_at": now
        })

        assert approved.status == ApprovalStatus.APPROVED
        assert approved.approved_by == "user456"
        assert approved.approved_at == now
        assert approval.status == ApprovalStatus.PENDING  # Original unchanged

    def test_approval_repr(self):
        """Should have useful repr."""
        approval = ApprovalNode(
            approval_type="human",
            prompt="Test"
        )
        repr_str = repr(approval)

        assert "approval" in repr_str
        assert approval.id[:8] in repr_str

    def test_approval_requires_prompt(self):
        """Should require prompt field."""
        with pytest.raises(ValidationError):
            ApprovalNode(approval_type="human")

    def test_approval_requires_type(self):
        """Should require approval_type field."""
        with pytest.raises(ValidationError):
            ApprovalNode(prompt="Test")


class TestApprovalStatusTransitions:
    """Test approval status transitions."""

    def test_pending_to_approved(self):
        """Should transition from PENDING to APPROVED."""
        approval = ApprovalNode(
            approval_type="human",
            prompt="Test",
            status=ApprovalStatus.PENDING
        )

        approved = approval.model_copy(update={"status": ApprovalStatus.APPROVED})
        assert approved.status == ApprovalStatus.APPROVED

    def test_pending_to_rejected(self):
        """Should transition from PENDING to REJECTED."""
        approval = ApprovalNode(
            approval_type="human",
            prompt="Test",
            status=ApprovalStatus.PENDING
        )

        rejected = approval.model_copy(update={"status": ApprovalStatus.REJECTED})
        assert rejected.status == ApprovalStatus.REJECTED

    def test_pending_to_timeout(self):
        """Should transition from PENDING to TIMEOUT."""
        approval = ApprovalNode(
            approval_type="human",
            prompt="Test",
            status=ApprovalStatus.PENDING
        )

        timeout = approval.model_copy(update={"status": ApprovalStatus.TIMEOUT})
        assert timeout.status == ApprovalStatus.TIMEOUT

    def test_pending_to_escalated(self):
        """Should transition from PENDING to ESCALATED."""
        approval = ApprovalNode(
            approval_type="human",
            prompt="Test",
            status=ApprovalStatus.PENDING
        )

        escalated = approval.model_copy(update={"status": ApprovalStatus.ESCALATED})
        assert escalated.status == ApprovalStatus.ESCALATED
