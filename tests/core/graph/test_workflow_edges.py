"""
Tests for workflow-specific edges.

Tests for ApprovalEdge, FallbackEdge, and ArtifactDependencyEdge.
"""

import pytest
from pydantic import ValidationError

from chuk_ai_planner.core.graph.edges.workflow import (
    ApprovalEdge,
    FallbackEdge,
    ArtifactDependencyEdge,
)
from chuk_ai_planner.core.graph.types import EdgeType


class TestApprovalEdge:
    """Test ApprovalEdge for approval gate flow."""

    def test_create_approval_edge(self):
        """Should create approval edge with all routing."""
        edge = ApprovalEdge(
            src="approval_node_123",
            dst="publish_step",  # Default destination
            approval_node_id="approval_node_123",
            on_approved="publish_step",
            on_rejected="revise_step",
            on_timeout="escalate_step",
        )

        assert edge.kind == EdgeType.APPROVAL
        assert edge.src == "approval_node_123"
        assert edge.dst == "publish_step"
        assert edge.approval_node_id == "approval_node_123"
        assert edge.on_approved == "publish_step"
        assert edge.on_rejected == "revise_step"
        assert edge.on_timeout == "escalate_step"

    def test_approval_edge_immutable(self):
        """Should be immutable (frozen)."""
        edge = ApprovalEdge(
            src="approval",
            dst="next",
            approval_node_id="approval",
            on_approved="approved",
            on_rejected="rejected",
            on_timeout="timeout",
        )

        with pytest.raises(ValidationError):
            edge.on_approved = "different"

    def test_approval_edge_has_id(self):
        """Should have auto-generated ID."""
        edge = ApprovalEdge(
            src="approval",
            dst="next",
            approval_node_id="approval",
            on_approved="approved",
            on_rejected="rejected",
            on_timeout="timeout",
        )

        assert edge.id is not None
        assert len(edge.id) > 0

    def test_approval_edge_repr(self):
        """Should have useful repr."""
        edge = ApprovalEdge(
            src="approval",
            dst="next",
            approval_node_id="approval",
            on_approved="approved",
            on_rejected="rejected",
            on_timeout="timeout",
        )
        repr_str = repr(edge)

        assert "approval" in repr_str

    def test_approval_edge_requires_fields(self):
        """Should require all approval routing fields."""
        with pytest.raises(ValidationError):
            ApprovalEdge(
                src="approval",
                dst="next",
                on_approved="approved",
                on_rejected="rejected",
                on_timeout="timeout",
                # Missing approval_node_id
            )

    def test_approval_edge_supports_different_outcomes(self):
        """Should support routing to different steps for each outcome."""
        edge = ApprovalEdge(
            src="approval",
            dst="step1",
            approval_node_id="approval",
            on_approved="step1",
            on_rejected="step2",
            on_timeout="step3",
        )

        assert edge.on_approved != edge.on_rejected
        assert edge.on_rejected != edge.on_timeout
        assert edge.on_approved != edge.on_timeout


class TestFallbackEdge:
    """Test FallbackEdge for error handling."""

    def test_create_fallback_edge(self):
        """Should create fallback edge with default trigger."""
        edge = FallbackEdge(src="risky_step", dst="safe_step")

        assert edge.kind == EdgeType.FALLBACK
        assert edge.src == "risky_step"
        assert edge.dst == "safe_step"
        assert edge.trigger_on == ["error"]  # Default
        assert edge.priority == 0
        assert edge.max_cost_exceeded is False

    def test_fallback_with_multiple_triggers(self):
        """Should support multiple trigger conditions."""
        edge = FallbackEdge(
            src="step",
            dst="fallback",
            trigger_on=["error", "timeout", "max_retries_exceeded"],
        )

        assert "error" in edge.trigger_on
        assert "timeout" in edge.trigger_on
        assert "max_retries_exceeded" in edge.trigger_on

    def test_fallback_with_priority(self):
        """Should support priority for multiple fallbacks."""
        edge1 = FallbackEdge(src="step", dst="fallback1", priority=1)
        edge2 = FallbackEdge(src="step", dst="fallback2", priority=2)

        assert edge2.priority > edge1.priority

    def test_fallback_with_cost_trigger(self):
        """Should support cost-based triggering."""
        edge = FallbackEdge(
            src="expensive_step", dst="cheap_fallback", max_cost_exceeded=True
        )

        assert edge.max_cost_exceeded is True

    def test_fallback_edge_immutable(self):
        """Should be immutable (frozen)."""
        edge = FallbackEdge(src="step", dst="fallback")

        with pytest.raises(ValidationError):
            edge.priority = 10

    def test_fallback_edge_repr(self):
        """Should have useful repr."""
        edge = FallbackEdge(src="step", dst="fallback")
        repr_str = repr(edge)

        assert "fallback" in repr_str

    def test_multiple_fallbacks_different_priorities(self):
        """Should allow multiple fallbacks with different priorities."""
        fallback1 = FallbackEdge(src="step", dst="fallback1", priority=1)
        fallback2 = FallbackEdge(src="step", dst="fallback2", priority=2)
        fallback3 = FallbackEdge(src="step", dst="fallback3", priority=3)

        assert fallback1.priority < fallback2.priority < fallback3.priority


class TestArtifactDependencyEdge:
    """Test ArtifactDependencyEdge for artifact flow."""

    def test_create_artifact_dependency(self):
        """Should create artifact dependency edge."""
        edge = ArtifactDependencyEdge(
            src="produce_step",
            dst="consume_step",
            artifact_id="video_123",
            artifact_type="video",
        )

        assert edge.kind == EdgeType.ARTIFACT_DEPENDENCY
        assert edge.src == "produce_step"
        assert edge.dst == "consume_step"
        assert edge.artifact_id == "video_123"
        assert edge.artifact_type == "video"
        assert edge.required is True  # Default

    def test_artifact_dependency_optional(self):
        """Should support optional artifact dependencies."""
        edge = ArtifactDependencyEdge(
            src="step1",
            dst="step2",
            artifact_id="optional_art",
            artifact_type="image",
            required=False,
        )

        assert edge.required is False

    def test_artifact_dependency_types(self):
        """Should support various artifact types."""
        for artifact_type in ["video", "script", "image", "audio", "slides"]:
            edge = ArtifactDependencyEdge(
                src="producer",
                dst="consumer",
                artifact_id=f"art_{artifact_type}",
                artifact_type=artifact_type,
            )
            assert edge.artifact_type == artifact_type

    def test_artifact_dependency_immutable(self):
        """Should be immutable (frozen)."""
        edge = ArtifactDependencyEdge(
            src="producer", dst="consumer", artifact_id="art", artifact_type="video"
        )

        with pytest.raises(ValidationError):
            edge.required = False

    def test_artifact_dependency_repr(self):
        """Should have useful repr."""
        edge = ArtifactDependencyEdge(
            src="producer", dst="consumer", artifact_id="art", artifact_type="video"
        )
        repr_str = repr(edge)

        assert "artifact_dependency" in repr_str

    def test_artifact_dependency_requires_fields(self):
        """Should require artifact_id and artifact_type."""
        with pytest.raises(ValidationError):
            ArtifactDependencyEdge(
                src="producer",
                dst="consumer",
                artifact_type="video",
                # Missing artifact_id
            )

        with pytest.raises(ValidationError):
            ArtifactDependencyEdge(
                src="producer",
                dst="consumer",
                artifact_id="art",
                # Missing artifact_type
            )


class TestWorkflowEdgeCollections:
    """Test workflow edges in collections."""

    def test_edges_in_sets(self):
        """Should work in sets (hashable)."""
        edge1 = ApprovalEdge(
            src="a",
            dst="b",
            approval_node_id="a",
            on_approved="b",
            on_rejected="c",
            on_timeout="d",
        )
        edge2 = FallbackEdge(src="e", dst="f")
        edge3 = ArtifactDependencyEdge(
            src="g", dst="h", artifact_id="art", artifact_type="video"
        )

        edge_set = {edge1, edge2, edge3}
        assert len(edge_set) == 3

    def test_edges_as_dict_keys(self):
        """Should work as dict keys (hashable)."""
        edge = ApprovalEdge(
            src="a",
            dst="b",
            approval_node_id="a",
            on_approved="b",
            on_rejected="c",
            on_timeout="d",
        )

        edge_map = {edge: "approval_route"}
        assert edge_map[edge] == "approval_route"

    def test_filtering_by_kind(self):
        """Should be able to filter by edge kind."""
        edges = [
            ApprovalEdge(
                src="a",
                dst="b",
                approval_node_id="a",
                on_approved="b",
                on_rejected="c",
                on_timeout="d",
            ),
            FallbackEdge(src="e", dst="f"),
            ArtifactDependencyEdge(
                src="g", dst="h", artifact_id="art", artifact_type="video"
            ),
        ]

        approval_edges = [e for e in edges if e.kind == EdgeType.APPROVAL]
        fallback_edges = [e for e in edges if e.kind == EdgeType.FALLBACK]
        artifact_edges = [e for e in edges if e.kind == EdgeType.ARTIFACT_DEPENDENCY]

        assert len(approval_edges) == 1
        assert len(fallback_edges) == 1
        assert len(artifact_edges) == 1
