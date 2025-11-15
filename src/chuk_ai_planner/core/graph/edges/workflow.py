# chuk_ai_planner/graph/edges/workflow.py
"""
Workflow-specific edge types.

Edges for approval gates, error fallbacks, and artifact dependencies.
These enable resilient, interactive workflows.
"""

from typing import List, Literal

from pydantic import Field

from chuk_ai_planner.core.graph.edges.base import GraphEdge
from chuk_ai_planner.core.graph.types import EdgeType

__all__ = ["ApprovalEdge", "FallbackEdge", "ArtifactDependencyEdge"]


class ApprovalEdge(GraphEdge):
    """
    Approval gate edge with conditional flow.

    Routes to different steps based on approval outcome.
    Connects an ApprovalNode to next steps.

    Examples
    --------
    >>> edge = ApprovalEdge(
    ...     src="approval_123",
    ...     dst="publish_step",
    ...     approval_node_id="approval_123",
    ...     on_approved="publish_step",
    ...     on_rejected="revise_step",
    ...     on_timeout="escalate_step"
    ... )
    """

    kind: Literal[EdgeType.APPROVAL] = EdgeType.APPROVAL

    # Approval behavior
    approval_node_id: str  # The ApprovalNode this edge is for
    on_approved: str  # Step ID to run if approved
    on_rejected: str  # Step ID to run if rejected
    on_timeout: str  # Step ID to run if approval times out


class FallbackEdge(GraphEdge):
    """
    Error fallback edge.

    Defines alternate execution path when a step fails.
    Enables graceful degradation and error recovery.

    Examples
    --------
    >>> edge = FallbackEdge(
    ...     src="risky_step",
    ...     dst="fallback_step",
    ...     trigger_on=["error", "timeout"],
    ...     priority=1
    ... )
    """

    kind: Literal[EdgeType.FALLBACK] = EdgeType.FALLBACK

    # Fallback conditions
    trigger_on: List[str] = Field(
        default_factory=lambda: ["error"]
    )  # Conditions that trigger fallback
    priority: int = 0  # Priority for multiple fallbacks (higher = higher priority)
    max_cost_exceeded: bool = False  # Trigger if cost budget exceeded


class ArtifactDependencyEdge(GraphEdge):
    """
    Artifact dependency edge.

    Tracks artifact flow between steps.
    One step produces an artifact, another consumes it.

    Examples
    --------
    >>> edge = ArtifactDependencyEdge(
    ...     src="render_video_step",
    ...     dst="upload_video_step",
    ...     artifact_id="video_123",
    ...     artifact_type="video",
    ...     required=True
    ... )
    """

    kind: Literal[EdgeType.ARTIFACT_DEPENDENCY] = EdgeType.ARTIFACT_DEPENDENCY

    # Artifact flow
    artifact_id: str  # The artifact this edge tracks
    artifact_type: str  # Type: "video", "script", "image", etc.
    required: bool = True  # Block downstream step without this artifact?
