# chuk_ai_planner/graph/nodes/artifact.py
"""
Artifact-specific graph nodes.

Nodes for tracking artifacts and their lineage through the workflow.
Integrates with chuk-artifacts for storage and retrieval.
"""

from typing import List, Literal, Optional

from pydantic import Field

from chuk_ai_planner.core.graph.nodes.base import GraphNode
from chuk_ai_planner.core.graph.types import NodeType

__all__ = ["ArtifactNode"]


class ArtifactNode(GraphNode):
    """
    Artifact reference and lineage tracking.

    Represents an artifact (video, script, image, etc.) in the workflow.
    Tracks which steps produced and consumed it.

    Integrates with chuk-artifacts for actual storage.

    Examples
    --------
    >>> artifact = ArtifactNode(
    ...     artifact_id="vid_123",
    ...     artifact_type="video",
    ...     storage_path="/session/foo/video.mp4",
    ...     produced_by_step="render_step"
    ... )
    >>> artifact.artifact_type
    'video'
    """

    kind: Literal[NodeType.ARTIFACT] = NodeType.ARTIFACT

    # Artifact identity
    artifact_id: str  # Unique artifact ID (chuk-artifacts reference)
    artifact_type: str  # "video", "script", "thumbnail", "audio", etc.

    # Storage
    storage_path: str  # Path in artifact storage
    presigned_url: Optional[str] = None  # Temporary download URL

    # Lineage tracking
    produced_by_step: Optional[str] = None  # Step ID that created this
    consumed_by_steps: List[str] = Field(default_factory=list)  # Steps that use this

    # Metadata
    size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    checksum: Optional[str] = None  # For integrity verification
