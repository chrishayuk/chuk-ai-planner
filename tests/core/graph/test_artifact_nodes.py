"""
Tests for artifact-specific nodes.

Tests for ArtifactNode and artifact lineage tracking.
"""

import pytest
from pydantic import ValidationError

from chuk_ai_planner.core.graph.nodes.artifact import ArtifactNode
from chuk_ai_planner.core.graph.types import NodeType


class TestArtifactNode:
    """Test ArtifactNode for artifact tracking and lineage."""

    def test_create_basic_artifact(self):
        """Should create artifact node with required fields."""
        artifact = ArtifactNode(
            artifact_id="vid_123",
            artifact_type="video",
            storage_path="/session/foo/video.mp4",
        )

        assert artifact.kind == NodeType.ARTIFACT
        assert artifact.artifact_id == "vid_123"
        assert artifact.artifact_type == "video"
        assert artifact.storage_path == "/session/foo/video.mp4"
        assert artifact.presigned_url is None
        assert artifact.produced_by_step is None
        assert artifact.consumed_by_steps == []

    def test_artifact_with_lineage(self):
        """Should track producer and consumers."""
        artifact = ArtifactNode(
            artifact_id="script_456",
            artifact_type="script",
            storage_path="/artifacts/script.txt",
            produced_by_step="write_script_step",
            consumed_by_steps=["review_step", "video_step"],
        )

        assert artifact.produced_by_step == "write_script_step"
        assert len(artifact.consumed_by_steps) == 2
        assert "review_step" in artifact.consumed_by_steps
        assert "video_step" in artifact.consumed_by_steps

    def test_artifact_with_presigned_url(self):
        """Should support presigned URLs for temporary access."""
        artifact = ArtifactNode(
            artifact_id="img_789",
            artifact_type="image",
            storage_path="/s3/images/thumb.jpg",
            presigned_url="https://s3.aws.com/temp/thumb.jpg?expires=...",
        )

        assert artifact.presigned_url == "https://s3.aws.com/temp/thumb.jpg?expires=..."

    def test_artifact_with_metadata(self):
        """Should track size, mime type, and checksum."""
        artifact = ArtifactNode(
            artifact_id="doc_101",
            artifact_type="document",
            storage_path="/docs/report.pdf",
            size_bytes=1024000,
            mime_type="application/pdf",
            checksum="sha256:abc123def456",
        )

        assert artifact.size_bytes == 1024000
        assert artifact.mime_type == "application/pdf"
        assert artifact.checksum == "sha256:abc123def456"

    def test_artifact_types(self):
        """Should support various artifact types."""
        for artifact_type in [
            "video",
            "script",
            "image",
            "audio",
            "slide_deck",
            "thumbnail",
            "document",
        ]:
            artifact = ArtifactNode(
                artifact_id=f"test_{artifact_type}",
                artifact_type=artifact_type,
                storage_path=f"/path/to/{artifact_type}",
            )
            assert artifact.artifact_type == artifact_type

    def test_artifact_immutable(self):
        """Should be immutable (frozen)."""
        artifact = ArtifactNode(
            artifact_id="test", artifact_type="video", storage_path="/path"
        )

        with pytest.raises(ValidationError):
            artifact.storage_path = "/new/path"

    def test_artifact_has_graph_node_fields(self):
        """Should have all GraphNode base fields."""
        artifact = ArtifactNode(
            artifact_id="test", artifact_type="video", storage_path="/path"
        )

        assert hasattr(artifact, "id")
        assert hasattr(artifact, "kind")
        assert hasattr(artifact, "ts")
        assert hasattr(artifact, "metadata")

    def test_artifact_model_copy(self):
        """Should support model_copy for updates."""
        artifact = ArtifactNode(
            artifact_id="test",
            artifact_type="video",
            storage_path="/path",
            consumed_by_steps=[],
        )

        updated = artifact.model_copy(update={"consumed_by_steps": ["step1", "step2"]})

        assert len(updated.consumed_by_steps) == 2
        assert len(artifact.consumed_by_steps) == 0  # Original unchanged

    def test_artifact_repr(self):
        """Should have useful repr."""
        artifact = ArtifactNode(
            artifact_id="test", artifact_type="video", storage_path="/path"
        )
        repr_str = repr(artifact)

        assert "artifact" in repr_str
        assert artifact.id[:8] in repr_str

    def test_artifact_requires_id(self):
        """Should require artifact_id field."""
        with pytest.raises(ValidationError):
            ArtifactNode(artifact_type="video", storage_path="/path")

    def test_artifact_requires_type(self):
        """Should require artifact_type field."""
        with pytest.raises(ValidationError):
            ArtifactNode(artifact_id="test", storage_path="/path")

    def test_artifact_requires_storage_path(self):
        """Should require storage_path field."""
        with pytest.raises(ValidationError):
            ArtifactNode(artifact_id="test", artifact_type="video")


class TestArtifactLineage:
    """Test artifact lineage tracking."""

    def test_producer_tracking(self):
        """Should track which step produced the artifact."""
        artifact = ArtifactNode(
            artifact_id="test",
            artifact_type="video",
            storage_path="/path",
            produced_by_step="render_step",
        )

        assert artifact.produced_by_step == "render_step"

    def test_consumer_tracking(self):
        """Should track which steps consume the artifact."""
        artifact = ArtifactNode(
            artifact_id="test",
            artifact_type="video",
            storage_path="/path",
            consumed_by_steps=["upload_step", "thumbnail_step", "preview_step"],
        )

        assert len(artifact.consumed_by_steps) == 3
        assert "upload_step" in artifact.consumed_by_steps
        assert "thumbnail_step" in artifact.consumed_by_steps
        assert "preview_step" in artifact.consumed_by_steps

    def test_adding_consumers_via_copy(self):
        """Should be able to add consumers via model_copy."""
        artifact = ArtifactNode(
            artifact_id="test",
            artifact_type="video",
            storage_path="/path",
            consumed_by_steps=["step1"],
        )

        updated = artifact.model_copy(
            update={"consumed_by_steps": artifact.consumed_by_steps + ["step2"]}
        )

        assert len(updated.consumed_by_steps) == 2
        assert "step1" in updated.consumed_by_steps
        assert "step2" in updated.consumed_by_steps

    def test_no_consumers_initially(self):
        """Should have empty consumers list initially."""
        artifact = ArtifactNode(
            artifact_id="test", artifact_type="video", storage_path="/path"
        )

        assert artifact.consumed_by_steps == []
        assert isinstance(artifact.consumed_by_steps, list)
