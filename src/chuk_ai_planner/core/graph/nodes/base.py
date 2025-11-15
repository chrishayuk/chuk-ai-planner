# chuk_ai_planner/graph/nodes/base.py
"""
Base node class for the graph system.

Pure Pydantic implementation - no dictionary goop!
All fields are typed and validated at the model level.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


__all__ = ["GraphNode"]


class GraphNode(BaseModel):
    """
    Base class for all graph nodes.

    This is a pure Pydantic model with no dictionary goop.
    Subclasses define their own typed fields based on their needs.

    Design principles:
    - Immutable (frozen=True)
    - Type-safe (all fields explicitly typed)
    - Self-documenting (field names are the API)
    - Extensible (metadata for custom data)

    Examples
    --------
    Subclasses override with Literal types for kind:

    >>> class PlanNode(GraphNode):
    ...     kind: Literal[NodeType.PLAN] = NodeType.PLAN
    ...     title: str
    ...     description: Optional[str] = None
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable
        arbitrary_types_allowed=True,  # For custom types
        use_enum_values=True,  # Serialize enums as values
    )

    # Core fields - every node has these
    id: str = Field(default_factory=lambda: str(uuid4()))
    kind: str  # Subclasses override with specific Literal types (core or extension)
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional metadata for extensibility
    # Use typed fields in subclasses for known data!
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        """Nodes are hashed by ID for use in sets/dicts."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Nodes are equal if they have the same ID."""
        if not isinstance(other, GraphNode):
            return NotImplemented
        return self.id == other.id

    def __repr__(self) -> str:
        """Clean repr showing kind and short ID."""
        # Handle both enum (NodeType.PLAN) and string ("user_message") kinds
        kind_str = self.kind.value if hasattr(self.kind, "value") else self.kind
        return f"<{kind_str}:{self.id[:8]}>"
