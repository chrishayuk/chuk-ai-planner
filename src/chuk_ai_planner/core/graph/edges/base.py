# chuk_ai_planner/graph/edges/base.py
"""
Base edge class for the graph system.

Pure Pydantic implementation - no dictionary goop!
All fields are typed and validated at the model level.
"""

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from chuk_ai_planner.core.graph.types import EdgeType

__all__ = ["GraphEdge"]


class GraphEdge(BaseModel):
    """
    Base class for all graph edges.

    Represents a directed connection between two nodes in the graph.
    Pure Pydantic - subclasses define their own typed fields.

    Design principles:
    - Immutable (frozen=True)
    - Type-safe (all fields explicitly typed)
    - Directional (src → dst)
    - Extensible (metadata for custom data)

    Examples
    --------
    Subclasses override with Literal types for kind:

    >>> class RouteEdge(GraphEdge):
    ...     kind: Literal[EdgeType.ROUTE] = EdgeType.ROUTE
    ...     route_key: str
    ...     is_default: bool = False
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    # Core fields - every edge has these
    id: str = Field(default_factory=lambda: str(uuid4()))
    kind: EdgeType  # Enforced by subclasses with Literal types
    src: str  # Source node ID
    dst: str  # Destination node ID

    # Optional metadata for extensibility
    # Use typed fields in subclasses for known data!
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        """Edges are hashed by ID for use in sets/dicts."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Edges are equal if they have the same ID."""
        if not isinstance(other, GraphEdge):
            return NotImplemented
        return self.id == other.id

    def __repr__(self) -> str:
        """Clean repr showing kind and connection."""
        return f"<{self.kind.value}:{self.src[:6]}→{self.dst[:6]}>"
