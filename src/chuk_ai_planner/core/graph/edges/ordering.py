# chuk_ai_planner/graph/edges/ordering.py
"""
Ordering edge types.

Edges representing temporal or sequential relationships.
Pure Pydantic with typed fields.
"""

from typing import Any, Literal, Optional

from pydantic import Field

from chuk_ai_planner.core.graph.edges.base import GraphEdge
from chuk_ai_planner.core.graph.types import EdgeType

__all__ = ["NextEdge", "CustomEdge"]


class NextEdge(GraphEdge):
    """
    A temporal/sequential ordering between nodes.

    Indicates that dst should be processed after src in time.
    More general than dependency - just indicates sequence.

    Examples
    --------
    >>> message_then_response = NextEdge(
    ...     src=user_message_id,
    ...     dst=assistant_message_id
    ... )
    """

    kind: Literal[EdgeType.NEXT] = EdgeType.NEXT

    # Optional weight/priority
    weight: Optional[float] = None


class CustomEdge(GraphEdge):
    """
    A custom edge type for extensibility.

    Use this for domain-specific edge types that don't fit
    the predefined categories. Store type info in custom_type.

    Examples
    --------
    >>> approval_edge = CustomEdge(
    ...     src=step_id,
    ...     dst=approval_id,
    ...     custom_type="requires_approval"
    ... )
    """

    kind: Literal[EdgeType.CUSTOM] = EdgeType.CUSTOM

    # Custom type identifier
    custom_type: str

    # Custom properties
    properties: dict[str, Any] = Field(default_factory=dict)
