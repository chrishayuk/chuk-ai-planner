# chuk_ai_planner/graph/edges/hierarchy.py
"""
Hierarchy edge types.

Edges representing parent-child and containment relationships.
Pure Pydantic with typed fields.
"""

from typing import Literal

from chuk_ai_planner.core.graph.edges.base import GraphEdge
from chuk_ai_planner.core.graph.types import EdgeType

__all__ = ["ParentChildEdge"]


class ParentChildEdge(GraphEdge):
    """
    A parent-child relationship between nodes.

    Represents hierarchical containment, such as a plan containing steps,
    or a session containing plans.

    The parent (src) contains or owns the child (dst).

    Examples
    --------
    >>> plan_contains_step = ParentChildEdge(
    ...     src=plan_id,
    ...     dst=step_id
    ... )
    >>> session_contains_plan = ParentChildEdge(
    ...     src=session_id,
    ...     dst=plan_id
    ... )
    """

    kind: Literal[EdgeType.PARENT_CHILD] = EdgeType.PARENT_CHILD
