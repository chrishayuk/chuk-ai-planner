# chuk_ai_planner/graph/edges/routing.py
"""
Routing edge types.

Edges used in conditional routing and decision points.
Pure Pydantic with typed fields.
"""

from typing import Literal

from chuk_ai_planner.core.graph.edges.base import GraphEdge
from chuk_ai_planner.core.graph.types import EdgeType

__all__ = ["RouteEdge"]


class RouteEdge(GraphEdge):
    """
    A routing path from a router step to a target step.

    Represents one possible path that a router can take.
    The router evaluates a condition and chooses which RouteEdge to follow.

    Examples
    --------
    >>> high_quality_route = RouteEdge(
    ...     src=router_id,
    ...     dst=publish_step_id,
    ...     route_key="high_quality"
    ... )
    >>> low_quality_route = RouteEdge(
    ...     src=router_id,
    ...     dst=revise_step_id,
    ...     route_key="low_quality",
    ...     is_default=True
    ... )
    """

    kind: Literal[EdgeType.ROUTE] = EdgeType.ROUTE

    # The key identifying this route
    # Must match one of the routes in the RouterStep
    route_key: str

    # Whether this is the default/fallback route
    is_default: bool = False
