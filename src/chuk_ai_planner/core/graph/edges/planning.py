# chuk_ai_planner/graph/edges/planning.py
"""
Planning edge types.

Edges used in plan structure and step relationships.
Pure Pydantic with typed fields.
"""

from typing import Literal, Optional

from chuk_ai_planner.core.graph.edges.base import GraphEdge
from chuk_ai_planner.core.graph.types import EdgeType

__all__ = ["PlanLinkEdge", "StepEdge"]


class PlanLinkEdge(GraphEdge):
    """
    Links a plan to one of its components.

    Represents the relationship between a plan and its steps,
    routers, or other plan elements.

    Examples
    --------
    >>> plan_to_step = PlanLinkEdge(
    ...     src=plan_id,
    ...     dst=step_id
    ... )
    """

    kind: Literal[EdgeType.PLAN_LINK] = EdgeType.PLAN_LINK


class StepEdge(GraphEdge):
    """
    Defines ordering or dependencies between steps.

    Represents that one step should execute after another,
    or that one step depends on another's completion.

    Examples
    --------
    >>> step1_then_step2 = StepEdge(
    ...     src=step1_id,
    ...     dst=step2_id,
    ...     dependency=True
    ... )
    """

    kind: Literal[EdgeType.STEP_ORDER] = EdgeType.STEP_ORDER

    # Whether dst depends on src completing
    dependency: bool = True

    # Optional condition for the edge
    condition: Optional[str] = None
