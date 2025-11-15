# chuk_ai_planner/graph/edges/__init__.py
"""
Graph edge types.

All edge types in the graph system, organized by category.
Import from here for convenience.
"""

from chuk_ai_planner.core.graph.edges.base import GraphEdge
from chuk_ai_planner.core.graph.edges.hierarchy import ParentChildEdge
from chuk_ai_planner.core.graph.edges.ordering import CustomEdge, NextEdge
from chuk_ai_planner.core.graph.edges.planning import PlanLinkEdge, StepEdge
from chuk_ai_planner.core.graph.edges.routing import RouteEdge
from chuk_ai_planner.core.graph.edges.workflow import (
    ApprovalEdge,
    ArtifactDependencyEdge,
    FallbackEdge,
)

__all__ = [
    # Base
    "GraphEdge",
    # Hierarchy
    "ParentChildEdge",
    # Planning
    "PlanLinkEdge",
    "StepEdge",
    # Routing
    "RouteEdge",
    # Ordering
    "NextEdge",
    "CustomEdge",
    # Workflow
    "ApprovalEdge",
    "FallbackEdge",
    "ArtifactDependencyEdge",
]
