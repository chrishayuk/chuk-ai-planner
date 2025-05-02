# chuk_ai_planner/models/edges/planning.py
from __future__ import annotations
from typing import Any, Dict, Literal
from pydantic import Field

# imports
from .base import GraphEdge, EdgeKind

# all
__all__ = ["PlanEdge", "StepEdge"]


class PlanEdge(GraphEdge):
    """PlanNode → TaskRun / ToolCall."""
    kind: Literal[EdgeKind.PLAN_LINK] = Field(EdgeKind.PLAN_LINK, frozen=True)
    data: Dict[str, Any] = Field(default_factory=dict)


class StepEdge(GraphEdge):
    """PlanStepᵢ → PlanStepᵢ₊₁ (linear or DAG)."""
    kind: Literal[EdgeKind.STEP_ORDER] = Field(EdgeKind.STEP_ORDER, frozen=True)
    data: Dict[str, Any] = Field(default_factory=dict)

