# chuk_ai_planner/models/planning.py
from __future__ import annotations
from typing import Any, Dict, Literal
from pydantic import Field, ConfigDict

# imports
from .base import GraphNode, NodeKind

# all
__all__ = ["PlanNode", "PlanStep"]


class PlanNode(GraphNode):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    # constant, immutable field
    kind: Literal[NodeKind.PLAN] = Field(NodeKind.PLAN, frozen=True)
    # payload
    data: Dict[str, Any] = Field(default_factory=dict)


class PlanStep(GraphNode):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    kind: Literal[NodeKind.PLAN_STEP] = Field(NodeKind.PLAN_STEP, frozen=True)
    data: Dict[str, Any] = Field(default_factory=dict)
