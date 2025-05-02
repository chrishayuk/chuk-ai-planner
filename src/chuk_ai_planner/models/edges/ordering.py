# chuk_ai_planner/models/edges/ordering.py
from __future__ import annotations
from typing import Any, Dict, Literal
from pydantic import Field

# imports
from .base import GraphEdge, EdgeKind

# all
__all__ = ["NextEdge"]


class NextEdge(GraphEdge):
    """Temporal ordering (e.g. msg-1 → msg-2)."""
    kind: Literal[EdgeKind.NEXT] = Field(EdgeKind.NEXT, frozen=True)
    data: Dict[str, Any] = Field(default_factory=dict)

