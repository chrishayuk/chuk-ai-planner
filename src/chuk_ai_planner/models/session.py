# chuk_ai_planner/models/session.py
from __future__ import annotations
from typing import Any, Dict, Literal
from pydantic import Field

# imports
from .base import GraphNode, NodeKind

# all
__all__ = ["SessionNode"]


class SessionNode(GraphNode):
    kind: Literal[NodeKind.SESSION] = Field(NodeKind.SESSION, frozen=True)
    data: Dict[str, Any] = Field(default_factory=dict)

