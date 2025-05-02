# chuk_ai_planner/models/edges/hierarchy.py
from __future__ import annotations
from typing import Any, Dict, Literal
from pydantic import Field

# imports
from .base import GraphEdge, EdgeKind

# all
__all__ = ["ParentChildEdge"]


class ParentChildEdge(GraphEdge):
    """session → message   /   message → tool_call …"""
    kind: Literal[EdgeKind.PARENT_CHILD] = Field(EdgeKind.PARENT_CHILD, frozen=True)
    data: Dict[str, Any] = Field(default_factory=dict)
