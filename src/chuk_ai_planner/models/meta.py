# chuk_ai_planner/models/meta.py
from __future__ import annotations
from typing import Dict, Literal
from pydantic import Field

# imports
from .base import GraphNode, NodeKind

# all
__all__ = ["Summary"]


class Summary(GraphNode):
    kind: Literal[NodeKind.SUMMARY] = Field(NodeKind.SUMMARY, frozen=True)
    data: Dict[str, str] = Field(default_factory=dict)  # {"note": "…"}
