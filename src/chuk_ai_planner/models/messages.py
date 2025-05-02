# chuk_ai_planner/models/messages.py
from __future__ import annotations
from typing import Any, Dict, Literal
from pydantic import Field

# imports
from .base import GraphNode, NodeKind

# all
__all__ = ["UserMessage", "AssistantMessage"]


class UserMessage(GraphNode):
    kind: Literal[NodeKind.USER_MSG] = Field(NodeKind.USER_MSG, frozen=True)
    data: Dict[str, str] = Field(default_factory=dict)  # {"content": "…"}


class AssistantMessage(GraphNode):
    kind: Literal[NodeKind.ASSIST_MSG] = Field(NodeKind.ASSIST_MSG, frozen=True)
    data: Dict[str, Any] = Field(default_factory=dict)  # {"content": …, "tool_calls": …}
