# chuk_ai_planner/models/execution.py
from __future__ import annotations
from typing import Any, Dict, Literal
from pydantic import Field

# imports
from .base import GraphNode, NodeKind

# all
__all__ = ["ToolCall", "TaskRun"]

class ToolCall(GraphNode):
    kind: Literal[NodeKind.TOOL_CALL] = Field(NodeKind.TOOL_CALL, frozen=True)
    data: Dict[str, Any] = Field(default_factory=dict)  # {name, args, result, …}


class TaskRun(GraphNode):
    kind: Literal[NodeKind.TASK_RUN] = Field(NodeKind.TASK_RUN, frozen=True)
    data: Dict[str, Any] = Field(default_factory=dict)  # executor metadata, cost, …
