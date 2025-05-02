# src/chuk_ai_planner/models/base.py
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict
from uuid import uuid4
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

# all
__all__ = ["NodeKind", "GraphNode"]


class NodeKind(str, Enum):
    SESSION      = "session"
    PLAN         = "plan"
    PLAN_STEP    = "plan_step"
    USER_MSG     = "user_message"
    ASSIST_MSG   = "assistant_message"
    TOOL_CALL    = "tool_call"
    TASK_RUN     = "task_run"
    SUMMARY      = "summary"


class _FrozenMixin:
    """
    Convert Pydantic-v2’s ValidationError on mutation into a TypeError,
    so tests with `with pytest.raises(TypeError)` pass.
    """
    def __setattr__(self, name: str, value: Any) -> None:
        try:
            super().__setattr__(name, value)  # type: ignore[misc]
        except ValidationError as exc:
            raise TypeError(str(exc)) from None


class GraphNode(_FrozenMixin, BaseModel):
    """Common fields shared by every node type."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    id:   str       = Field(default_factory=lambda: str(uuid4()))
    kind: NodeKind
    ts:   datetime  = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _freeze_data(self):
        # Wrap whatever data is in a read-only mapping
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))
        return self

    def __repr__(self) -> str:
        return f"<{self.kind.value}:{self.id[:8]}>"
    
    def __hash__(self) -> int:
        # allow GraphEdge instances to be used in sets/maps
        return hash(self.id)
