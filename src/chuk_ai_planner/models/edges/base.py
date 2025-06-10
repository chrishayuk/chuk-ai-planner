# chuk_ai_planner/models/edges/base.py
from __future__ import annotations
import copy
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator, field_validator

# Import the freezing utilities from the main base module
from ..base import _deep_freeze, _unfreeze_for_json

# all
__all__ = ["EdgeKind", "GraphEdge"]


class EdgeKind(str, Enum):
    """Canonical edge semantics."""
    PARENT_CHILD = "parent_child"   # hierarchy
    NEXT = "next"                   # temporal / sequential
    PLAN_LINK = "plan_link"         # plan-to-task / step
    STEP_ORDER = "step_order"       # step-1 → step-2
    CUSTOM = "custom"               # catch-all


class _FrozenMixin:
    """
    Convert Pydantic-v2's ValidationError on mutation into a TypeError,
    so tests with `with pytest.raises(TypeError)` pass.
    """
    def __setattr__(self, name: str, value: Any) -> None:  # noqa: D401
        try:
            super().__setattr__(name, value)  # type: ignore[misc]
        except ValidationError as exc:
            raise TypeError(str(exc)) from None


class GraphEdge(_FrozenMixin, BaseModel):
    """Directed edge between two GraphNode ids."""
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    kind: EdgeKind
    src: str
    dst: str
    data: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('src', 'dst')
    @classmethod
    def validate_node_ids(cls, v: Any) -> str:
        """Validate that src and dst are strings."""
        if not isinstance(v, str):
            raise TypeError(f"Node ID must be a string, got {type(v)}")
        # Allow empty strings - they might be valid in some contexts
        return v

    @field_validator('data')
    @classmethod
    def validate_data_type(cls, v: Any) -> Dict[str, Any]:
        """Validate that data is a dictionary."""
        if not isinstance(v, dict):
            raise TypeError(f"Data must be a dictionary, got {type(v)}")
        return v

    @model_validator(mode="after")
    def _freeze_data(self):
        """
        Deep freeze the data dictionary to ensure complete immutability.
        
        This validator:
        1. Deep copies the data to prevent external mutations
        2. Recursively freezes all nested structures
        3. Wraps the result in MappingProxyType for top-level immutability
        """
        # Deep copy the data to prevent external references from affecting the edge
        data_copy = copy.deepcopy(dict(self.data))
        
        # Deep freeze all nested structures
        frozen_data = _deep_freeze(data_copy)
        
        # Use object.__setattr__ to bypass the frozen model restriction
        # This is safe because we're in the validator during object creation
        object.__setattr__(self, "data", frozen_data)
        
        return self

    def __repr__(self) -> str:  # noqa: D401
        return f"<{self.kind.value}:{self.src[:6]}→{self.dst[:6]}>"
    
    def __hash__(self) -> int:
        # Allow GraphEdge instances to be used in sets/maps
        return hash(self.id)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert edge to a JSON-serializable dictionary.
        
        Returns
        -------
        Dict[str, Any]
            A dictionary representation suitable for JSON serialization
        """
        return {
            "id": self.id,
            "kind": self.kind.value,
            "src": self.src,
            "dst": self.dst,
            "data": _unfreeze_for_json(self.data)
        }
    
    def get_json_serializable_data(self) -> Dict[str, Any]:
        """
        Get the data field in a JSON-serializable format.
        
        Returns
        -------
        Dict[str, Any]
            The data field converted to regular Python types
        """
        return _unfreeze_for_json(self.data)
    
    def __eq__(self, other: object) -> bool:
        """Edges are equal if they have the same ID."""
        if not isinstance(other, GraphEdge):
            return NotImplemented
        return self.id == other.id