# chuk_ai_planner/models/base.py
from __future__ import annotations
import copy
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Union
from uuid import uuid4
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
    field_validator,
)

# all
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


def _deep_freeze(obj: Any) -> Any:
    """
    Recursively freeze data structures to make them deeply immutable.
    
    This version preserves more natural Python types while still ensuring immutability:
    - Dicts become MappingProxyType (read-only but still dict-like)
    - Lists stay as lists but become read-only (we'll handle this differently)
    - Sets become frozensets
    - Primitive types remain unchanged
    
    Parameters
    ----------
    obj : Any
        The object to freeze
        
    Returns
    -------
    Any
        The frozen object
    """
    if isinstance(obj, dict):
        # Recursively freeze all values, then wrap in MappingProxyType
        frozen_dict = {key: _deep_freeze(value) for key, value in obj.items()}
        return MappingProxyType(frozen_dict)
    elif isinstance(obj, list):
        # Keep as list for JSON serialization compatibility, but make read-only
        frozen_items = [_deep_freeze(item) for item in obj]
        # Return a custom read-only list wrapper
        return _ReadOnlyList(frozen_items)
    elif isinstance(obj, tuple):
        # Tuples are already immutable, just freeze contents
        frozen_items = tuple(_deep_freeze(item) for item in obj)
        return frozen_items
    elif isinstance(obj, set):
        # Convert sets to frozensets and freeze contents
        frozen_items = frozenset(_deep_freeze(item) for item in obj)
        return frozen_items
    else:
        # Primitive types (str, int, float, bool, None) are already immutable
        return obj


class _ReadOnlyList:
    """
    A read-only list wrapper that prevents modifications while maintaining
    list-like interface for JSON serialization and equality checks.
    """
    
    def __init__(self, items):
        object.__setattr__(self, '_items', tuple(items))
    
    def __getitem__(self, index):
        return self._items[index]
    
    def __len__(self):
        return len(self._items)
    
    def __iter__(self):
        return iter(self._items)
    
    def __eq__(self, other):
        if isinstance(other, (list, tuple, _ReadOnlyList)):
            return list(self._items) == list(other)
        return False
    
    def __repr__(self):
        return f"ReadOnlyList({list(self._items)})"
    
    def __str__(self):
        return str(list(self._items))
    
    # Prevent all mutating operations
    def append(self, *args):
        raise TypeError("Cannot modify read-only list")
    
    def extend(self, *args):
        raise TypeError("Cannot modify read-only list")
    
    def insert(self, *args):
        raise TypeError("Cannot modify read-only list")
    
    def remove(self, *args):
        raise TypeError("Cannot modify read-only list")
    
    def pop(self, *args):
        raise TypeError("Cannot modify read-only list")
    
    def clear(self):
        raise TypeError("Cannot modify read-only list")
    
    def sort(self, *args):
        raise TypeError("Cannot modify read-only list")
    
    def reverse(self):
        raise TypeError("Cannot modify read-only list")
    
    def __setitem__(self, *args):
        raise TypeError("Cannot modify read-only list")
    
    def __delitem__(self, *args):
        raise TypeError("Cannot modify read-only list")
    
    # Make it JSON serializable
    def __json__(self):
        return list(self._items)
    
    # Support for json.dumps
    def __iter__(self):
        return iter(self._items)


def _unfreeze_for_json(obj: Any) -> Any:
    """
    Convert frozen structures back to regular Python types for JSON serialization.
    
    Parameters
    ----------
    obj : Any
        The frozen object to unfreeze
        
    Returns
    -------
    Any
        Regular Python object suitable for JSON serialization
    """
    if isinstance(obj, MappingProxyType):
        return {key: _unfreeze_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, _ReadOnlyList):
        return [_unfreeze_for_json(item) for item in obj._items]
    elif isinstance(obj, tuple):
        # Convert tuples back to lists for JSON compatibility
        return [_unfreeze_for_json(item) for item in obj]
    elif isinstance(obj, frozenset):
        # Convert frozensets back to lists (sets aren't JSON serializable)
        return [_unfreeze_for_json(item) for item in obj]
    else:
        return obj


class _FrozenMixin:
    """
    Convert Pydantic-v2's ValidationError on mutation into a TypeError,
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

    @field_validator('ts')
    @classmethod
    def validate_timestamp(cls, v: Any) -> datetime:
        """Validate that timestamp is a timezone-aware datetime."""
        if isinstance(v, str):
            try:
                # Try to parse ISO format strings
                parsed = datetime.fromisoformat(v.replace('Z', '+00:00'))
                # Ensure it has timezone info
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except ValueError:
                raise ValueError(f"Invalid datetime string: {v}")
        elif isinstance(v, datetime):
            # If it's a naive datetime, add UTC timezone
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v
        elif isinstance(v, (int, float)):
            # Accept Unix timestamps and convert them
            try:
                return datetime.fromtimestamp(v, tz=timezone.utc)
            except (ValueError, OSError):
                raise ValueError(f"Invalid timestamp: {v}")
        else:
            raise TypeError(f"Timestamp must be datetime, ISO string, or Unix timestamp, got {type(v)}")

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
        # Deep copy the data to prevent external references from affecting the node
        data_copy = copy.deepcopy(dict(self.data))
        
        # Deep freeze all nested structures
        frozen_data = _deep_freeze(data_copy)
        
        # Use object.__setattr__ to bypass the frozen model restriction
        # This is safe because we're in the validator during object creation
        object.__setattr__(self, "data", frozen_data)
        
        return self

    def __repr__(self) -> str:
        return f"<{self.kind.value}:{self.id[:8]}>"
    
    def __hash__(self) -> int:
        # Allow GraphNode instances to be used in sets/maps
        return hash(self.id)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert node to a JSON-serializable dictionary.
        
        This method converts all frozen structures back to regular Python types
        for JSON serialization while preserving the data structure.
        
        Returns
        -------
        Dict[str, Any]
            A dictionary representation suitable for JSON serialization
        """
        return {
            "id": self.id,
            "kind": self.kind.value,
            "ts": self.ts.isoformat(),
            "data": _unfreeze_for_json(self.data)
        }
    
    def get_json_serializable_data(self) -> Dict[str, Any]:
        """
        Get the data field in a JSON-serializable format.
        
        This is useful when you need to serialize just the data field
        (e.g., for tool calls in the executor).
        
        Returns
        -------
        Dict[str, Any]
            The data field converted to regular Python types
        """
        return _unfreeze_for_json(self.data)
    
    def __eq__(self, other: object) -> bool:
        """Nodes are equal if they have the same ID."""
        if not isinstance(other, GraphNode):
            return NotImplemented
        return self.id == other.id