# chuk_ai_planner/models/meta.py
from __future__ import annotations
from typing import Any, Dict, Literal
from pydantic import Field, field_validator

# imports
from .base import GraphNode, NodeKind

# all
__all__ = ["Summary"]


class Summary(GraphNode):
    kind: Literal[NodeKind.SUMMARY] = Field(NodeKind.SUMMARY, frozen=True)
    data: Dict[str, str] = Field(default_factory=dict)  # Only string values allowed
    
    @field_validator('data')
    @classmethod
    def validate_string_data(cls, v: Dict[str, Any]) -> Dict[str, str]:
        """Validate that all data values are strings."""
        if not isinstance(v, dict):
            raise TypeError(f"Data must be a dictionary, got {type(v)}")
        
        string_data = {}
        for key, value in v.items():
            if not isinstance(key, str):
                raise TypeError(f"Data keys must be strings, got {type(key)} for key: {key}")
            if not isinstance(value, str):
                raise TypeError(f"Data values must be strings, got {type(value)} for key '{key}': {value}")
            string_data[key] = value
        
        return string_data