# chuk_ai_planner/utils/serialization.py
"""
JSON serialization utilities for graph models.

Provides functions to safely serialize graph nodes and edges to JSON,
handling the MappingProxyType and other immutable structures.
"""

import json
from types import MappingProxyType
from typing import Any, Dict

__all__ = ["unfreeze_data", "serialize_node_data", "serialize_tool_args"]


def unfreeze_data(obj: Any) -> Any:
    """
    Recursively convert frozen structures to regular Python types for JSON serialization.
    
    Parameters
    ----------
    obj : Any
        The object to unfreeze
        
    Returns
    -------
    Any
        The unfrozen object suitable for JSON serialization
    """
    if isinstance(obj, MappingProxyType):
        return {key: unfreeze_data(value) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        # Convert tuples back to lists for JSON compatibility
        return [unfreeze_data(item) for item in obj]
    elif isinstance(obj, frozenset):
        # Convert frozensets back to lists (sets aren't JSON serializable)
        return [unfreeze_data(item) for item in obj]
    else:
        return obj


def serialize_node_data(data: Any) -> str:
    """
    Serialize node data to JSON string, handling frozen structures.
    
    Parameters
    ----------
    data : Any
        The data to serialize
        
    Returns
    -------
    str
        JSON string representation
    """
    unfrozen = unfreeze_data(data)
    return json.dumps(unfrozen)


def serialize_tool_args(args: Any) -> str:
    """
    Serialize tool arguments to JSON string, handling frozen structures.
    
    This is specifically for tool call arguments that need to be JSON serialized.
    
    Parameters
    ----------
    args : Any
        The arguments to serialize
        
    Returns
    -------
    str
        JSON string representation
    """
    unfrozen = unfreeze_data(args)
    return json.dumps(unfrozen)