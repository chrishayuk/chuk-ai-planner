# chuk_ai_planner/utils/registry_helpers.py
"""
Utility for running a tool that is registered in chuk_tool_processor.

`execute_tool(tool_call, parent_event_id, assistant_node_id)` is meant to be
passed straight into `PlanExecutor.execute_step` (or any other place that
expects the `process_tool_call` signature).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict
from uuid import uuid4

# Updated imports for latest chuk_tool_processor
from chuk_tool_processor.registry import get_default_registry
from chuk_tool_processor.models.tool_call import ToolCall
from chuk_tool_processor.execution.strategies.inprocess_strategy import InProcessStrategy
from chuk_tool_processor.execution.tool_executor import ToolExecutor

# Global executor instance
_executor = None

async def _get_executor():
    """Get or initialize the global tool executor."""
    global _executor
    if _executor is None:
        registry = await get_default_registry()
        strategy = InProcessStrategy(registry)
        _executor = ToolExecutor(registry=registry, strategy=strategy)
    return _executor

async def execute_tool(
    tool_call: Dict[str, Any],
    _parent_event_id: str,
    _assistant_node_id: str,
) -> Dict[str, Any]:
    """
    Dispatch *tool_call* (a Chat-Completions-style dict) via the tool
    registry and return the tool's result.

    Parameters
    ----------
    tool_call : dict
        {
          "id": "…",
          "type": "function",
          "function": {
              "name": "weather",
              "arguments": "{\"location\": \"New York\"}"
          }
        }
    _parent_event_id : str
        The ID of the parent event
    _assistant_node_id : str
        The ID of the assistant node
        
    Returns
    -------
    Dict[str, Any]
        The result of executing the tool
    """
    name = tool_call["function"]["name"]
    args_text = tool_call["function"].get("arguments", "{}")
    
    try:
        args = json.loads(args_text)
    except json.JSONDecodeError:
        args = {"raw_text": args_text}
    
    # Get the executor (initialize if needed)
    executor = await _get_executor()
    
    # Create a tool call in the new format
    tc = ToolCall(
        id=tool_call.get("id", str(uuid4())),
        tool=name,
        arguments=args
    )
    
    # Execute the tool call
    results = await executor.execute([tc])
    if not results:
        raise RuntimeError(f"No results returned for tool {name}")
        
    result = results[0]
    if result.error:
        raise RuntimeError(f"Error executing {name}: {result.error}")
        
    return result.result