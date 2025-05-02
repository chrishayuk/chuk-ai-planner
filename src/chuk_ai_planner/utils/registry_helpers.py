# ── src/chuk_ai_planner/utils/registry_helpers.py ──────────────────────────
"""
Utility for running a tool that is registered in
`chuk_tool_processor.default_registry`.

`execute_tool(tool_call, parent_event_id, assistant_node_id)` is meant to be
passed straight into `PlanExecutor.execute_step` (or any other place that
expects the `process_tool_call` signature).

The helper transparently deals with:

  • ValidatedTool *classes*    – instantiates, then `.arun(**kwargs)`
  • ValidatedTool *instances*  – calls        `.arun(**kwargs)`
  • Plain async callables      – await `fn(args_dict)`
  • Plain sync  callables      – call  `fn(args_dict)`
"""
from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Dict

from chuk_tool_processor.registry import default_registry
from chuk_tool_processor.models.validated_tool import ValidatedTool


async def execute_tool(
    tool_call: Dict[str, Any],
    _parent_event_id: str,
    _assistant_node_id: str,
) -> Dict[str, Any]:
    """
    Dispatch *tool_call* (a Chat-Completions-style dict) via the global tool
    registry and return the tool’s result.

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
    """
    name = tool_call["function"]["name"]
    args: Dict[str, Any] = json.loads(tool_call["function"].get("arguments", "{}"))

    fn = default_registry.get_tool(name)  # raises KeyError if not found

    # ── ValidatedTool (class or instance) ────────────────────────────
    if inspect.isclass(fn) and issubclass(fn, ValidatedTool):
        return await fn().arun(**args)          # create → async run
    if isinstance(fn, ValidatedTool):
        return await fn.arun(**args)            # already an instance

    # ── plain python callable (async or sync) ────────────────────────
    if asyncio.iscoroutinefunction(fn):
        return await fn(args)
    return fn(args)  # sync

