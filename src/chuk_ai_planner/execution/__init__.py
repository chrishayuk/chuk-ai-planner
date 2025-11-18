"""Execution backends for chuk-ai-planner.

Universal execution via chuk-tool-processor:
- Python functions (via register_fn_tool)
- MCP tools (Notion, GitHub, etc.)
- ACP agents
- Container execution
- Built-in retries, caching, rate limiting

Pydantic-native: Uses Pydantic models instead of dictionaries.
Async-native: All execution is async.
"""

from .interfaces import ToolExecutionBackend
from .models import ToolExecutionRequest, ToolExecutionResult

# Lazy import for ToolProcessorBackend
__all__ = [
    "ToolExecutionBackend",
    "ToolProcessorBackend",
    "ToolExecutionRequest",
    "ToolExecutionResult",
]


def __getattr__(name):
    """Lazy import for ToolProcessorBackend."""
    if name == "ToolProcessorBackend":
        from .ctp_backend import ToolProcessorBackend

        return ToolProcessorBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
