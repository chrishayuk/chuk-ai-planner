"""
Tool Execution Backend Interface

Defines the abstraction layer for pluggable tool execution backends.
This allows the planner to remain pure orchestration while delegating
execution to different runtimes (local, MCP, ACP, containers, etc.).

Pydantic-native: Uses Pydantic models instead of dictionaries.
Async-native: All methods are async.
"""

from typing import Protocol

from .models import ToolExecutionRequest, ToolExecutionResult


class ToolExecutionBackend(Protocol):
    """
    Protocol for tool execution backends.

    This interface defines the contract for executing tools at the boundary
    of the planner. The default implementation (ToolProcessorBackend) uses
    chuk-tool-processor to support:
    - Python functions (via register_fn_tool)
    - MCP tools (Notion, GitHub, etc.)
    - ACP agents
    - External containers
    - Built-in retries, caching, rate limiting

    Custom implementations could add support for:
    - Remote APIs
    - Distributed execution
    - Cloud functions
    - Etc.

    The planner remains pure orchestration and doesn't care about
    the execution details.

    Design Principles:
    - Pydantic-native: Uses Pydantic models, not dicts
    - Async-native: All I/O is async
    - Immutable: Request and result models are frozen
    """

    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        """
        Execute a tool and return the result.

        Args:
            request: Pydantic model containing tool execution request data

        Returns:
            Pydantic model containing tool execution result

        Raises:
            Exception: If execution fails catastrophically
        """
        ...
