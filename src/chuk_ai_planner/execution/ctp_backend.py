"""
chuk-tool-processor Backend

Executes tools via chuk-tool-processor, enabling:
- MCP tool support (Model Context Protocol)
- ACP tool support (Agent Communication Protocol)
- Built-in caching, retries, rate limiting
- Container-based execution
- Unified tool registry

This makes the planner the universal agent runtime for the CHUK ecosystem.
"""

import json
import time
from typing import Any, Optional

from .models import ToolExecutionRequest, ToolExecutionResult


class ToolProcessorBackend:
    """
    Planner → chuk-tool-processor adapter.

    This backend uses chuk-tool-processor to execute tools, which means:
    1. Tools can be local (@tool decorator)
    2. Tools can be MCP servers (setup_mcp_stdio, setup_mcp_sse, etc.)
    3. Tools can be ACP agents
    4. Tools can run in containers
    5. Built-in reliability (retries, circuit breakers, rate limits)

    The planner becomes pure orchestration.
    """

    def __init__(
        self,
        processor: Any,  # ToolProcessor, but avoid import at module level
        *,
        namespace: Optional[str] = None,
    ):
        """
        Initialize the ToolProcessor backend.

        Args:
            processor: The ToolProcessor instance to use for execution.
                       Should be pre-configured with tools via @tool decorator,
                       setup_mcp_*, or other registration methods.
            namespace: Optional namespace prefix for tool names.
                      E.g., "mcp" → tool_name becomes "mcp:search"
        """
        # Validate processor type at runtime
        if not hasattr(processor, "process"):
            raise TypeError(
                f"processor must have a 'process' method, got {type(processor)}"
            )

        self._processor = processor
        self._namespace = namespace

    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        """
        Execute a tool via chuk-tool-processor.

        We create a minimal OpenAI-style tool_calls payload so CTP
        can parse and execute it using its existing pipeline.

        Args:
            request: Pydantic model containing execution request

        Returns:
            Pydantic model containing execution result

        Raises:
            RuntimeError: If no result returned or execution fails
        """
        start_time = time.perf_counter()

        # Apply namespace if configured
        qualified_name = (
            f"{self._namespace}:{request.tool_name}"
            if self._namespace
            else request.tool_name
        )

        # Create OpenAI-style tool_calls payload
        # CTP expects this format and knows how to parse it
        llm_output: dict[str, Any] = {
            "tool_calls": [
                {
                    "id": request.step_id,
                    "type": "function",
                    "function": {
                        "name": qualified_name,
                        "arguments": json.dumps(request.args),
                    },
                }
            ],
        }

        # Add session metadata if provided
        if request.session_id:
            llm_output["session_id"] = request.session_id

        try:
            # Process through CTP (async-native)
            results = await self._processor.process(llm_output)

            if not results:
                raise RuntimeError(
                    f"No result returned for step {request.step_id} ({request.tool_name})"
                )

            first = results[0]
            duration = time.perf_counter() - start_time

            # Return Pydantic model
            return ToolExecutionResult(
                tool_name=request.tool_name,
                result=first.result if not first.error else None,
                error=first.error,
                duration=first.duration or duration,
                cached=first.cached or False,
            )

        except Exception as e:
            duration = time.perf_counter() - start_time
            return ToolExecutionResult(
                tool_name=request.tool_name,
                result=None,
                error=str(e),
                duration=duration,
                cached=False,
            )
