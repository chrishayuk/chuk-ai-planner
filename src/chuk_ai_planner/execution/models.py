"""
Pydantic models for tool execution backends.

These models provide type-safe, validated data structures for tool execution,
replacing dictionary-based approaches with Pydantic-native types.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class ToolExecutionRequest(BaseModel):
    """
    Request to execute a tool.

    Pydantic-native replacement for dictionary-based tool execution requests.
    """

    model_config = ConfigDict(frozen=True)  # Immutable

    tool_name: str = Field(..., description="Name of the tool to execute")
    args: dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )
    step_id: str = Field(..., description="Unique ID of the step requesting execution")
    session_id: Optional[str] = Field(
        None, description="Optional session ID for context/caching"
    )


class ToolExecutionResult(BaseModel):
    """
    Result of tool execution.

    Pydantic-native replacement for dictionary-based tool execution results.
    """

    model_config = ConfigDict(frozen=True)  # Immutable

    tool_name: str = Field(..., description="Name of the tool that was executed")
    result: Optional[Any] = Field(None, description="Tool execution result")
    error: Optional[str] = Field(None, description="Error message if failed")
    duration: Optional[float] = Field(
        None, description="Execution time in seconds", ge=0
    )
    cached: bool = Field(False, description="Whether result came from cache")

    @property
    def success(self) -> bool:
        """Whether the execution was successful."""
        return self.error is None
