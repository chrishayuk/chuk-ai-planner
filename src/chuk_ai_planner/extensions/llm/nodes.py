# chuk_ai_planner/extensions/llm/nodes.py
"""
LLM-specific graph nodes.

These nodes extend the core graph for LLM/chat-specific use cases.
Projects that don't use LLMs don't need to import these.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import Field

from chuk_ai_planner.core.graph.nodes.base import GraphNode
from .types import LLMNodeType

__all__ = [
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
]


class UserMessage(GraphNode):
    """
    A message from the user in an LLM conversation.

    This is an LLM-specific extension of the core graph.
    Use this for chat-based workflows.

    Examples
    --------
    >>> user_msg = UserMessage(
    ...     content="What's the weather in New York?",
    ...     role="user"
    ... )
    """

    kind: Literal[LLMNodeType.USER_MESSAGE] = LLMNodeType.USER_MESSAGE

    # LLM-specific fields
    content: str
    role: str = "user"

    # Optional metadata
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


class AssistantMessage(GraphNode):
    """
    A message from the LLM assistant.

    This is an LLM-specific extension of the core graph.
    Use this for chat-based workflows.

    Examples
    --------
    >>> assistant_msg = AssistantMessage(
    ...     content="The weather in New York is...",
    ...     role="assistant",
    ...     tool_calls=[]
    ... )
    """

    kind: Literal[LLMNodeType.ASSISTANT_MESSAGE] = LLMNodeType.ASSISTANT_MESSAGE

    # LLM-specific fields
    content: str
    role: str = "assistant"

    # Tool calls made by assistant
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)

    # Model info
    model: Optional[str] = None
    finish_reason: Optional[str] = None


class SystemMessage(GraphNode):
    """
    A system message for LLM context.

    Examples
    --------
    >>> system_msg = SystemMessage(
    ...     content="You are a helpful assistant.",
    ...     role="system"
    ... )
    """

    kind: Literal[LLMNodeType.SYSTEM_MESSAGE] = LLMNodeType.SYSTEM_MESSAGE

    content: str
    role: str = "system"
