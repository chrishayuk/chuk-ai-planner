# chuk_ai_planner/extensions/llm/types.py
"""
LLM-specific type definitions.

Extension types for chat/LLM workflows.
Kept separate from core types for modularity.
"""

from enum import Enum

__all__ = ["LLMNodeType"]


class LLMNodeType(str, Enum):
    """
    LLM-specific node types.

    Extension of the core graph for chat/LLM workflows.
    Kept separate from core NodeType for modularity - allows
    projects to use LLM features without polluting core types.

    Future extensions (audio, video, etc.) should follow this pattern.
    """

    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    SYSTEM_MESSAGE = "system_message"
