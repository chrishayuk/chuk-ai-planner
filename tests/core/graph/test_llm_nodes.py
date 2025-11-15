"""
Tests for LLM extension nodes.

These tests cover the domain-specific LLM nodes that extend the core graph.
"""

import pytest
from pydantic import ValidationError

from chuk_ai_planner.extensions.llm.nodes import (
    UserMessage,
    AssistantMessage,
    SystemMessage,
)


class TestUserMessage:
    """Test UserMessage LLM extension node."""

    def test_create_with_required_fields(self):
        """Should create UserMessage with just content."""
        msg = UserMessage(content="Hello, assistant!")

        assert msg.kind == "user_message"
        assert msg.content == "Hello, assistant!"
        assert msg.role == "user"
        assert msg.user_id is None
        assert msg.conversation_id is None
        assert msg.id is not None
        assert msg.ts is not None

    def test_create_with_all_fields(self):
        """Should create UserMessage with all optional fields."""
        msg = UserMessage(
            content="What's the weather?",
            role="user",
            user_id="user123",
            conversation_id="conv456",
        )

        assert msg.content == "What's the weather?"
        assert msg.user_id == "user123"
        assert msg.conversation_id == "conv456"

    def test_role_defaults_to_user(self):
        """Should default role to 'user'."""
        msg = UserMessage(content="Hello")
        assert msg.role == "user"

    def test_custom_role(self):
        """Should allow custom role."""
        msg = UserMessage(content="Hello", role="customer")
        assert msg.role == "customer"

    def test_immutable(self):
        """Should be immutable (frozen)."""
        msg = UserMessage(content="Hello")

        with pytest.raises(ValidationError):
            msg.content = "Goodbye"

    def test_auto_generated_fields(self):
        """Should auto-generate id and ts."""
        msg = UserMessage(content="Hello")

        assert msg.id is not None
        assert len(msg.id) > 0
        assert msg.ts is not None

    def test_repr(self):
        """Should have useful repr."""
        msg = UserMessage(content="Hello")
        repr_str = repr(msg)

        assert "user_message" in repr_str
        assert msg.id[:8] in repr_str


class TestAssistantMessage:
    """Test AssistantMessage LLM extension node."""

    def test_create_with_required_fields(self):
        """Should create AssistantMessage with just content."""
        msg = AssistantMessage(content="I can help with that!")

        assert msg.kind == "assistant_message"
        assert msg.content == "I can help with that!"
        assert msg.role == "assistant"
        assert msg.tool_calls == []
        assert msg.model is None
        assert msg.finish_reason is None

    def test_create_with_tool_calls(self):
        """Should create AssistantMessage with tool calls."""
        tool_calls = [
            {"id": "call_123", "name": "get_weather", "arguments": {"city": "New York"}}
        ]
        msg = AssistantMessage(
            content="Let me check the weather.", tool_calls=tool_calls
        )

        assert msg.tool_calls == tool_calls
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"

    def test_create_with_model_info(self):
        """Should create AssistantMessage with model info."""
        msg = AssistantMessage(content="Response", model="gpt-4", finish_reason="stop")

        assert msg.model == "gpt-4"
        assert msg.finish_reason == "stop"

    def test_role_defaults_to_assistant(self):
        """Should default role to 'assistant'."""
        msg = AssistantMessage(content="Hello")
        assert msg.role == "assistant"

    def test_tool_calls_defaults_to_empty_list(self):
        """Should default tool_calls to empty list."""
        msg = AssistantMessage(content="Hello")
        assert msg.tool_calls == []
        assert isinstance(msg.tool_calls, list)

    def test_immutable(self):
        """Should be immutable (frozen)."""
        msg = AssistantMessage(content="Hello")

        with pytest.raises(ValidationError):
            msg.content = "Goodbye"

    def test_repr(self):
        """Should have useful repr."""
        msg = AssistantMessage(content="Hello")
        repr_str = repr(msg)

        assert "assistant_message" in repr_str
        assert msg.id[:8] in repr_str


class TestSystemMessage:
    """Test SystemMessage LLM extension node."""

    def test_create_with_required_fields(self):
        """Should create SystemMessage with just content."""
        msg = SystemMessage(content="You are a helpful assistant.")

        assert msg.kind == "system_message"
        assert msg.content == "You are a helpful assistant."
        assert msg.role == "system"

    def test_role_defaults_to_system(self):
        """Should default role to 'system'."""
        msg = SystemMessage(content="You are helpful.")
        assert msg.role == "system"

    def test_custom_role(self):
        """Should allow custom role."""
        msg = SystemMessage(content="Context", role="context")
        assert msg.role == "context"

    def test_immutable(self):
        """Should be immutable (frozen)."""
        msg = SystemMessage(content="Hello")

        with pytest.raises(ValidationError):
            msg.content = "Goodbye"

    def test_repr(self):
        """Should have useful repr."""
        msg = SystemMessage(content="You are helpful.")
        repr_str = repr(msg)

        assert "system_message" in repr_str
        assert msg.id[:8] in repr_str


class TestLLMNodeComparisons:
    """Test equality and hashing for LLM nodes."""

    def test_user_message_equality(self):
        """Should compare UserMessage by id."""
        msg1 = UserMessage(content="Hello")
        msg2 = UserMessage(content="Hello")

        # Different instances have different IDs
        assert msg1 != msg2

        # Same instance equals itself
        assert msg1 == msg1

    def test_assistant_message_in_set(self):
        """Should be usable in sets."""
        msg1 = AssistantMessage(content="Response 1")
        msg2 = AssistantMessage(content="Response 2")
        msg3 = msg1

        messages = {msg1, msg2, msg3}
        assert len(messages) == 2  # msg1 and msg3 are the same

    def test_system_message_as_dict_key(self):
        """Should be usable as dict keys."""
        msg1 = SystemMessage(content="Context 1")
        msg2 = SystemMessage(content="Context 2")

        message_map = {msg1: "first", msg2: "second"}
        assert message_map[msg1] == "first"
        assert message_map[msg2] == "second"


class TestLLMNodeIntegration:
    """Test LLM nodes work with the graph."""

    def test_user_message_has_graph_node_interface(self):
        """Should have all GraphNode fields."""
        msg = UserMessage(content="Hello")

        # GraphNode fields
        assert hasattr(msg, "id")
        assert hasattr(msg, "kind")
        assert hasattr(msg, "ts")
        assert hasattr(msg, "metadata")

    def test_assistant_message_metadata(self):
        """Should support metadata dict."""
        msg = AssistantMessage(
            content="Response", metadata={"source": "api", "version": "1.0"}
        )

        assert msg.metadata["source"] == "api"
        assert msg.metadata["version"] == "1.0"

    def test_system_message_model_copy(self):
        """Should support model_copy for updates."""
        msg = SystemMessage(content="You are helpful.")

        updated = msg.model_copy(update={"content": "You are very helpful."})

        assert updated.content == "You are very helpful."
        assert msg.content == "You are helpful."  # Original unchanged
        assert updated.id == msg.id  # Same ID


class TestLLMNodeValidation:
    """Test validation for LLM nodes."""

    def test_user_message_requires_content(self):
        """Should require content field."""
        with pytest.raises(ValidationError):
            UserMessage()

    def test_assistant_message_requires_content(self):
        """Should require content field."""
        with pytest.raises(ValidationError):
            AssistantMessage()

    def test_system_message_requires_content(self):
        """Should require content field."""
        with pytest.raises(ValidationError):
            SystemMessage()

    def test_assistant_message_tool_calls_must_be_list(self):
        """Should validate tool_calls is a list."""
        msg = AssistantMessage(
            content="Hello", tool_calls=[{"id": "1", "name": "tool"}]
        )
        assert isinstance(msg.tool_calls, list)
