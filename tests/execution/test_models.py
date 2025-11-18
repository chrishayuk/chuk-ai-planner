"""Tests for execution models."""

import pytest
from pydantic import ValidationError

from chuk_ai_planner.execution.models import (
    ToolExecutionRequest,
    ToolExecutionResult,
)


class TestToolExecutionRequest:
    """Tests for ToolExecutionRequest model."""

    def test_create_basic_request(self):
        """Test creating a basic request."""
        request = ToolExecutionRequest(
            tool_name="test_tool",
            args={"key": "value"},
            step_id="step-123",
        )

        assert request.tool_name == "test_tool"
        assert request.args == {"key": "value"}
        assert request.step_id == "step-123"
        assert request.session_id is None

    def test_create_with_session_id(self):
        """Test creating request with session ID."""
        request = ToolExecutionRequest(
            tool_name="test_tool",
            args={},
            step_id="step-123",
            session_id="session-456",
        )

        assert request.session_id == "session-456"

    def test_empty_args_default(self):
        """Test that args defaults to empty dict."""
        request = ToolExecutionRequest(
            tool_name="test_tool",
            step_id="step-123",
        )

        assert request.args == {}

    def test_immutable(self):
        """Test that request is immutable."""
        request = ToolExecutionRequest(
            tool_name="test_tool",
            args={},
            step_id="step-123",
        )

        with pytest.raises(ValidationError):
            request.tool_name = "new_name"  # type: ignore

    def test_model_config_frozen(self):
        """Test that model config is frozen."""
        assert ToolExecutionRequest.model_config["frozen"] is True


class TestToolExecutionResult:
    """Tests for ToolExecutionResult model."""

    def test_create_success_result(self):
        """Test creating a successful result."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            result={"data": "value"},
            error=None,
            duration=0.123,
            cached=False,
        )

        assert result.tool_name == "test_tool"
        assert result.result == {"data": "value"}
        assert result.error is None
        assert result.duration == 0.123
        assert result.cached is False

    def test_create_error_result(self):
        """Test creating an error result."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            result=None,
            error="Something went wrong",
            duration=0.050,
            cached=False,
        )

        assert result.error == "Something went wrong"
        assert result.result is None

    def test_success_property_true(self):
        """Test success property returns True when no error."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            result={"data": "value"},
            error=None,
        )

        assert result.success is True

    def test_success_property_false(self):
        """Test success property returns False when error exists."""
        result = ToolExecutionResult(
            tool_name="test_tool",
            result=None,
            error="Failed",
        )

        assert result.success is False

    def test_cached_default_false(self):
        """Test that cached defaults to False."""
        result = ToolExecutionResult(
            tool_name="test_tool",
        )

        assert result.cached is False

    def test_optional_fields_default_none(self):
        """Test that optional fields default to None."""
        result = ToolExecutionResult(
            tool_name="test_tool",
        )

        assert result.result is None
        assert result.error is None
        assert result.duration is None

    def test_immutable(self):
        """Test that result is immutable."""
        result = ToolExecutionResult(
            tool_name="test_tool",
        )

        with pytest.raises(ValidationError):
            result.tool_name = "new_name"  # type: ignore

    def test_model_config_frozen(self):
        """Test that model config is frozen."""
        assert ToolExecutionResult.model_config["frozen"] is True

    def test_duration_validation(self):
        """Test that duration must be >= 0."""
        # Valid
        result = ToolExecutionResult(
            tool_name="test_tool",
            duration=0.0,
        )
        assert result.duration == 0.0

        # Invalid - negative duration
        with pytest.raises(ValidationError):
            ToolExecutionResult(
                tool_name="test_tool",
                duration=-1.0,
            )
