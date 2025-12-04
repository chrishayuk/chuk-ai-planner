"""Tests for utils/registry_helpers.py"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chuk_ai_planner.utils.registry_helpers import execute_tool, _get_executor


class TestExecuteTool:
    """Test the execute_tool function."""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution."""
        # Create a mock executor
        mock_result = MagicMock()
        mock_result.result = {"output": "success"}
        mock_result.error = None

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=[mock_result])

        # Patch _get_executor to return our mock
        with patch(
            "chuk_ai_planner.utils.registry_helpers._get_executor",
            return_value=mock_executor,
        ):
            tool_call = {
                "id": "call_123",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"param": "value"}'},
            }

            result = await execute_tool(tool_call)

            assert result == {"output": "success"}
            mock_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_with_invalid_json(self):
        """Test tool execution with invalid JSON arguments."""
        # Create a mock executor
        mock_result = MagicMock()
        mock_result.result = {"output": "processed"}
        mock_result.error = None

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=[mock_result])

        with patch(
            "chuk_ai_planner.utils.registry_helpers._get_executor",
            return_value=mock_executor,
        ):
            tool_call = {
                "id": "call_456",
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "arguments": "not valid json",  # Invalid JSON
                },
            }

            result = await execute_tool(tool_call)

            # Should still work, wrapping in raw_text
            assert result == {"output": "processed"}
            mock_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_tool_with_error(self):
        """Test tool execution that returns an error."""
        # Create a mock executor with an error
        mock_result = MagicMock()
        mock_result.result = None
        mock_result.error = "Something went wrong"

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=[mock_result])

        with patch(
            "chuk_ai_planner.utils.registry_helpers._get_executor",
            return_value=mock_executor,
        ):
            tool_call = {
                "id": "call_789",
                "type": "function",
                "function": {"name": "failing_tool", "arguments": "{}"},
            }

            with pytest.raises(RuntimeError, match="Error executing failing_tool"):
                await execute_tool(tool_call)

    @pytest.mark.asyncio
    async def test_execute_tool_no_results(self):
        """Test tool execution that returns no results."""
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=[])

        with patch(
            "chuk_ai_planner.utils.registry_helpers._get_executor",
            return_value=mock_executor,
        ):
            tool_call = {
                "id": "call_999",
                "type": "function",
                "function": {"name": "empty_tool", "arguments": "{}"},
            }

            with pytest.raises(RuntimeError, match="No results returned for tool"):
                await execute_tool(tool_call)

    @pytest.mark.asyncio
    async def test_execute_tool_without_id(self):
        """Test tool execution without an explicit ID."""
        mock_result = MagicMock()
        mock_result.result = {"status": "ok"}
        mock_result.error = None

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=[mock_result])

        with patch(
            "chuk_ai_planner.utils.registry_helpers._get_executor",
            return_value=mock_executor,
        ):
            tool_call = {
                "type": "function",
                "function": {"name": "no_id_tool", "arguments": "{}"},
            }

            result = await execute_tool(tool_call)

            assert result == {"status": "ok"}
            # Should have generated a UUID for the call
            call_args = mock_executor.execute.call_args[0][0][0]
            assert call_args.id is not None

    @pytest.mark.asyncio
    async def test_execute_tool_with_parent_and_assistant_ids(self):
        """Test tool execution with parent event ID and assistant node ID."""
        mock_result = MagicMock()
        mock_result.result = {"data": "test"}
        mock_result.error = None

        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(return_value=[mock_result])

        with patch(
            "chuk_ai_planner.utils.registry_helpers._get_executor",
            return_value=mock_executor,
        ):
            tool_call = {
                "id": "call_123",
                "type": "function",
                "function": {"name": "tool_with_context", "arguments": "{}"},
            }

            result = await execute_tool(
                tool_call, _parent_event_id="event_456", _assistant_node_id="asst_789"
            )

            assert result == {"data": "test"}

    @pytest.mark.asyncio
    async def test_get_executor_singleton(self):
        """Test that _get_executor returns the same instance."""
        # Reset the global executor
        import chuk_ai_planner.utils.registry_helpers as module

        module._executor = None

        mock_registry = MagicMock()
        mock_strategy = MagicMock()
        mock_executor = MagicMock()

        with (
            patch(
                "chuk_ai_planner.utils.registry_helpers.get_default_registry",
                return_value=mock_registry,
            ),
            patch(
                "chuk_ai_planner.utils.registry_helpers.InProcessStrategy",
                return_value=mock_strategy,
            ),
            patch(
                "chuk_ai_planner.utils.registry_helpers.ToolExecutor",
                return_value=mock_executor,
            ),
        ):
            # First call should create executor
            executor1 = await _get_executor()
            assert executor1 == mock_executor

            # Second call should return the same instance
            executor2 = await _get_executor()
            assert executor2 == executor1

        # Reset for other tests
        module._executor = None
