"""Tests for processor.py - GraphAwareToolProcessor"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from chuk_ai_planner.processor import GraphAwareToolProcessor
from chuk_ai_planner.core.store.memory import InMemoryGraphStore
from chuk_tool_processor.models.tool_result import ToolResult


@pytest.fixture
def graph_store():
    """Create a graph store."""
    return InMemoryGraphStore()


@pytest.fixture
def session_id():
    """Create a session ID."""
    return str(uuid4())


@pytest.fixture
def processor(session_id, graph_store):
    """Create a processor instance."""
    return GraphAwareToolProcessor(session_id, graph_store)


@pytest.fixture
def mock_session_store(session_id):
    """Create a mock session store with a session."""
    from chuk_session_manager.models.session import Session
    from chuk_session_manager.storage import SessionStoreProvider

    store = MagicMock()
    session = Session()
    session.id = session_id
    session.events = []
    session.runs = []

    store.get = MagicMock(return_value=session)
    store.save = MagicMock()

    SessionStoreProvider.set_store(store)

    return store


class TestProcessorInitialization:
    """Test processor initialization."""

    def test_init_default_parameters(self, session_id, graph_store):
        """Test initialization with default parameters."""
        processor = GraphAwareToolProcessor(session_id, graph_store)

        assert processor.session_id == session_id
        assert processor.graph_store == graph_store
        assert processor.max_llm_retries == 2
        assert processor.enable_caching is True
        assert processor.enable_retries is True
        assert "Return ONLY a JSON block" in processor.llm_retry_prompt
        assert isinstance(processor.tool_registry, dict)
        assert len(processor.tool_registry) == 0

    def test_init_custom_parameters(self, session_id, graph_store):
        """Test initialization with custom parameters."""
        custom_prompt = "Please try again with a valid tool call"
        processor = GraphAwareToolProcessor(
            session_id,
            graph_store,
            max_llm_retries=5,
            llm_retry_prompt=custom_prompt,
            enable_caching=False,
            enable_retries=False,
        )

        assert processor.max_llm_retries == 5
        assert processor.llm_retry_prompt == custom_prompt
        assert processor.enable_caching is False
        assert processor.enable_retries is False

    def test_error_event_type_detection(self, session_id, graph_store):
        """Test that error event type is detected."""
        processor = GraphAwareToolProcessor(session_id, graph_store)

        # Should find ERROR, FAILURE, or EXCEPTION
        assert processor._error_event_type is not None


class TestToolRegistration:
    """Test tool registration."""

    def test_register_tool(self, processor):
        """Test registering a tool."""

        def my_tool(args):
            return {"result": "success"}

        processor.register_tool("my_tool", my_tool)

        assert "my_tool" in processor.tool_registry
        assert processor.tool_registry["my_tool"] == my_tool

    def test_register_multiple_tools(self, processor):
        """Test registering multiple tools."""

        def tool1(args):
            return "result1"

        def tool2(args):
            return "result2"

        processor.register_tool("tool1", tool1)
        processor.register_tool("tool2", tool2)

        assert len(processor.tool_registry) == 2
        assert processor.tool_registry["tool1"] == tool1
        assert processor.tool_registry["tool2"] == tool2


class TestGetToolExecutor:
    """Test _get_tool_executor method."""

    @pytest.mark.asyncio
    async def test_get_tool_executor_success(self, processor):
        """Test getting tool executor successfully."""
        executor = await processor._get_tool_executor()

        # Should return the mock executor from conftest
        assert executor is not None
        assert processor._executor is not None

    @pytest.mark.asyncio
    async def test_get_tool_executor_singleton(self, processor):
        """Test that executor is a singleton."""
        executor1 = await processor._get_tool_executor()
        executor2 = await processor._get_tool_executor()

        assert executor1 is executor2


class TestProcessLLMMessage:
    """Test process_llm_message method."""

    @pytest.mark.asyncio
    async def test_process_with_tool_calls(self, processor, mock_session_store):
        """Test processing message with tool calls."""

        # Register a tool
        async def test_tool(args):
            return {"output": "test result"}

        processor.register_tool("test_tool", test_tool)

        # LLM message with tool calls
        assistant_msg = {
            "role": "assistant",
            "content": "I'll use the tool",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": '{"input": "test"}'},
                }
            ],
        }

        llm_call_fn = AsyncMock()

        results = await processor.process_llm_message(assistant_msg, llm_call_fn)

        assert len(results) == 1
        assert results[0].tool == "test_tool"
        assert results[0].result == {"output": "test result"}
        assert results[0].error is None

    @pytest.mark.asyncio
    async def test_process_without_tool_calls_retries(
        self, processor, mock_session_store
    ):
        """Test processing message without tool calls triggers retry."""

        # First call has no tool calls
        assistant_msg_no_tools = {"role": "assistant", "content": "Just text"}

        # Second call has tool calls
        assistant_msg_with_tools = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_456",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": "{}"},
                }
            ],
        }

        async def test_tool(args):
            return {"result": "success"}

        processor.register_tool("test_tool", test_tool)

        # Mock LLM call to return message with tools on retry
        llm_call_fn = AsyncMock(return_value=assistant_msg_with_tools)

        results = await processor.process_llm_message(
            assistant_msg_no_tools, llm_call_fn
        )

        # Should have called LLM to retry
        assert llm_call_fn.called
        assert llm_call_fn.call_args[0][0] == processor.llm_retry_prompt

        # Should eventually get results
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_process_max_retries_exceeded(self, processor, mock_session_store):
        """Test that max retries raises error."""
        # Message with no tool calls
        assistant_msg = {"role": "assistant", "content": "No tools here"}

        # LLM always returns no tools
        llm_call_fn = AsyncMock(return_value=assistant_msg)

        with pytest.raises(RuntimeError, match="Max LLM retries exceeded"):
            await processor.process_llm_message(assistant_msg, llm_call_fn)

        # Should have tried max_llm_retries times
        assert llm_call_fn.call_count == processor.max_llm_retries

    @pytest.mark.asyncio
    async def test_process_session_not_found(self, processor):
        """Test error when session not found."""
        from chuk_session_manager.storage import SessionStoreProvider

        # Set up store that returns None
        store = MagicMock()
        store.get = MagicMock(return_value=None)
        SessionStoreProvider.set_store(store)

        assistant_msg = {"tool_calls": []}
        llm_call_fn = AsyncMock()

        with pytest.raises(RuntimeError, match="Session .* not found"):
            await processor.process_llm_message(assistant_msg, llm_call_fn)


class TestProcessSingleToolCall:
    """Test _process_single_tool_call method."""

    @pytest.mark.asyncio
    async def test_process_tool_call_success(self, processor, mock_session_store):
        """Test processing a successful tool call."""

        async def success_tool(args):
            return {"result": f"processed {args['input']}"}

        processor.register_tool("success_tool", success_tool)

        tool_call = {
            "id": "call_789",
            "type": "function",
            "function": {"name": "success_tool", "arguments": '{"input": "data"}'},
        }

        result = await processor._process_single_tool_call(tool_call, "parent_id", None)

        assert result.tool == "success_tool"
        assert result.result == {"result": "processed data"}
        assert result.error is None

    @pytest.mark.asyncio
    async def test_process_tool_call_unknown_tool(self, processor, mock_session_store):
        """Test processing with unknown tool."""
        tool_call = {
            "id": "call_unknown",
            "type": "function",
            "function": {"name": "nonexistent_tool", "arguments": "{}"},
        }

        with pytest.raises(ValueError, match="Unknown tool: nonexistent_tool"):
            await processor._process_single_tool_call(tool_call, "parent_id", None)

    @pytest.mark.asyncio
    async def test_process_tool_call_with_error(self, processor, mock_session_store):
        """Test processing tool that raises error."""

        async def failing_tool(args):
            raise RuntimeError("Tool failed!")

        processor.register_tool("failing_tool", failing_tool)

        tool_call = {
            "id": "call_fail",
            "type": "function",
            "function": {"name": "failing_tool", "arguments": "{}"},
        }

        result = await processor._process_single_tool_call(tool_call, "parent_id", None)

        assert result.tool == "failing_tool"
        assert result.result is None
        assert "Tool failed!" in result.error

    @pytest.mark.asyncio
    async def test_process_tool_call_with_caching(self, processor, mock_session_store):
        """Test that caching works."""
        call_count = 0

        async def counting_tool(args):
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        processor.register_tool("counting_tool", counting_tool)

        tool_call = {
            "id": "call_cache",
            "type": "function",
            "function": {"name": "counting_tool", "arguments": '{"param": "value"}'},
        }

        # First call
        result1 = await processor._process_single_tool_call(
            tool_call, "parent_id", None
        )
        assert result1.result == {"count": 1}

        # Second call with same args should use cache
        result2 = await processor._process_single_tool_call(
            tool_call, "parent_id", None
        )
        assert result2.result == {"count": 1}  # Same result, not incremented
        assert call_count == 1  # Tool only called once

    @pytest.mark.asyncio
    async def test_process_tool_call_caching_disabled(
        self, session_id, graph_store, mock_session_store
    ):
        """Test that caching can be disabled."""
        processor = GraphAwareToolProcessor(
            session_id, graph_store, enable_caching=False
        )

        call_count = 0

        async def counting_tool(args):
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        processor.register_tool("counting_tool", counting_tool)

        tool_call = {
            "id": "call_no_cache",
            "type": "function",
            "function": {"name": "counting_tool", "arguments": '{"param": "value"}'},
        }

        # First call
        result1 = await processor._process_single_tool_call(
            tool_call, "parent_id", None
        )
        assert result1.result == {"count": 1}

        # Second call should NOT use cache
        result2 = await processor._process_single_tool_call(
            tool_call, "parent_id", None
        )
        assert result2.result == {"count": 2}  # Incremented
        assert call_count == 2  # Tool called twice

    @pytest.mark.asyncio
    async def test_process_tool_call_invalid_json_args(
        self, processor, mock_session_store
    ):
        """Test handling of invalid JSON arguments."""

        async def test_tool(args):
            # Should receive raw_text key when JSON is invalid
            return {"received": args}

        processor.register_tool("test_tool", test_tool)

        tool_call = {
            "id": "call_bad_json",
            "type": "function",
            "function": {
                "name": "test_tool",
                "arguments": "not valid json {{{",  # Invalid JSON
            },
        }

        result = await processor._process_single_tool_call(tool_call, "parent_id", None)

        assert result.error is None
        assert "raw_text" in result.result["received"]

    @pytest.mark.asyncio
    async def test_process_tool_call_with_assistant_node(
        self, processor, mock_session_store
    ):
        """Test processing with assistant_node_id creates graph nodes."""

        async def test_tool(args):
            return {"result": "success"}

        processor.register_tool("test_tool", test_tool)

        # Mock the node manager methods
        processor.node_mgr.create_tool_call_node = AsyncMock(
            return_value=MagicMock(id="tool_node_123")
        )
        processor.node_mgr.create_task_run_node = AsyncMock()

        tool_call = {
            "id": "call_with_node",
            "type": "function",
            "function": {"name": "test_tool", "arguments": "{}"},
        }

        await processor._process_single_tool_call(
            tool_call, "parent_id", "assistant_node_123"
        )

        # Should have created graph nodes
        processor.node_mgr.create_tool_call_node.assert_called_once()
        processor.node_mgr.create_task_run_node.assert_called_once()


class TestCreateChildEvent:
    """Test _create_child_event method."""

    def test_create_child_event(self, processor, mock_session_store):
        """Test creating a child event."""
        from chuk_session_manager.models.event_type import EventType

        message = {"key": "value"}
        parent_id = "parent_123"

        event = processor._create_child_event(EventType.SUMMARY, message, parent_id)

        assert event.message == message
        assert event.type == EventType.SUMMARY
        assert event.metadata["parent_event_id"] == parent_id


class TestProcessPlan:
    """Test process_plan method."""

    @pytest.mark.asyncio
    async def test_process_plan_basic(self, processor, mock_session_store, graph_store):
        """Test basic plan processing."""
        from chuk_ai_planner.core.graph import PlanNode, PlanStep, ParentChildEdge

        # Create a plan with one step
        plan = PlanNode(id="plan_1", title="Test Plan", description="Test")
        step = PlanStep(id="step_1", description="Step 1", index="1")

        await graph_store.add_node(plan)
        await graph_store.add_node(step)
        await graph_store.add_edge(ParentChildEdge(src=plan.id, dst=step.id))

        # Mock the plan executor methods
        processor.plan_executor.get_plan_steps = AsyncMock(return_value=[step.id])
        processor.plan_executor.determine_execution_order = AsyncMock(
            return_value=[[step.id]]
        )
        processor.plan_executor.execute_step = AsyncMock(return_value=[])

        llm_call_fn = AsyncMock()

        results = await processor.process_plan(
            plan.id, "assistant_id", llm_call_fn, on_step=None
        )

        assert isinstance(results, list)
        processor.plan_executor.execute_step.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_plan_no_steps(self, processor, mock_session_store):
        """Test error when plan has no steps."""
        # Mock get_plan_steps to return empty list
        processor.plan_executor.get_plan_steps = AsyncMock(return_value=[])

        llm_call_fn = AsyncMock()

        with pytest.raises(ValueError, match="No steps found for plan"):
            await processor.process_plan("plan_empty", "assistant_id", llm_call_fn)

    @pytest.mark.asyncio
    async def test_process_plan_with_on_step_callback(
        self, processor, mock_session_store, graph_store
    ):
        """Test plan processing with on_step callback."""
        from chuk_ai_planner.core.graph import PlanNode, PlanStep, ParentChildEdge

        # Create a plan with multiple steps
        plan = PlanNode(id="plan_2", title="Multi-step Plan", description="Test")
        step1 = PlanStep(id="step_1", description="Step 1", index="1")
        step2 = PlanStep(id="step_2", description="Step 2", index="2")

        await graph_store.add_node(plan)
        await graph_store.add_node(step1)
        await graph_store.add_node(step2)
        await graph_store.add_edge(ParentChildEdge(src=plan.id, dst=step1.id))
        await graph_store.add_edge(ParentChildEdge(src=plan.id, dst=step2.id))

        # Mock the plan executor
        processor.plan_executor.get_plan_steps = AsyncMock(
            return_value=[step1.id, step2.id]
        )
        processor.plan_executor.determine_execution_order = AsyncMock(
            return_value=[[step1.id], [step2.id]]
        )
        processor.plan_executor.execute_step = AsyncMock(return_value=[])

        # Callback that stops after first step
        def on_step_callback(step_id, results):
            if step_id == step1.id:
                return False  # Stop execution
            return True

        llm_call_fn = AsyncMock()

        await processor.process_plan(
            plan.id, "assistant_id", llm_call_fn, on_step=on_step_callback
        )

        # Should only execute first step
        assert processor.plan_executor.execute_step.call_count == 1

    @pytest.mark.asyncio
    async def test_process_plan_session_not_found(self, processor):
        """Test error when session not found during plan processing."""
        from chuk_session_manager.storage import SessionStoreProvider

        # Set up store that returns None
        store = MagicMock()
        store.get = MagicMock(return_value=None)
        SessionStoreProvider.set_store(store)

        llm_call_fn = AsyncMock()

        with pytest.raises(RuntimeError, match="Session .* not found"):
            await processor.process_plan("plan_id", "assistant_id", llm_call_fn)

    @pytest.mark.asyncio
    async def test_process_plan_with_results(
        self, processor, mock_session_store, graph_store
    ):
        """Test that plan processing returns accumulated results."""
        from chuk_ai_planner.core.graph import PlanNode, PlanStep, ParentChildEdge

        plan = PlanNode(id="plan_3", title="Results Plan", description="Test")
        step = PlanStep(id="step_1", description="Step 1", index="1")

        await graph_store.add_node(plan)
        await graph_store.add_node(step)
        await graph_store.add_edge(ParentChildEdge(src=plan.id, dst=step.id))

        # Mock execution to return some results
        mock_results = [
            ToolResult(id="r1", tool="tool1", result={"data": "result1"}, error=None)
        ]
        processor.plan_executor.get_plan_steps = AsyncMock(return_value=[step.id])
        processor.plan_executor.determine_execution_order = AsyncMock(
            return_value=[[step.id]]
        )
        processor.plan_executor.execute_step = AsyncMock(return_value=mock_results)

        llm_call_fn = AsyncMock()

        results = await processor.process_plan(plan.id, "assistant_id", llm_call_fn)

        assert len(results) == 1
        assert results[0].tool == "tool1"
        assert results[0].result == {"data": "result1"}
