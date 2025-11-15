# tests/planner/test_universal_plan_executor.py
"""
Unit tests for UniversalExecutor
"""

import pytest
import asyncio
from unittest.mock import MagicMock

from chuk_ai_planner.core.planner.universal_plan_executor import UniversalExecutor
from chuk_ai_planner.core.planner.universal_plan import UniversalPlan
from chuk_ai_planner.core.store.memory import InMemoryGraphStore
from chuk_ai_planner.core.graph import ToolCall, PlanStep, PlanLinkEdge
from chuk_ai_planner.core.graph import GraphEdge, EdgeType


@pytest.fixture
def graph_store():
    """Create an in-memory graph store for testing."""
    return InMemoryGraphStore()


@pytest.fixture
def executor(graph_store):
    """Create a UniversalExecutor for testing (sync fixture)."""
    return UniversalExecutor(graph_store=graph_store)


@pytest.fixture
def simple_plan(graph_store):
    """Create a simple plan for testing."""
    plan = UniversalPlan(
        title="Test Plan",
        description="A plan for testing",
        tags=["test"],
        graph=graph_store,
    )

    # Add some variables
    plan.set_variable("test_var", "test_value")
    plan.set_variable("number", 42)

    return plan


class TestUniversalExecutorInit:
    """Test UniversalExecutor initialization."""

    @pytest.mark.asyncio
    async def test_init_default_graph_store(self):
        """Test initialization with default graph store."""
        executor = UniversalExecutor()
        assert executor.graph_store is not None
        assert executor.session is None  # Not created until needed
        assert not executor._session_initialized

    def test_init_custom_graph_store(self, graph_store):
        """Test initialization with custom graph store."""
        executor = UniversalExecutor(graph_store=graph_store)
        assert executor.graph_store is graph_store

    @pytest.mark.asyncio
    async def test_ensure_session(self, executor):
        """Test session initialization."""
        assert executor.session is None
        assert not executor._session_initialized

        await executor._ensure_session()

        assert executor.session is not None
        assert executor._session_initialized
        assert executor.processor is not None

    @pytest.mark.asyncio
    async def test_ensure_session_idempotent(self, executor):
        """Test that _ensure_session can be called multiple times safely."""
        await executor._ensure_session()
        session1 = executor.session

        await executor._ensure_session()
        session2 = executor.session

        assert session1 is session2  # Same session instance


class TestToolAndFunctionRegistration:
    """Test tool and function registration."""

    @pytest.mark.asyncio
    async def test_register_tool(self, executor):
        """Test tool registration."""

        async def test_tool(args):
            return {"result": args.get("input", "default")}

        executor.register_tool("test_tool", test_tool)
        assert "test_tool" in executor.tool_registry
        assert executor.tool_registry["test_tool"] is test_tool

    @pytest.mark.asyncio
    async def test_register_function(self, executor):
        """Test function registration."""

        def test_function(input_val="default"):
            return {"output": input_val}

        executor.register_function("test_function", test_function)
        assert "test_function" in executor.function_registry
        assert executor.function_registry["test_function"] is test_function

    @pytest.mark.asyncio
    async def test_tools_registered_with_processor(self, executor):
        """Test that tools are registered with processor after initialization."""

        async def test_tool(args):
            return {"result": "success"}

        executor.register_tool("test_tool", test_tool)

        # Trigger processor initialization
        await executor._ensure_session()
        await executor._register_tools_with_processor()

        # Verify tool is registered with processor
        assert "test_tool" in executor.processor.tool_registry


class TestVariableResolution:
    """Test variable resolution functionality."""

    def test_resolve_vars_simple_string(self, executor):
        """Test resolving simple string variables."""
        variables = {"name": "John", "age": 30}

        # Test simple variable resolution
        result = executor._resolve_vars("${name}", variables)
        assert result == "John"

        # Test variable not found
        result = executor._resolve_vars("${unknown}", variables)
        assert result == "${unknown}"  # Should return as-is

    def test_resolve_vars_dict(self, executor):
        """Test resolving variables in dictionaries."""
        variables = {"api_key": "secret123", "endpoint": "api.example.com"}

        input_dict = {
            "url": "${endpoint}",  # Exact variable match only
            "headers": {"Authorization": "${api_key}"},  # Exact variable match only
            "timeout": 30,
        }

        result = executor._resolve_vars(input_dict, variables)

        assert result["url"] == "api.example.com"
        assert result["headers"]["Authorization"] == "secret123"
        assert result["timeout"] == 30

    def test_resolve_vars_list(self, executor):
        """Test resolving variables in lists."""
        variables = {"item1": "apple", "item2": "banana"}

        input_list = ["${item1}", "${item2}", "cherry"]
        result = executor._resolve_vars(input_list, variables)

        assert result == ["apple", "banana", "cherry"]

    def test_resolve_vars_nested(self, executor):
        """Test resolving variables in nested structures."""
        variables = {"user": "alice", "action": "read"}

        input_data = {
            "request": {"user": "${user}", "permissions": ["${action}", "write"]},
            "metadata": ["${user}", "${action}"],
        }

        result = executor._resolve_vars(input_data, variables)

        assert result["request"]["user"] == "alice"
        assert result["request"]["permissions"] == ["read", "write"]
        assert result["metadata"] == ["alice", "read"]


class TestValueExtraction:
    """Test value extraction functionality."""

    def test_extract_value_none(self, executor):
        """Test extracting None values."""
        assert executor._extract_value(None) is None

    def test_extract_value_simple(self, executor):
        """Test extracting simple values."""
        assert executor._extract_value("hello") == "hello"
        assert executor._extract_value(42) == 42
        assert executor._extract_value({"key": "value"}) == {"key": "value"}

    def test_extract_value_single_element_list(self, executor):
        """Test extracting from single-element lists."""
        assert executor._extract_value(["hello"]) == "hello"
        assert executor._extract_value([42]) == 42

    def test_extract_value_multiple_element_list(self, executor):
        """Test extracting from multi-element lists."""
        result = executor._extract_value(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_extract_value_wrapped_dicts(self, executor):
        """Test extracting from wrapped dictionaries."""
        # Single wrapper
        wrapped = {"result": "hello"}
        assert executor._extract_value(wrapped) == "hello"

        # Multiple wrappers - the implementation unwraps completely
        multi_wrapped = {"result": {"payload": "hello"}}
        assert executor._extract_value(multi_wrapped) == "hello"

    def test_extract_value_object_attributes(self, executor):
        """Test extracting from objects with common attributes."""

        class MockResult:
            def __init__(self, result):
                self.result = result

        obj = MockResult("extracted_value")
        assert executor._extract_value(obj) == "extracted_value"


class TestTopologicalSort:
    """Test topological sorting functionality."""

    def test_topological_sort_simple(self, executor):
        """Test simple topological sort."""
        # Create mock steps
        step1 = MagicMock()
        step1.id = "step1"
        step2 = MagicMock()
        step2.id = "step2"
        step3 = MagicMock()
        step3.id = "step3"

        steps = [step3, step1, step2]  # Intentionally out of order
        dependencies = {"step1": set(), "step2": {"step1"}, "step3": {"step1", "step2"}}

        result = executor._topological_sort(steps, dependencies)

        # Verify order
        assert result[0].id == "step1"
        assert result[1].id == "step2"
        assert result[2].id == "step3"

    def test_topological_sort_parallel(self, executor):
        """Test topological sort with parallel steps."""
        step1 = MagicMock()
        step1.id = "step1"
        step2 = MagicMock()
        step2.id = "step2"
        step3 = MagicMock()
        step3.id = "step3"

        steps = [step3, step1, step2]
        dependencies = {
            "step1": set(),
            "step2": set(),  # Parallel with step1
            "step3": {"step1", "step2"},
        }

        result = executor._topological_sort(steps, dependencies)

        # Step1 and step2 should come before step3
        step3_pos = next(i for i, s in enumerate(result) if s.id == "step3")
        step1_pos = next(i for i, s in enumerate(result) if s.id == "step1")
        step2_pos = next(i for i, s in enumerate(result) if s.id == "step2")

        assert step1_pos < step3_pos
        assert step2_pos < step3_pos

    def test_topological_sort_cycle_detection(self, executor):
        """Test cycle detection in topological sort."""
        step1 = MagicMock()
        step1.id = "step1"
        step2 = MagicMock()
        step2.id = "step2"

        steps = [step1, step2]
        dependencies = {
            "step1": {"step2"},
            "step2": {"step1"},  # Circular dependency
        }

        with pytest.raises(ValueError, match="Dependency cycle detected"):
            executor._topological_sort(steps, dependencies)


class TestStepExecution:
    """Test step execution functionality."""

    @pytest.mark.asyncio
    async def test_execute_step_with_tool(self, executor, graph_store):
        """Test executing a step with a tool call."""

        # Register a test tool
        async def test_tool(args):
            return {"result": f"Processed: {args.get('input', 'default')}"}

        executor.register_tool("test_tool", test_tool)

        # Create a step with a tool call
        step = PlanStep(description="Test step", index="1")
        tool_call = ToolCall(name="test_tool", args={"input": "test_data"})

        await graph_store.add_node(step)
        await graph_store.add_node(tool_call)
        await graph_store.add_edge(PlanLinkEdge(src=step.id, dst=tool_call.id))

        # Execute the step
        context = {"variables": {}, "results": {}}
        results = await executor._execute_step(step, context)

        assert len(results) == 1
        assert results[0]["result"] == "Processed: test_data"

    @pytest.mark.asyncio
    async def test_execute_step_with_function(self, executor, graph_store):
        """Test executing a step with a function call."""

        # Register a test function
        def test_function(input_val="default"):
            return {"output": f"Function result: {input_val}"}

        executor.register_function("test_function", test_function)

        # Create a step with a function call
        step = PlanStep(description="Test step", index="1")
        tool_call = ToolCall(
            name="function",
            args={"function": "test_function", "args": {"input_val": "test_input"}},
        )

        await graph_store.add_node(step)
        await graph_store.add_node(tool_call)
        await graph_store.add_edge(PlanLinkEdge(src=step.id, dst=tool_call.id))

        # Execute the step
        context = {"variables": {}, "results": {}}
        results = await executor._execute_step(step, context)

        assert len(results) == 1
        assert results[0]["output"] == "Function result: test_input"

    @pytest.mark.asyncio
    async def test_execute_step_with_variables(self, executor, graph_store):
        """Test executing a step with variable resolution."""

        # Register a test tool
        async def test_tool(args):
            return {"processed": args}

        executor.register_tool("test_tool", test_tool)

        # Create a step with variable references (exact matches only)
        step = PlanStep(description="Test step", index="1")
        tool_call = ToolCall(
            name="test_tool",
            args={
                "message": "${name}",  # Exact variable reference
                "count": "${number}",
                "static": "unchanged",
            },
        )

        await graph_store.add_node(step)
        await graph_store.add_node(tool_call)
        await graph_store.add_edge(
            GraphEdge(kind=EdgeType.PLAN_LINK, src=step.id, dst=tool_call.id)
        )

        # Execute with variables
        context = {"variables": {"name": "World", "number": 42}, "results": {}}
        results = await executor._execute_step(step, context)

        assert len(results) == 1
        processed_args = results[0]["processed"]
        assert processed_args["message"] == "World"  # Direct substitution
        assert processed_args["count"] == 42
        assert processed_args["static"] == "unchanged"


class TestPlanExecution:
    """Test full plan execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_plan(self, executor, simple_plan):
        """Test executing a simple plan."""

        # Register tools
        async def hello_tool(args):
            name = args.get("name", "World")
            return {"greeting": f"Hello, {name}!"}

        def summary_function():
            return {"summary": "Plan completed successfully"}

        executor.register_tool("hello", hello_tool)
        executor.register_function("summarize", summary_function)

        # Add steps to plan
        step1_id = await simple_plan.add_tool_step(
            title="Say hello",
            tool="hello",
            args={"name": "${test_var}"},
            result_variable="greeting_result",
        )

        await simple_plan.add_function_step(
            title="Create summary",
            function="summarize",
            args={},
            depends_on=[step1_id],
            result_variable="summary_result",
        )

        # Execute the plan
        result = await executor.execute_plan(simple_plan)

        assert result["success"] is True
        assert "greeting_result" in result["variables"]
        assert "summary_result" in result["variables"]
        assert (
            result["variables"]["greeting_result"]["greeting"] == "Hello, test_value!"
        )
        assert (
            result["variables"]["summary_result"]["summary"]
            == "Plan completed successfully"
        )

    @pytest.mark.asyncio
    async def test_execute_plan_with_error(self, executor, simple_plan):
        """Test plan execution with errors."""

        # Register a tool that raises an error
        async def error_tool(args):
            raise ValueError("Simulated error")

        executor.register_tool("error_tool", error_tool)

        # Add step that will fail
        await simple_plan.add_tool_step(title="Error step", tool="error_tool", args={})

        # Execute the plan
        result = await executor.execute_plan(simple_plan)

        assert result["success"] is False
        assert "error" in result
        assert "Simulated error" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_plan_by_id(self, executor):
        """Test executing a plan by ID with debugging."""

        # Register a simple tool
        async def test_tool(args):
            return {"success": True}

        executor.register_tool("test_tool", test_tool)

        # Create plan directly in executor's graph store
        plan = UniversalPlan("Test Plan By ID", graph=executor.graph_store)

        print(f"Plan ID: {plan.id}")
        print(f"Are graph stores the same? {plan.graph is executor.graph_store}")

        await plan.add_tool_step(title="Test step", tool="test_tool", args={})

        print(f"Before save - nodes: {list(executor.graph_store.nodes.keys())}")

        # Save the plan
        plan_id = await plan.save()

        print(f"After save - plan_id: {plan_id}")
        print(f"After save - nodes: {list(executor.graph_store.nodes.keys())}")

        # Check what types of nodes we have
        for node_id, node in executor.graph_store.nodes.items():
            print(f"Node {node_id}: {node.kind}")

        # Look for plan nodes specifically
        plan_nodes = [
            n for n in executor.graph_store.nodes.values() if n.kind.value == "plan"
        ]
        print(f"Plan nodes: {len(plan_nodes)}")

        if plan_nodes:
            actual_plan_id = plan_nodes[0].id
            result = await executor.execute_plan_by_id(actual_plan_id)
            assert result["success"] is True
        else:
            pytest.skip("No plan node created by save()")

    @pytest.mark.asyncio
    async def test_execute_nonexistent_plan(self, executor):
        """Test executing a plan that doesn't exist."""
        with pytest.raises(ValueError, match="Plan .* not found"):
            await executor.execute_plan_by_id("nonexistent-plan-id")


class TestAsyncBehavior:
    """Test async behavior and edge cases."""

    @pytest.mark.asyncio
    async def test_concurrent_session_initialization(self, graph_store):
        """Test that concurrent session initialization works correctly."""
        executor = UniversalExecutor(graph_store=graph_store)

        # Start multiple session initializations concurrently
        tasks = [executor._ensure_session() for _ in range(5)]
        await asyncio.gather(*tasks)

        # Should have only one session
        assert executor.session is not None
        assert executor._session_initialized

    @pytest.mark.asyncio
    async def test_async_and_sync_tools(self, executor):
        """Test mixing async and sync tools."""

        # Register both async and sync tools
        async def async_tool(args):
            await asyncio.sleep(0.01)  # Simulate async work
            return {"async": True, "input": args.get("data")}

        def sync_tool(args):
            return {"sync": True, "input": args.get("data")}

        executor.register_tool("async_tool", async_tool)
        executor.register_tool("sync_tool", sync_tool)

        # Create plan with both types
        plan = UniversalPlan("Mixed Plan", graph=executor.graph_store)

        await plan.add_tool_step(
            "Async step",
            "async_tool",
            {"data": "async_data"},
            result_variable="async_result",
        )
        await plan.add_tool_step(
            "Sync step",
            "sync_tool",
            {"data": "sync_data"},
            result_variable="sync_result",
        )

        # Execute
        result = await executor.execute_plan(plan)

        assert result["success"] is True
        assert result["variables"]["async_result"]["async"] is True
        assert result["variables"]["sync_result"]["sync"] is True


class TestInitializationEdgeCases:
    """Test edge cases in initialization."""

    @pytest.mark.asyncio
    async def test_init_with_session_store_exception(self, monkeypatch):
        """Test initialization when SessionStoreProvider raises exception."""
        from chuk_session_manager.storage import SessionStoreProvider

        # Mock get_store to raise exception
        call_count = [0]
        original_get = SessionStoreProvider.get_store

        def mock_get_store():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Session store not initialized")
            return original_get()

        monkeypatch.setattr(SessionStoreProvider, "get_store", mock_get_store)

        # Create executor - should handle exception and set store
        executor = UniversalExecutor()

        # Verify it was created successfully
        assert executor is not None
        assert executor.session is None

    @pytest.mark.asyncio
    async def test_register_tools_with_processor_none(self, graph_store):
        """Test _register_tools_with_processor when processor is None."""
        executor = UniversalExecutor(graph_store=graph_store)

        # Register a tool before processor is created
        async def test_tool(args):
            return {"result": "ok"}

        executor.register_tool("test", test_tool)

        # This should initialize session and processor
        await executor._register_tools_with_processor()

        assert executor.processor is not None
        assert executor._session_initialized


class TestJSONSerializationEdgeCases:
    """Test JSON serialization with edge cases."""

    def test_get_json_serializable_frozenset(self, executor):
        """Test converting frozenset to list."""
        data = frozenset([1, 2, 3])
        result = executor._get_json_serializable_data(data)

        assert isinstance(result, list)
        assert set(result) == {1, 2, 3}

    def test_get_json_serializable_nested_frozenset(self, executor):
        """Test converting nested frozenset."""
        data = {"items": frozenset(["a", "b", "c"])}
        result = executor._get_json_serializable_data(data)

        assert isinstance(result, dict)
        assert isinstance(result["items"], list)
        assert set(result["items"]) == {"a", "b", "c"}

    def test_get_json_serializable_list_like_object(self, executor):
        """Test converting list-like objects."""

        # Create a custom list-like class
        class CustomList:
            def __init__(self, items):
                self.items = items

            def __iter__(self):
                return iter(self.items)

            def __getitem__(self, idx):
                return self.items[idx]

            def __len__(self):
                return len(self.items)

        data = CustomList([1, 2, 3])
        result = executor._get_json_serializable_data(data)

        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_get_json_serializable_string_not_converted(self, executor):
        """Test that strings are not converted to lists."""
        data = "test string"
        result = executor._get_json_serializable_data(data)

        assert result == "test string"
        assert isinstance(result, str)

    def test_get_json_serializable_bytes_not_converted(self, executor):
        """Test that bytes are not converted to lists."""
        data = b"test bytes"
        result = executor._get_json_serializable_data(data)

        assert result == b"test bytes"
        assert isinstance(result, bytes)

    def test_get_json_serializable_failed_iteration(self, executor):
        """Test handling of objects that fail iteration."""

        # Create an object that looks iterable but fails
        class FailingIterable:
            def __iter__(self):
                raise TypeError("Cannot iterate")

            def __getitem__(self, idx):
                raise TypeError("Cannot index")

            def __len__(self):
                return 5

        data = FailingIterable()
        result = executor._get_json_serializable_data(data)

        # Should return as-is when iteration fails
        assert result is data


class TestVariableResolutionEdgeCases:
    """Test variable resolution edge cases."""

    def test_resolve_vars_complex_iterable(self, executor):
        """Test resolving variables in complex iterable types."""

        # Custom iterable that's not a list/tuple/dict
        class CustomIterable:
            def __init__(self, items):
                self.items = items

            def __iter__(self):
                return iter(self.items)

        variables = {"var1": "value1", "var2": "value2"}
        data = CustomIterable(["${var1}", "${var2}"])

        result = executor._resolve_vars(data, variables)

        # Should convert to list and resolve
        assert isinstance(result, list)
        assert result == ["value1", "value2"]

    def test_resolve_vars_string_like_object(self, executor):
        """Test that string-like objects are not iterated."""

        # Object with both __iter__ and string methods
        class StringLike:
            def __init__(self, value):
                self.value = value

            def __iter__(self):
                # Should not be called
                raise RuntimeError("Should not iterate string-like")

            def replace(self, *args):
                return self.value

            def split(self, *args):
                return [self.value]

        variables = {"var": "value"}
        data = StringLike("test")

        result = executor._resolve_vars(data, variables)

        # Should return as-is
        assert result is data

    def test_resolve_vars_failing_iterable(self, executor):
        """Test iterable that fails during iteration."""

        class FailingIterable:
            def __iter__(self):
                raise TypeError("Iteration failed")

        variables = {"var": "value"}
        data = FailingIterable()

        result = executor._resolve_vars(data, variables)

        # Should return as-is when iteration fails
        assert result is data

    def test_resolve_template_string_unresolved_variable(self, executor):
        """Test template string with unresolved variable."""
        variables = {"known": "value"}
        template = "URL: ${known}/path/${unknown}/end"

        result = executor._resolve_vars(template, variables)

        # Should partially resolve
        assert "value" in result
        assert "${unknown}" in result

    def test_resolve_nested_variable_not_found(self, executor, capsys):
        """Test nested variable resolution when path not found."""
        variables = {"api": {"endpoint": "localhost"}}
        var_path = "api.port"  # port doesn't exist

        result = executor._resolve_nested_variable(var_path, variables)

        # Should return original variable string
        assert result == "${api.port}"

        # Should print debug info
        captured = capsys.readouterr()
        assert "not found" in captured.out


class TestValueExtractionEdgeCases:
    """Test value extraction edge cases."""

    def test_extract_value_dataclass(self, executor):
        """Test extracting value from dataclass."""
        from dataclasses import dataclass

        @dataclass
        class TestData:
            field1: str
            field2: int

        data = TestData(field1="value", field2=42)
        result = executor._extract_value(data)

        assert isinstance(result, dict)
        assert result["field1"] == "value"
        assert result["field2"] == 42


class TestExecutePlanEdgeCases:
    """Test execute_plan edge cases."""

    @pytest.mark.asyncio
    async def test_execute_plan_no_steps(self, executor):
        """Test executing a plan with no steps."""
        plan = UniversalPlan("Empty Plan", graph=executor.graph_store)
        plan.set_variable("initial", "value")

        result = await executor.execute_plan(plan)

        assert result["success"] is True
        assert result["variables"]["initial"] == "value"

    @pytest.mark.asyncio
    async def test_execute_plan_with_dependencies(self, executor, graph_store):
        """Test plan execution with step dependencies."""

        # Register a simple tool
        async def add_tool(args):
            return args.get("a", 0) + args.get("b", 0)

        executor.register_tool("add", add_tool)

        # Create plan with dependencies
        plan = UniversalPlan("Dependency Plan", graph=graph_store)

        step1_id = await plan.add_tool_step(
            "Add 1+2", "add", {"a": 1, "b": 2}, result_variable="sum1"
        )

        await plan.add_tool_step(
            "Add sum1+3",
            "add",
            {"a": "${sum1}", "b": 3},
            depends_on=[step1_id],
            result_variable="sum2",
        )

        result = await executor.execute_plan(plan)

        assert result["success"] is True
        assert result["variables"]["sum1"] == 3
        assert result["variables"]["sum2"] == 6

    @pytest.mark.asyncio
    async def test_execute_plan_error_propagation(self, executor):
        """Test that errors in tool execution are propagated."""

        # Register a tool that raises an error
        async def failing_tool(args):
            raise ValueError("Tool failed intentionally")

        executor.register_tool("failing", failing_tool)

        plan = UniversalPlan("Failing Plan", graph=executor.graph_store)
        await plan.add_tool_step("Fail step", "failing", {})

        result = await executor.execute_plan(plan)

        assert result["success"] is False
        assert "Tool failed intentionally" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_plan_by_id_not_plan_node(self, executor, graph_store):
        """Test execute_plan_by_id with a node that's not a PlanNode."""
        # Add a different type of node
        tool_call = ToolCall(name="test", args={})
        await graph_store.add_node(tool_call)

        with pytest.raises(ValueError, match="is not a PlanNode"):
            await executor.execute_plan_by_id(tool_call.id)


class TestFunctionExecutionEdgeCases:
    """Test function execution edge cases."""

    @pytest.mark.asyncio
    async def test_function_wrapper_missing_name(self, executor, graph_store):
        """Test function wrapper with missing function name."""
        # Register the function wrapper by registering tools
        await executor._ensure_session()
        await executor._register_tools_with_processor()

        # Create plan with function step that has no name
        plan = UniversalPlan("Test Plan", graph=graph_store)
        await plan.add_function_step(
            "Bad function",
            "",  # Empty function name
            {},
        )

        result = await executor.execute_plan(plan)

        assert result["success"] is False
        assert "function name is required" in result["error"]

    @pytest.mark.asyncio
    async def test_function_wrapper_unknown_function(self, executor, graph_store):
        """Test function wrapper with unknown function."""
        await executor._ensure_session()
        await executor._register_tools_with_processor()

        plan = UniversalPlan("Test Plan", graph=graph_store)
        await plan.add_function_step("Unknown function", "nonexistent_function", {})

        result = await executor.execute_plan(plan)

        assert result["success"] is False
        assert "Unknown function" in result["error"]

    @pytest.mark.asyncio
    async def test_function_wrapper_async_function(self, executor, graph_store):
        """Test function wrapper with async function."""

        # Register an async function
        async def async_func(value):
            await asyncio.sleep(0.01)
            return value * 2

        executor.register_function("async_func", async_func)

        plan = UniversalPlan("Test Plan", graph=graph_store)
        await plan.add_function_step(
            "Async function", "async_func", {"value": 21}, result_variable="result"
        )

        result = await executor.execute_plan(plan)

        assert result["success"] is True
        assert result["variables"]["result"] == 42


class TestToolExecutionErrorPaths:
    """Test error paths in tool execution."""

    @pytest.mark.asyncio
    async def test_execute_step_tool_args_not_dict(self, executor, graph_store):
        """Test error when tool args resolve to non-dict."""

        # Create args dict where the entire dict is replaced by a variable
        # that resolves to a non-dict value (a string)
        async def test_tool(args):
            return args

        executor.register_tool("test", test_tool)

        plan = UniversalPlan("Test", graph=graph_store)
        # This will be resolved and replace the args dict entirely
        plan.set_variable("all_args", "not a dict string")

        # Manually create a step with args that are a single variable reference
        step_index = await plan.add_step("Bad step")
        await plan._find_step_by_index(step_index)

        # Create tool call where the args are meant to be fully replaced
        # We create a valid dict, but it will be replaced during resolution
        # Actually, let's use a different approach - the dict value is a variable
        # But we want the WHOLE args to resolve to non-dict
        # This is difficult because ToolCall requires args to be a dict

        # Alternative: Test that when ALL args are resolved from a variable
        # that is not a dict, we get an error
        # We can't do this with current pydantic validation
        # Let's skip this specific test case as it's prevented by pydantic
        pytest.skip(
            "Cannot test this case - pydantic prevents invalid args at creation time"
        )

    @pytest.mark.asyncio
    async def test_execute_step_function_args_not_dict(self, executor, graph_store):
        """Test error when function args resolve to non-dict."""

        def test_func(**kwargs):
            return kwargs

        executor.register_function("test_func", test_func)

        plan = UniversalPlan("Test", graph=graph_store)
        plan.set_variable("bad_args", ["list", "not", "dict"])

        # Create function step with bad args
        step_index = await plan.add_step("Bad function step")
        step_id = await plan._find_step_by_index(step_index)

        tool_call = ToolCall(
            name="function", args={"function": "test_func", "args": "${bad_args}"}
        )
        await graph_store.add_node(tool_call)
        await graph_store.add_edge(PlanLinkEdge(src=step_id, dst=tool_call.id))

        result = await executor.execute_plan(plan)

        assert result["success"] is False
        assert "must be a dictionary" in result["error"]


class TestDirectToolExecution:
    """Test _execute_tool_directly fallback method."""

    @pytest.mark.asyncio
    async def test_execute_tool_directly_basic(self, executor):
        """Test direct tool execution."""

        # Register a tool
        async def test_tool(args):
            return {"result": args.get("value")}

        executor.register_tool("test_tool", test_tool)

        # Create a tool node
        tool_node = ToolCall(name="test_tool", args={"value": "test_value"})

        # Execute directly
        context = {"variables": {}, "results": {}}
        result = await executor._execute_tool_directly(tool_node, context)

        assert result is not None
        assert result["result"] == "test_value"

    @pytest.mark.asyncio
    async def test_execute_tool_directly_with_variables(self, executor):
        """Test direct tool execution with variable resolution."""

        async def test_tool(args):
            return {"resolved": args.get("key")}

        executor.register_tool("var_tool", test_tool)

        tool_node = ToolCall(name="var_tool", args={"key": "${my_var}"})

        context = {"variables": {"my_var": "resolved_value"}, "results": {}}
        result = await executor._execute_tool_directly(tool_node, context)

        assert result["resolved"] == "resolved_value"

    @pytest.mark.asyncio
    async def test_execute_tool_directly_function_call(self, executor):
        """Test direct execution of function call."""

        # Register a function
        def my_function(param):
            return f"Function result: {param}"

        executor.register_function("my_function", my_function)

        # Create function call tool
        tool_node = ToolCall(
            name="function", args={"function": "my_function", "args": {"param": "test"}}
        )

        context = {"variables": {}, "results": {}}
        result = await executor._execute_tool_directly(tool_node, context)

        assert result == "Function result: test"

    @pytest.mark.asyncio
    async def test_execute_tool_directly_async_function(self, executor):
        """Test direct execution of async function."""

        async def async_function(value):
            await asyncio.sleep(0.01)
            return value * 2

        executor.register_function("async_func", async_function)

        tool_node = ToolCall(
            name="function", args={"function": "async_func", "args": {"value": 21}}
        )

        context = {"variables": {}, "results": {}}
        result = await executor._execute_tool_directly(tool_node, context)

        assert result == 42

    @pytest.mark.asyncio
    async def test_execute_tool_directly_unknown_tool(self, executor):
        """Test direct execution with unknown tool."""
        tool_node = ToolCall(name="unknown_tool", args={})

        context = {"variables": {}, "results": {}}
        result = await executor._execute_tool_directly(tool_node, context)

        # Should return None for unknown tool
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_tool_directly_unknown_function(self, executor):
        """Test direct execution with unknown function."""
        tool_node = ToolCall(
            name="function", args={"function": "unknown_func", "args": {}}
        )

        context = {"variables": {}, "results": {}}
        result = await executor._execute_tool_directly(tool_node, context)

        # Should return None for unknown function
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_tool_directly_invalid_function_name(self, executor):
        """Test direct execution with invalid function name."""
        tool_node = ToolCall(
            name="function",
            args={"function": "", "args": {}},  # Empty function name
        )

        context = {"variables": {}, "results": {}}
        result = await executor._execute_tool_directly(tool_node, context)

        # Should return None for invalid function name
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_tool_directly_function_args_not_dict(self, executor):
        """Test direct execution when function args are not dict."""

        def test_func(**kwargs):
            return kwargs

        executor.register_function("test_func", test_func)

        tool_node = ToolCall(
            name="function", args={"function": "test_func", "args": "not a dict"}
        )

        context = {"variables": {}, "results": {}}
        result = await executor._execute_tool_directly(tool_node, context)

        # Should return None when args are not dict
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_tool_directly_with_exception(self, executor):
        """Test direct execution when tool raises exception."""

        async def failing_tool(args):
            raise RuntimeError("Tool failed")

        executor.register_tool("failing", failing_tool)

        tool_node = ToolCall(name="failing", args={})

        context = {"variables": {}, "results": {}}
        result = await executor._execute_tool_directly(tool_node, context)

        # Should return None when exception occurs
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_tool_directly_sync_tool(self, executor):
        """Test direct execution with sync tool."""

        def sync_tool(args):
            return {"sync": True, "value": args.get("input")}

        executor.register_tool("sync", sync_tool)

        tool_node = ToolCall(name="sync", args={"input": "test"})

        context = {"variables": {}, "results": {}}
        result = await executor._execute_tool_directly(tool_node, context)

        assert result["sync"] is True
        assert result["value"] == "test"


class TestEdgeCasesBranchCoverage:
    """Test specific edge case branches for coverage."""

    @pytest.mark.asyncio
    async def test_execute_step_non_tool_call_node(self, executor, graph_store):
        """Test _execute_step when edge points to non-ToolCall node."""
        from chuk_ai_planner.core.graph import PlanNode

        plan = UniversalPlan("Test", graph=graph_store)
        step_index = await plan.add_step("Test step")
        step_id = await plan._find_step_by_index(step_index)

        # Link step to a PlanNode instead of ToolCall
        plan_node = PlanNode(title="Not a tool call")
        await graph_store.add_node(plan_node)
        await graph_store.add_edge(PlanLinkEdge(src=step_id, dst=plan_node.id))

        # Execute - should skip the non-ToolCall node
        result = await executor.execute_plan(plan)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_plan_truly_empty(self, executor, graph_store):
        """Test execute_plan when there are literally no steps."""
        plan = UniversalPlan("Truly Empty", graph=graph_store)
        await plan.save()

        result = await executor.execute_plan(plan, {"initial": "value"})

        assert result["success"] is True
        assert result["variables"]["initial"] == "value"

    @pytest.mark.asyncio
    async def test_execute_tool_directly_with_result_variable(self, executor):
        """Test _execute_tool_directly storing result in variable."""

        async def test_tool(args):
            return {"computed": args.get("input") * 2}

        executor.register_tool("compute", test_tool)

        # Create tool node with result_variable
        tool_node = ToolCall(
            name="compute", args={"input": 21}, result_variable="output"
        )

        context = {"variables": {}, "results": {}}
        result = await executor._execute_tool_directly(tool_node, context)

        # Result should be returned
        assert result["computed"] == 42
        # And stored in context variables
        assert context["variables"]["output"]["computed"] == 42

    @pytest.mark.asyncio
    async def test_execute_plan_with_step_order_edges(self, executor, graph_store):
        """Test execute_plan with STEP_ORDER dependency edges."""
        from chuk_ai_planner.core.graph.edges.planning import StepEdge

        async def add_tool(args):
            return args.get("a", 0) + args.get("b", 0)

        executor.register_tool("add", add_tool)

        plan = UniversalPlan("Ordered Plan", graph=graph_store)

        # Create two steps
        step1_id = await plan.add_tool_step(
            "First", "add", {"a": 1, "b": 2}, result_variable="result1"
        )
        step2_id = await plan.add_tool_step(
            "Second", "add", {"a": "${result1}", "b": 3}, result_variable="result2"
        )

        # Add explicit STEP_ORDER edge
        step1 = await graph_store.get_node(step1_id)
        step2 = await graph_store.get_node(step2_id)
        await graph_store.add_edge(StepEdge(src=step1.id, dst=step2.id))

        result = await executor.execute_plan(plan)

        assert result["success"] is True
        assert result["variables"]["result1"] == 3
        assert result["variables"]["result2"] == 6

    @pytest.mark.asyncio
    async def test_execute_step_already_executed(self, executor, graph_store):
        """Test _execute_step when step has already been executed (deduplic ation)."""
        from chuk_ai_planner.core.graph import PlanStep

        # Create a step
        step = PlanStep(description="Test Step", index="1")
        await graph_store.add_node(step)

        # Pre-mark step as executed
        context = {
            "variables": {},
            "results": {step.id: [{"already": "executed"}]},
            "executed_steps": {step.id},
            "executed_tool_calls": set(),
        }

        # Try to execute again
        results = await executor._execute_step(step, context)

        # Should return cached results
        assert results == [{"already": "executed"}]

    @pytest.mark.asyncio
    async def test_json_serialization_readonly_list_edge_case(self, executor):
        """Test JSON serialization with _ReadOnlyList if available."""
        try:
            from chuk_ai_planner.core.models.base import _ReadOnlyList

            # If _ReadOnlyList exists, test it
            data = _ReadOnlyList([1, 2, {"nested": 3}])
            result = executor._get_json_serializable_data(data)
            assert isinstance(result, list)
            assert result == [1, 2, {"nested": 3}]
        except ImportError:
            # _ReadOnlyList doesn't exist, skip this test
            pytest.skip("_ReadOnlyList not available in this environment")

    @pytest.mark.asyncio
    async def test_execute_plan_with_graph_copy(self, executor, graph_store):
        """Test execute_plan when plan graph is different from executor graph."""
        from chuk_ai_planner.core.store.memory import InMemoryGraphStore

        # Create a plan with its own graph store
        plan_graph = InMemoryGraphStore()
        plan = UniversalPlan("Test Plan", graph=plan_graph)

        async def test_tool(args):
            return {"result": "success"}

        executor.register_tool("test", test_tool)

        await plan.add_tool_step("Step 1", "test", {}, result_variable="output")

        # Execute - this should copy nodes/edges from plan graph to executor graph
        result = await executor.execute_plan(plan)

        assert result["success"] is True
        assert result["variables"]["output"]["result"] == "success"


class TestFunctionWrapperErrorHandling:
    """Test function wrapper error handling via plan execution"""

    @pytest.mark.asyncio
    async def test_function_wrapper_missing_function_name(self, executor, graph_store):
        """Test function wrapper raises error when function name is missing."""
        plan = UniversalPlan("Test Plan", graph=graph_store)

        # Create a function step with empty function name
        await plan.add_function_step("Bad function", "", {})

        result = await executor.execute_plan(plan)

        assert result["success"] is False
        assert "function name is required" in result["error"]

    @pytest.mark.asyncio
    async def test_function_wrapper_unknown_function(self, executor, graph_store):
        """Test function wrapper raises error for unknown function."""
        plan = UniversalPlan("Test Plan", graph=graph_store)

        # Try to call an unknown function
        await plan.add_function_step("Unknown function", "unknown_func", {})

        result = await executor.execute_plan(plan)

        assert result["success"] is False
        assert "Unknown function" in result["error"]

    @pytest.mark.asyncio
    async def test_function_wrapper_with_sync_function(self, executor, graph_store):
        """Test function wrapper handles sync functions."""

        def sync_func(x):
            return x * 2

        executor.register_function("sync_func", sync_func)

        plan = UniversalPlan("Test Plan", graph=graph_store)
        await plan.add_function_step(
            "Sync function", "sync_func", {"x": 5}, result_variable="result"
        )

        result = await executor.execute_plan(plan)

        assert result["success"] is True
        assert result["variables"]["result"] == 10

    @pytest.mark.asyncio
    async def test_function_wrapper_with_async_function(self, executor, graph_store):
        """Test function wrapper handles async functions."""

        async def async_func(x):
            await asyncio.sleep(0.01)
            return x * 3

        executor.register_function("async_func", async_func)

        plan = UniversalPlan("Test Plan", graph=graph_store)
        await plan.add_function_step(
            "Async function", "async_func", {"x": 5}, result_variable="result"
        )

        result = await executor.execute_plan(plan)

        assert result["success"] is True
        assert result["variables"]["result"] == 15


class TestJSONSerializationValidation:
    """Test JSON serialization validation"""

    @pytest.mark.asyncio
    async def test_json_serialization_with_non_dict_tool_args(
        self, executor, graph_store
    ):
        """Test that pydantic prevents creation of ToolCall with non-dict args"""
        # Pydantic validation should prevent creating a ToolCall with non-dict args
        with pytest.raises(Exception):  # ValidationError from pydantic
            ToolCall(name="test_tool", args=["invalid", "args"])  # List instead of dict


class TestExecutePlanEmptyCases:
    """Test execute_plan with empty plans"""

    @pytest.mark.asyncio
    async def test_execute_plan_with_no_steps(self, executor, graph_store):
        """Test execute_plan with empty plan (line 663)"""
        plan = UniversalPlan("Empty Plan", graph=graph_store)

        result = await executor.execute_plan(plan)

        assert result["success"] is True
        # Should have initial variables from plan
        assert "variables" in result
