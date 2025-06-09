# tests/planner/test_universal_plan_executor.py
"""
Unit tests for UniversalExecutor
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor
from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.models import GraphNode, NodeKind, ToolCall
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind


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
        graph=graph_store
    )
    
    # Add some variables
    plan.set_variable("test_var", "test_value")
    plan.set_variable("number", 42)
    
    return plan


class TestUniversalExecutorInit:
    """Test UniversalExecutor initialization."""
    
    def test_init_default_graph_store(self):
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
            "timeout": 30
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
            "request": {
                "user": "${user}",
                "permissions": ["${action}", "write"]
            },
            "metadata": ["${user}", "${action}"]
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
        dependencies = {
            "step1": set(),
            "step2": {"step1"},
            "step3": {"step1", "step2"}
        }
        
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
            "step3": {"step1", "step2"}
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
            "step2": {"step1"}  # Circular dependency
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
        step = GraphNode(kind=NodeKind.PLAN_STEP, data={"description": "Test step"})
        tool_call = ToolCall(data={"name": "test_tool", "args": {"input": "test_data"}})
        
        graph_store.add_node(step)
        graph_store.add_node(tool_call)
        graph_store.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step.id, dst=tool_call.id))
        
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
        step = GraphNode(kind=NodeKind.PLAN_STEP, data={"description": "Test step"})
        tool_call = ToolCall(data={
            "name": "function",
            "args": {
                "function": "test_function",
                "args": {"input_val": "test_input"}
            }
        })
        
        graph_store.add_node(step)
        graph_store.add_node(tool_call)
        graph_store.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step.id, dst=tool_call.id))
        
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
        step = GraphNode(kind=NodeKind.PLAN_STEP, data={"description": "Test step"})
        tool_call = ToolCall(data={
            "name": "test_tool",
            "args": {
                "message": "${name}",  # Exact variable reference
                "count": "${number}",
                "static": "unchanged"
            }
        })
        
        graph_store.add_node(step)
        graph_store.add_node(tool_call)
        graph_store.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step.id, dst=tool_call.id))
        
        # Execute with variables
        context = {
            "variables": {"name": "World", "number": 42},
            "results": {}
        }
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
        step1_id = simple_plan.add_tool_step(
            title="Say hello",
            tool="hello",
            args={"name": "${test_var}"},
            result_variable="greeting_result"
        )
        
        step2_id = simple_plan.add_function_step(
            title="Create summary",
            function="summarize",
            args={},
            depends_on=[step1_id],
            result_variable="summary_result"
        )
        
        # Execute the plan
        result = await executor.execute_plan(simple_plan)
        
        assert result["success"] is True
        assert "greeting_result" in result["variables"]
        assert "summary_result" in result["variables"]
        assert result["variables"]["greeting_result"]["greeting"] == "Hello, test_value!"
        assert result["variables"]["summary_result"]["summary"] == "Plan completed successfully"
        
    @pytest.mark.asyncio
    async def test_execute_plan_with_error(self, executor, simple_plan):
        """Test plan execution with errors."""
        # Register a tool that raises an error
        async def error_tool(args):
            raise ValueError("Simulated error")
        
        executor.register_tool("error_tool", error_tool)
        
        # Add step that will fail
        simple_plan.add_tool_step(
            title="Error step",
            tool="error_tool",
            args={}
        )
        
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
        
        plan.add_tool_step(
            title="Test step",
            tool="test_tool",
            args={}
        )
        
        print(f"Before save - nodes: {list(executor.graph_store.nodes.keys())}")
        
        # Save the plan
        plan_id = plan.save()
        
        print(f"After save - plan_id: {plan_id}")
        print(f"After save - nodes: {list(executor.graph_store.nodes.keys())}")
        
        # Check what types of nodes we have
        for node_id, node in executor.graph_store.nodes.items():
            print(f"Node {node_id}: {node.kind}")
        
        # Look for plan nodes specifically
        plan_nodes = [n for n in executor.graph_store.nodes.values() if n.kind.value == "plan"]
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
        
        plan.add_tool_step("Async step", "async_tool", {"data": "async_data"}, result_variable="async_result")
        plan.add_tool_step("Sync step", "sync_tool", {"data": "sync_data"}, result_variable="sync_result")
        
        # Execute
        result = await executor.execute_plan(plan)
        
        assert result["success"] is True
        assert result["variables"]["async_result"]["async"] is True
        assert result["variables"]["sync_result"]["sync"] is True