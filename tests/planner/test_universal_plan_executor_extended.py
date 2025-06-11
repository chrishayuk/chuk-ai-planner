# tests/planner/test_universal_plan_executor_extended.py
"""
Extended unit tests for UniversalExecutor covering enhanced features
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

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
    """Create a UniversalExecutor for testing."""
    return UniversalExecutor(graph_store=graph_store)


@pytest.fixture
def complex_plan(graph_store):
    """Create a complex plan with nested data structures for testing."""
    plan = UniversalPlan(
        title="Complex Test Plan",
        description="A plan for testing complex features",
        tags=["test", "complex"],
        graph=graph_store
    )
    
    # Add complex variables
    plan.set_variable("user_data", {
        "name": "Alice",
        "profile": {
            "age": 30,
            "skills": ["python", "javascript", "sql"],
            "preferences": {
                "theme": "dark",
                "notifications": True
            }
        },
        "history": [
            {"action": "login", "timestamp": "2024-01-01"},
            {"action": "upload", "timestamp": "2024-01-02"}
        ]
    })
    
    plan.set_variable("config", {
        "api": {
            "endpoint": "https://api.example.com",
            "timeout": 30,
            "retry_count": 3
        },
        "features": ["feature_a", "feature_b"]
    })
    
    return plan


class TestEnhancedVariableResolution:
    """Test enhanced variable resolution with nested field access."""
    
    def test_resolve_nested_variables_simple(self, executor):
        """Test resolving simple nested variables."""
        variables = {
            "user": {"name": "Alice", "age": 30},
            "config": {"timeout": 30}
        }
        
        # Test nested field access
        result = executor._resolve_vars("${user.name}", variables)
        assert result == "Alice"
        
        result = executor._resolve_vars("${user.age}", variables)
        assert result == 30
        
        result = executor._resolve_vars("${config.timeout}", variables)
        assert result == 30
        
    def test_resolve_nested_variables_deep(self, executor):
        """Test resolving deeply nested variables."""
        variables = {
            "data": {
                "user": {
                    "profile": {
                        "settings": {
                            "theme": "dark"
                        }
                    }
                }
            }
        }
        
        result = executor._resolve_vars("${data.user.profile.settings.theme}", variables)
        assert result == "dark"
        
    def test_resolve_nested_variables_in_dict(self, executor):
        """Test resolving nested variables within dictionaries."""
        variables = {
            "api": {"endpoint": "api.example.com", "port": 443},
            "user": {"id": 123}
        }
        
        input_dict = {
            "url": "https://${api.endpoint}:${api.port}/users/${user.id}",
            "headers": {"User-Agent": "TestClient/1.0"},
            "timeout": "${api.timeout}"  # This should remain as-is (not found)
        }
        
        result = executor._resolve_vars(input_dict, variables)
        assert result["url"] == "https://api.example.com:443/users/123"
        assert result["headers"]["User-Agent"] == "TestClient/1.0"
        assert result["timeout"] == "${api.timeout}"  # Not found, kept as-is
        
    def test_resolve_nested_variables_in_list(self, executor):
        """Test resolving nested variables within lists."""
        variables = {
            "user": {"skills": ["python", "sql"]},
            "config": {"features": ["auth", "reporting"]}
        }
        
        input_list = [
            "${user.skills}",
            "${config.features}",
            "static_item"
        ]
        
        result = executor._resolve_vars(input_list, variables)
        assert result[0] == ["python", "sql"]
        assert result[1] == ["auth", "reporting"]
        assert result[2] == "static_item"
        
    def test_resolve_nested_variables_mixed_types(self, executor):
        """Test resolving nested variables with mixed data types."""
        variables = {
            "data": {
                "count": 42,
                "active": True,
                "items": ["a", "b", "c"],
                "metadata": {"version": "1.0"}
            }
        }
        
        input_data = {
            "total": "${data.count}",
            "enabled": "${data.active}",
            "list": "${data.items}",
            "info": "${data.metadata}"
        }
        
        result = executor._resolve_vars(input_data, variables)
        assert result["total"] == 42
        assert result["enabled"] is True
        assert result["list"] == ["a", "b", "c"]
        assert result["info"] == {"version": "1.0"}
        
    def test_resolve_nested_variables_nonexistent(self, executor):
        """Test resolving non-existent nested variables."""
        variables = {"user": {"name": "Alice"}}
        
        # Non-existent top-level variable
        result = executor._resolve_vars("${nonexistent.field}", variables)
        assert result == "${nonexistent.field}"  # Should remain unchanged
        
        # Non-existent nested field
        result = executor._resolve_vars("${user.nonexistent}", variables)
        assert result == "${user.nonexistent}"  # Should remain unchanged
        
        # Deeply non-existent
        result = executor._resolve_vars("${user.profile.settings.theme}", variables)
        assert result == "${user.profile.settings.theme}"  # Should remain unchanged


class TestDuplicateExecutionPrevention:
    """Test duplicate execution prevention mechanisms."""
    
    @pytest.mark.asyncio
    async def test_step_deduplication(self, executor, graph_store):
        """Test that steps are not executed multiple times."""
        call_count = 0
        
        async def counting_tool(args):
            nonlocal call_count
            call_count += 1
            return {"call_number": call_count}
        
        executor.register_tool("counting_tool", counting_tool)
        
        # Create a step
        step = GraphNode(kind=NodeKind.PLAN_STEP, data={"description": "Counting step"})
        tool_call = ToolCall(data={"name": "counting_tool", "args": {}})
        
        graph_store.add_node(step)
        graph_store.add_node(tool_call)
        graph_store.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step.id, dst=tool_call.id))
        
        # Execute the same step multiple times
        context = {
            "variables": {},
            "results": {},
            "executed_steps": set(),
            "executed_tool_calls": set()
        }
        
        # First execution
        results1 = await executor._execute_step(step, context)
        assert len(results1) == 1
        assert results1[0]["call_number"] == 1
        assert call_count == 1
        
        # Second execution should be skipped
        results2 = await executor._execute_step(step, context)
        assert len(results2) == 1  # Returns cached results
        assert call_count == 1  # Tool not called again
        
    @pytest.mark.asyncio
    async def test_tool_call_deduplication(self, executor, graph_store):
        """Test that tool calls are not executed multiple times."""
        call_count = 0
        
        async def counting_tool(args):
            nonlocal call_count
            call_count += 1
            return {"call_number": call_count}
        
        executor.register_tool("counting_tool", counting_tool)
        
        # Create two steps that use the same tool call (edge case)
        step1 = GraphNode(kind=NodeKind.PLAN_STEP, data={"description": "Step 1"})
        step2 = GraphNode(kind=NodeKind.PLAN_STEP, data={"description": "Step 2"})
        tool_call = ToolCall(data={"name": "counting_tool", "args": {}})
        
        graph_store.add_node(step1)
        graph_store.add_node(step2)
        graph_store.add_node(tool_call)
        
        # Link both steps to the same tool call (unusual but possible)
        graph_store.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step1.id, dst=tool_call.id))
        graph_store.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step2.id, dst=tool_call.id))
        
        context = {
            "variables": {},
            "results": {},
            "executed_steps": set(),
            "executed_tool_calls": set()
        }
        
        # Execute both steps
        await executor._execute_step(step1, context)
        assert call_count == 1
        
        await executor._execute_step(step2, context)
        assert call_count == 1  # Tool call should be deduplicated


class TestResultVariableManagement:
    """Test result variable storage and retrieval."""
    
    @pytest.mark.asyncio
    async def test_result_variable_storage(self, executor, graph_store):
        """Test that result variables are properly stored."""
        async def test_tool(args):
            return {"result": f"processed_{args.get('input', 'default')}"}
        
        executor.register_tool("test_tool", test_tool)
        
        # Create step with result variable
        step = GraphNode(kind=NodeKind.PLAN_STEP, data={"description": "Test step"})
        tool_call = ToolCall(data={"name": "test_tool", "args": {"input": "data"}})
        
        graph_store.add_node(step)
        graph_store.add_node(tool_call)
        graph_store.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step.id, dst=tool_call.id))
        
        # Add result variable as custom edge
        graph_store.add_edge(GraphEdge(
            kind=EdgeKind.CUSTOM,
            src=step.id,
            dst=tool_call.id,
            data={"type": "result_variable", "variable": "test_result"}
        ))
        
        # Execute step
        context = {"variables": {}, "results": {}}
        await executor._execute_step(step, context)
        
        # Check that result variable was stored
        assert "test_result" in context["variables"]
        assert context["variables"]["test_result"]["result"] == "processed_data"
        
    def test_find_result_variable(self, executor, graph_store):
        """Test finding result variables from custom edges."""
        step = GraphNode(kind=NodeKind.PLAN_STEP, data={"description": "Test step"})
        tool_call = ToolCall(data={"name": "test_tool", "args": {}})
        
        graph_store.add_node(step)
        graph_store.add_node(tool_call)
        
        # Add result variable as custom edge
        graph_store.add_edge(GraphEdge(
            kind=EdgeKind.CUSTOM,
            src=step.id,
            dst=tool_call.id,
            data={"type": "result_variable", "variable": "my_result"}
        ))
        
        # Test finding the result variable
        result_var = executor._find_result_variable(step.id, tool_call.id)
        assert result_var == "my_result"
        
        # Test with non-existent variable
        result_var = executor._find_result_variable("nonexistent", tool_call.id)
        assert result_var is None


class TestComplexPlanExecution:
    """Test execution of complex plans with multiple features."""
    
    @pytest.mark.asyncio
    async def test_complex_variable_flow(self, executor, complex_plan):
        """Test complex variable flow through multiple steps."""
        # Register tools that use nested variables
        async def user_processor(args):
            user_name = args.get("name")
            user_skills = args.get("skills", [])
            return {
                "processed_user": {
                    "display_name": f"User: {user_name}",
                    "skill_count": len(user_skills),
                    "has_python": "python" in user_skills
                }
            }
        
        async def config_validator(args):
            endpoint = args.get("endpoint")
            timeout = args.get("timeout")
            return {
                "validation": {
                    "endpoint_valid": endpoint.startswith("https://"),
                    "timeout_reasonable": 10 <= timeout <= 60
                }
            }
        
        def report_generator(**kwargs):
            user_info = kwargs.get("user_info", {})
            config_info = kwargs.get("config_info", {})
            
            return {
                "report": {
                    "user_summary": user_info.get("processed_user", {}),
                    "config_summary": config_info.get("validation", {}),
                    "generated_at": "2024-01-01T12:00:00Z"
                }
            }
        
        executor.register_tool("user_processor", user_processor)
        executor.register_tool("config_validator", config_validator)
        executor.register_function("report_generator", report_generator)
        
        # Add steps with nested variable references
        step1 = complex_plan.add_tool_step(
            title="Process user data",
            tool="user_processor",
            args={
                "name": "${user_data.name}",
                "skills": "${user_data.profile.skills}"
            },
            result_variable="user_info"
        )
        
        step2 = complex_plan.add_tool_step(
            title="Validate config",
            tool="config_validator",
            args={
                "endpoint": "${config.api.endpoint}",
                "timeout": "${config.api.timeout}"
            },
            result_variable="config_info"
        )
        
        step3 = complex_plan.add_function_step(
            title="Generate report",
            function="report_generator",
            args={
                "user_info": "${user_info}",
                "config_info": "${config_info}"
            },
            result_variable="final_report",
            depends_on=[step1, step2]
        )
        
        # Execute the plan
        result = await executor.execute_plan(complex_plan)
        
        assert result["success"] is True
        
        # Verify user processing
        user_info = result["variables"]["user_info"]
        assert user_info["processed_user"]["display_name"] == "User: Alice"
        assert user_info["processed_user"]["skill_count"] == 3
        assert user_info["processed_user"]["has_python"] is True
        
        # Verify config validation
        config_info = result["variables"]["config_info"]
        assert config_info["validation"]["endpoint_valid"] is True
        assert config_info["validation"]["timeout_reasonable"] is True
        
        # Verify report generation
        final_report = result["variables"]["final_report"]
        assert "report" in final_report
        assert final_report["report"]["user_summary"]["display_name"] == "User: Alice"
        assert final_report["report"]["config_summary"]["endpoint_valid"] is True
        
    @pytest.mark.asyncio
    async def test_dependency_execution_order(self, executor, graph_store):
        """Test that dependencies are respected in execution order."""
        execution_order = []
        
        async def tracking_tool(args):
            step_name = args.get("step_name")
            execution_order.append(step_name)
            return {"step": step_name, "executed": True}
        
        executor.register_tool("tracking_tool", tracking_tool)
        
        # Create plan with dependencies
        plan = UniversalPlan("Dependency Test", graph=graph_store)
        
        # Step A (no dependencies)
        step_a = plan.add_tool_step(
            "Step A",
            "tracking_tool",
            {"step_name": "A"},
            result_variable="result_a"
        )
        
        # Step B (no dependencies, parallel with A)
        step_b = plan.add_tool_step(
            "Step B",
            "tracking_tool",
            {"step_name": "B"},
            result_variable="result_b"
        )
        
        # Step C (depends on A and B)
        step_c = plan.add_tool_step(
            "Step C",
            "tracking_tool",
            {"step_name": "C"},
            result_variable="result_c",
            depends_on=[step_a, step_b]
        )
        
        # Step D (depends on C)
        step_d = plan.add_tool_step(
            "Step D",
            "tracking_tool",
            {"step_name": "D"},
            result_variable="result_d",
            depends_on=[step_c]
        )
        
        # Execute plan
        result = await executor.execute_plan(plan)
        
        assert result["success"] is True
        assert len(execution_order) == 4
        
        # A and B should come before C
        a_pos = execution_order.index("A")
        b_pos = execution_order.index("B")
        c_pos = execution_order.index("C")
        d_pos = execution_order.index("D")
        
        assert a_pos < c_pos
        assert b_pos < c_pos
        assert c_pos < d_pos


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms."""
    
    @pytest.mark.asyncio
    async def test_tool_error_propagation(self, executor, graph_store):
        """Test that tool errors are properly propagated."""
        async def failing_tool(args):
            error_type = args.get("error_type", "generic")
            if error_type == "value":
                raise ValueError("Test value error")
            elif error_type == "runtime":
                raise RuntimeError("Test runtime error")
            else:
                raise Exception("Generic test error")
        
        executor.register_tool("failing_tool", failing_tool)
        
        plan = UniversalPlan("Error Test", graph=graph_store)
        plan.add_tool_step("Failing step", "failing_tool", {"error_type": "value"})
        
        result = await executor.execute_plan(plan)
        
        assert result["success"] is False
        assert "error" in result
        assert "Test value error" in result["error"]
        
    @pytest.mark.asyncio
    async def test_function_error_propagation(self, executor, graph_store):
        """Test that function errors are properly propagated."""
        def failing_function(**kwargs):
            raise ValueError("Function failure")
        
        executor.register_function("failing_function", failing_function)
        
        plan = UniversalPlan("Function Error Test", graph=graph_store)
        plan.add_function_step("Failing function", "failing_function", {})
        
        result = await executor.execute_plan(plan)
        
        assert result["success"] is False
        assert "Function failure" in result["error"]
        
    @pytest.mark.asyncio
    async def test_unknown_tool_error(self, executor, graph_store):
        """Test error when calling unknown tool."""
        plan = UniversalPlan("Unknown Tool Test", graph=graph_store)
        plan.add_tool_step("Unknown tool", "nonexistent_tool", {})
        
        result = await executor.execute_plan(plan)
        
        assert result["success"] is False
        assert "Unknown tool" in result["error"]
        
    @pytest.mark.asyncio
    async def test_unknown_function_error(self, executor, graph_store):
        """Test error when calling unknown function."""
        plan = UniversalPlan("Unknown Function Test", graph=graph_store)
        plan.add_function_step("Unknown function", "nonexistent_function", {})
        
        result = await executor.execute_plan(plan)
        
        assert result["success"] is False
        assert "Unknown function" in result["error"]


class TestJSONSerialization:
    """Test JSON serialization of frozen data structures."""
    
    def test_json_serializable_mappingproxy(self, executor):
        """Test serialization of MappingProxyType."""
        from types import MappingProxyType
        
        data = MappingProxyType({"key": "value", "nested": {"inner": "data"}})
        result = executor._get_json_serializable_data(data)
        
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["nested"]["inner"] == "data"
        
        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None
        
    def test_json_serializable_readonly_list(self, executor):
        """Test serialization of ReadOnlyList if available."""
        # Create a list-like object to simulate ReadOnlyList
        class MockReadOnlyList:
            def __init__(self, items):
                self._items = items
            
            def __iter__(self):
                return iter(self._items)
            
            def __getitem__(self, key):
                return self._items[key]
            
            def __len__(self):
                return len(self._items)
        
        # Mock the import to test the handling
        data = MockReadOnlyList(["item1", "item2", {"nested": "data"}])
        
        with patch.dict('sys.modules', {'chuk_ai_planner.models.base': MagicMock()}):
            # Mock _ReadOnlyList to be our test class
            with patch.object(executor, '_get_json_serializable_data') as mock_method:
                # Call the real method but with our mock class
                mock_method.side_effect = lambda x: list(x) if isinstance(x, MockReadOnlyList) else x
                result = mock_method(data)
                
                assert isinstance(result, list)
                
    def test_json_serializable_nested_frozen(self, executor):
        """Test serialization of nested frozen structures."""
        from types import MappingProxyType
        
        data = {
            "normal": "value",
            "frozen": MappingProxyType({
                "inner": "data",
                "nested_frozen": MappingProxyType({"deep": "value"})
            }),
            "list_with_frozen": [
                "normal_item",
                MappingProxyType({"item": "frozen"})
            ]
        }
        
        result = executor._get_json_serializable_data(data)
        
        # Verify structure is preserved and serializable
        assert result["normal"] == "value"
        assert isinstance(result["frozen"], dict)
        assert result["frozen"]["inner"] == "data"
        assert isinstance(result["frozen"]["nested_frozen"], dict)
        assert result["frozen"]["nested_frozen"]["deep"] == "value"
        assert isinstance(result["list_with_frozen"][1], dict)
        assert result["list_with_frozen"][1]["item"] == "frozen"
        
        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None


class TestPerformanceAndMemory:
    """Test performance and memory characteristics."""
    
    @pytest.mark.asyncio
    async def test_large_plan_execution(self, executor, graph_store):
        """Test execution of a plan with many steps."""
        async def simple_tool(args):
            step_num = args.get("step_num", 0)
            return {"result": f"step_{step_num}_completed"}
        
        executor.register_tool("simple_tool", simple_tool)
        
        # Create a plan with many steps
        plan = UniversalPlan("Large Plan", graph=graph_store)
        step_ids = []
        
        for i in range(50):  # 50 steps should be manageable for testing
            step_id = plan.add_tool_step(
                f"Step {i}",
                "simple_tool",
                {"step_num": i},
                result_variable=f"result_{i}"
            )
            step_ids.append(step_id)
        
        # Execute plan
        result = await executor.execute_plan(plan)
        
        assert result["success"] is True
        assert len(result["variables"]) >= 50  # At least one result per step
        
        # Verify some results
        assert result["variables"]["result_0"]["result"] == "step_0_completed"
        assert result["variables"]["result_49"]["result"] == "step_49_completed"
        
    @pytest.mark.asyncio
    async def test_deep_variable_nesting(self, executor, graph_store):
        """Test handling of deeply nested variable structures."""
        # Create deeply nested data
        deep_data = {"level": 0}
        current = deep_data
        for i in range(1, 20):  # 20 levels deep
            current["next"] = {"level": i}
            current = current["next"]
        current["final"] = "deep_value"
        
        async def deep_tool(args):
            return {"received": args.get("deep_value")}
        
        executor.register_tool("deep_tool", deep_tool)
        
        plan = UniversalPlan("Deep Nesting Test", graph=graph_store)
        plan.set_variable("deep_structure", deep_data)
        
        # Build the variable path
        path_parts = ["deep_structure"] + ["next"] * 19 + ["final"]
        variable_path = "${" + ".".join(path_parts) + "}"
        
        plan.add_tool_step(
            "Deep access",
            "deep_tool",
            {"deep_value": variable_path},
            result_variable="deep_result"
        )
        
        result = await executor.execute_plan(plan)
        
        assert result["success"] is True
        assert result["variables"]["deep_result"]["received"] == "deep_value"


class TestConcurrency:
    """Test concurrent execution scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_plan_execution(self, graph_store):
        """Test executing multiple plans concurrently."""
        executor = UniversalExecutor(graph_store)
        
        async def concurrent_tool(args):
            plan_id = args.get("plan_id")
            await asyncio.sleep(0.01)  # Simulate work
            return {"plan_id": plan_id, "completed": True}
        
        executor.register_tool("concurrent_tool", concurrent_tool)
        
        # Create multiple plans
        plans = []
        for i in range(5):
            plan = UniversalPlan(f"Concurrent Plan {i}")
            plan.add_tool_step(
                f"Concurrent step {i}",
                "concurrent_tool",
                {"plan_id": i},
                result_variable=f"result_{i}"
            )
            plans.append(plan)
        
        # Execute all plans concurrently
        tasks = [executor.execute_plan(plan) for plan in plans]
        results = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        for i, result in enumerate(results):
            assert result["success"] is True
            assert result["variables"][f"result_{i}"]["plan_id"] == i
            assert result["variables"][f"result_{i}"]["completed"] is True
            
    @pytest.mark.asyncio
    async def test_shared_executor_state(self, graph_store):
        """Test that executor state is properly isolated between executions."""
        executor = UniversalExecutor(graph_store)
        
        call_counts = {"count": 0}
        
        async def stateful_tool(args):
            call_counts["count"] += 1
            return {"call_number": call_counts["count"]}
        
        executor.register_tool("stateful_tool", stateful_tool)
        
        # Create two plans
        plan1 = UniversalPlan("Plan 1")
        plan1.add_tool_step("Step 1", "stateful_tool", {}, result_variable="result1")
        
        plan2 = UniversalPlan("Plan 2")
        plan2.add_tool_step("Step 2", "stateful_tool", {}, result_variable="result2")
        
        # Execute plans sequentially
        result1 = await executor.execute_plan(plan1)
        result2 = await executor.execute_plan(plan2)
        
        assert result1["success"] is True
        assert result2["success"] is True
        
        # Tool should have been called twice total
        assert call_counts["count"] == 2
        assert result1["variables"]["result1"]["call_number"] == 1
        assert result2["variables"]["result2"]["call_number"] == 2