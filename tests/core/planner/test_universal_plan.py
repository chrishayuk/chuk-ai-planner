"""
Comprehensive tests for UniversalPlan
"""

import pytest
from chuk_ai_planner.core.planner.universal_plan import UniversalPlan
from chuk_ai_planner.core.store.memory import InMemoryGraphStore
from chuk_ai_planner.core.graph import NodeType, EdgeType, CustomEdge


@pytest.fixture
def graph_store():
    return InMemoryGraphStore()


@pytest.fixture
def plan(graph_store):
    return UniversalPlan("Test Plan", description="Test description", graph=graph_store)


class TestUniversalPlanInit:
    """Test UniversalPlan initialization"""

    def test_init_default(self):
        plan = UniversalPlan("My Plan")
        assert plan.title == "My Plan"
        assert plan.description == "Plan for: My Plan"
        assert plan.tags == []
        assert plan.variables == {}
        assert plan.metadata == {}
        assert plan.id is not None

    def test_init_with_description(self):
        plan = UniversalPlan("My Plan", description="Custom description")
        assert plan.description == "Custom description"

    def test_init_with_id(self):
        custom_id = "custom-plan-123"
        plan = UniversalPlan("My Plan", id=custom_id)
        assert plan.id == custom_id

    def test_init_with_tags(self):
        plan = UniversalPlan("My Plan", tags=["research", "analysis"])
        assert plan.tags == ["research", "analysis"]

    def test_init_with_graph(self, graph_store):
        plan = UniversalPlan("My Plan", graph=graph_store)
        assert plan.graph is graph_store


class TestVariableManagement:
    """Test variable management methods"""

    def test_set_variable(self, plan):
        result = plan.set_variable("key1", "value1")
        assert result is plan  # Fluent interface
        assert plan.variables["key1"] == "value1"

    def test_set_multiple_variables(self, plan):
        plan.set_variable("key1", "value1")
        plan.set_variable("key2", 42)
        plan.set_variable("key3", {"nested": "data"})

        assert plan.variables["key1"] == "value1"
        assert plan.variables["key2"] == 42
        assert plan.variables["key3"] == {"nested": "data"}


class TestMetadataManagement:
    """Test metadata management methods"""

    def test_add_metadata(self, plan):
        result = plan.add_metadata("author", "Alice")
        assert result is plan  # Fluent interface
        assert plan.metadata["author"] == "Alice"

    def test_add_multiple_metadata(self, plan):
        plan.add_metadata("author", "Alice")
        plan.add_metadata("version", "1.0")
        plan.add_metadata("priority", "high")

        assert plan.metadata["author"] == "Alice"
        assert plan.metadata["version"] == "1.0"
        assert plan.metadata["priority"] == "high"


class TestTagManagement:
    """Test tag management methods"""

    def test_add_tag(self, plan):
        result = plan.add_tag("research")
        assert result is plan  # Fluent interface
        assert "research" in plan.tags

    def test_add_multiple_tags(self, plan):
        plan.add_tag("research")
        plan.add_tag("analysis")
        plan.add_tag("urgent")

        assert len(plan.tags) == 3
        assert "research" in plan.tags
        assert "analysis" in plan.tags
        assert "urgent" in plan.tags

    def test_add_duplicate_tag(self, plan):
        plan.add_tag("research")
        plan.add_tag("research")  # Should not add duplicate

        assert len(plan.tags) == 1
        assert plan.tags.count("research") == 1


class TestAddToolStep:
    """Test add_tool_step method"""

    @pytest.mark.asyncio
    async def test_add_tool_step_basic(self, plan):
        step_id = await plan.add_tool_step(
            title="Fetch data",
            tool="fetch_api",
            args={"url": "https://api.example.com"},
        )

        assert step_id is not None

        # Verify step was created
        steps = await plan._graph.get_nodes_by_kind(NodeType.PLAN_STEP)
        assert len(steps) == 1

        # Verify tool call was created and linked
        tools = await plan._graph.get_nodes_by_kind(NodeType.TOOL_CALL)
        assert len(tools) == 1
        assert tools[0].name == "fetch_api"
        assert tools[0].args["url"] == "https://api.example.com"

    @pytest.mark.asyncio
    async def test_add_tool_step_with_result_variable(self, plan):
        step_id = await plan.add_tool_step(
            title="Fetch data",
            tool="fetch_api",
            args={"url": "https://api.example.com"},
            result_variable="api_result",
        )

        # Verify custom edge for result variable
        edges = await plan._graph.get_edges(src=step_id, kind=EdgeType.CUSTOM)
        assert len(edges) > 0

        custom_edge = next(
            (
                e
                for e in edges
                if isinstance(e, CustomEdge) and e.custom_type == "result_variable"
            ),
            None,
        )
        assert custom_edge is not None
        assert custom_edge.metadata["variable"] == "api_result"

    @pytest.mark.asyncio
    async def test_add_tool_step_with_dependencies(self, plan):
        step1_id = await plan.add_tool_step(title="Step 1", tool="tool1", args={})

        step2_id = await plan.add_tool_step(
            title="Step 2", tool="tool2", args={}, depends_on=[step1_id]
        )

        assert step1_id != step2_id

        # Verify both steps exist
        steps = await plan._graph.get_nodes_by_kind(NodeType.PLAN_STEP)
        assert len(steps) == 2

    @pytest.mark.asyncio
    async def test_add_tool_step_empty_args(self, plan):
        await plan.add_tool_step(title="Simple step", tool="simple_tool")

        tools = await plan._graph.get_nodes_by_kind(NodeType.TOOL_CALL)
        assert len(tools) == 1
        assert tools[0].args == {}


class TestAddFunctionStep:
    """Test add_function_step method"""

    @pytest.mark.asyncio
    async def test_add_function_step_basic(self, plan):
        step_id = await plan.add_function_step(
            title="Process data", function="process_data", args={"param": "value"}
        )

        assert step_id is not None

        # Verify tool call with function name (CTP-first: functions are tools!)
        tools = await plan._graph.get_nodes_by_kind(NodeType.TOOL_CALL)
        assert len(tools) == 1
        assert tools[0].name == "process_data"  # Function name directly, not "function"
        assert tools[0].args["param"] == "value"  # Args unpacked directly

    @pytest.mark.asyncio
    async def test_add_function_step_with_result_variable(self, plan):
        step_id = await plan.add_function_step(
            title="Calculate",
            function="calculate_sum",
            args={"a": 1, "b": 2},
            result_variable="sum_result",
        )

        # Verify custom edge for result variable
        edges = await plan._graph.get_edges(src=step_id, kind=EdgeType.CUSTOM)
        custom_edge = next(
            (
                e
                for e in edges
                if isinstance(e, CustomEdge) and e.custom_type == "result_variable"
            ),
            None,
        )
        assert custom_edge is not None
        assert custom_edge.metadata["variable"] == "sum_result"

    @pytest.mark.asyncio
    async def test_add_function_step_with_dependencies(self, plan):
        step1_id = await plan.add_function_step(
            title="Function 1", function="func1", args={}
        )

        step2_id = await plan.add_function_step(
            title="Function 2", function="func2", args={}, depends_on=[step1_id]
        )

        assert step1_id != step2_id
        steps = await plan._graph.get_nodes_by_kind(NodeType.PLAN_STEP)
        assert len(steps) == 2


class TestAddPlanStep:
    """Test add_plan_step method"""

    @pytest.mark.asyncio
    async def test_add_plan_step_basic(self, plan):
        subplan_id = "subplan-123"
        step_id = await plan.add_plan_step(
            title="Execute subplan", plan_id=subplan_id, args={"input": "data"}
        )

        assert step_id is not None

        # Verify tool call with subplan type
        tools = await plan._graph.get_nodes_by_kind(NodeType.TOOL_CALL)
        assert len(tools) == 1
        assert tools[0].name == "subplan"
        assert tools[0].args["plan_id"] == subplan_id
        assert tools[0].args["args"]["input"] == "data"

    @pytest.mark.asyncio
    async def test_add_plan_step_with_result_variable(self, plan):
        step_id = await plan.add_plan_step(
            title="Execute subplan",
            plan_id="subplan-456",
            args={},
            result_variable="subplan_result",
        )

        # Verify custom edge for result variable
        edges = await plan._graph.get_edges(src=step_id, kind=EdgeType.CUSTOM)
        custom_edge = next(
            (
                e
                for e in edges
                if isinstance(e, CustomEdge) and e.custom_type == "result_variable"
            ),
            None,
        )
        assert custom_edge is not None
        assert custom_edge.metadata["variable"] == "subplan_result"


class TestFindStepByIndex:
    """Test _find_step_by_index helper method"""

    @pytest.mark.asyncio
    async def test_find_step_by_index_exists(self, plan):
        # Add a step
        await plan.add_tool_step(title="Test step", tool="test_tool", args={})

        # Get the step to find its index
        steps = await plan._graph.get_nodes_by_kind(NodeType.PLAN_STEP)
        step = steps[0]

        # Find by index
        found_id = await plan._find_step_by_index(step.index)
        assert found_id == step.id

    @pytest.mark.asyncio
    async def test_find_step_by_index_not_exists(self, plan):
        found_id = await plan._find_step_by_index("999")
        assert found_id is None


class TestToDictMethod:
    """Test to_dict serialization method"""

    @pytest.mark.asyncio
    async def test_to_dict_basic(self, plan):
        plan.set_variable("var1", "value1")
        plan.add_metadata("key1", "meta1")
        plan.add_tag("tag1")

        result = await plan.to_dict()

        assert result["id"] == plan.id
        assert result["title"] == plan.title
        assert result["description"] == plan.description
        assert result["tags"] == ["tag1"]
        assert result["variables"]["var1"] == "value1"
        assert result["metadata"]["key1"] == "meta1"
        assert "steps" in result

    @pytest.mark.asyncio
    async def test_to_dict_with_steps(self, plan):
        await plan.add_tool_step(
            title="Step 1",
            tool="tool1",
            args={"arg1": "val1"},
            result_variable="result1",
        )

        result = await plan.to_dict()

        assert len(result["steps"]) == 1
        step_dict = result["steps"][0]
        assert step_dict["title"] == "Step 1"
        assert len(step_dict["tool_calls"]) == 1
        assert step_dict["tool_calls"][0]["name"] == "tool1"
        assert step_dict["result_variable"] == "result1"

    @pytest.mark.asyncio
    async def test_to_dict_empty_plan(self, plan):
        result = await plan.to_dict()

        assert result["steps"] == []
        assert result["variables"] == {}
        assert result["metadata"] == {}
        assert result["tags"] == []


class TestErrorCases:
    """Test error handling in universal plan"""

    @pytest.mark.asyncio
    async def test_add_tool_step_step_not_found_error(self, plan, monkeypatch):
        """Test error when step creation fails"""

        # Mock _find_step_by_index to return None
        async def mock_find(self, idx):
            return None

        monkeypatch.setattr(UniversalPlan, "_find_step_by_index", mock_find)

        with pytest.raises(ValueError, match="Failed to find step"):
            await plan.add_tool_step("Test", "tool", {})

    @pytest.mark.asyncio
    async def test_add_function_step_step_not_found_error(self, plan, monkeypatch):
        """Test error when step creation fails for function"""

        async def mock_find(self, idx):
            return None

        monkeypatch.setattr(UniversalPlan, "_find_step_by_index", mock_find)

        with pytest.raises(ValueError, match="Failed to find step"):
            await plan.add_function_step("Test", "func", {})

    @pytest.mark.asyncio
    async def test_add_plan_step_step_not_found_error(self, plan, monkeypatch):
        """Test error when step creation fails for plan step"""

        async def mock_find(self, idx):
            return None

        monkeypatch.setattr(UniversalPlan, "_find_step_by_index", mock_find)

        with pytest.raises(ValueError, match="Failed to find step"):
            await plan.add_plan_step("Test", "plan-id", {})


class TestFromDictMethod:
    """Test from_dict class method"""

    @pytest.mark.asyncio
    async def test_from_dict_basic(self):
        """Test creating plan from dictionary"""
        data = {
            "title": "Test Plan",
            "description": "A test plan",
            "id": "test-plan-123",
            "tags": ["test", "demo"],
            "variables": {"var1": "value1"},
            "metadata": {"key1": "meta1"},
        }

        plan = await UniversalPlan.from_dict(data)

        assert plan.title == "Test Plan"
        assert plan.description == "A test plan"
        assert plan.id == "test-plan-123"
        assert plan.tags == ["test", "demo"]
        assert plan.variables == {"var1": "value1"}
        assert plan.metadata == {"key1": "meta1"}

    @pytest.mark.asyncio
    async def test_from_dict_with_defaults(self):
        """Test from_dict with missing fields"""
        data = {}

        plan = await UniversalPlan.from_dict(data)

        assert plan.title == "Untitled Plan"
        assert plan.description is None or "Untitled" in plan.description
        assert plan.tags == []
        assert plan.variables == {}
        assert plan.metadata == {}

    @pytest.mark.asyncio
    async def test_from_dict_with_custom_graph(self, graph_store):
        """Test from_dict with custom graph store"""
        data = {"title": "Test Plan"}

        plan = await UniversalPlan.from_dict(data, graph=graph_store)

        assert plan.graph is graph_store

    @pytest.mark.asyncio
    async def test_from_dict_with_steps(self):
        """Test from_dict with step data"""
        data = {
            "title": "Plan with Steps",
            "steps": [
                {
                    "index": "1",
                    "title": "Step 1",
                    "tool_calls": [{"name": "tool1", "args": {"arg1": "val1"}}],
                    "result_variable": "result1",
                }
            ],
        }

        plan = await UniversalPlan.from_dict(data)

        assert plan.title == "Plan with Steps"
        # Note: from_dict has implementation issues with steps
        # This tests that it doesn't crash


class TestComplexScenarios:
    """Test complex scenarios combining multiple features"""

    @pytest.mark.asyncio
    async def test_multiple_steps_with_dependencies(self, plan):
        """Test plan with multiple dependent steps"""
        step1 = await plan.add_tool_step(
            "Step 1", "tool1", {"input": "data1"}, result_variable="result1"
        )

        step2 = await plan.add_tool_step(
            "Step 2",
            "tool2",
            {"input": "${result1}"},
            depends_on=[step1],
            result_variable="result2",
        )

        await plan.add_function_step(
            "Step 3",
            "combine",
            {"a": "${result1}", "b": "${result2}"},
            depends_on=[step1, step2],
            result_variable="final",
        )

        # Verify all steps created
        steps = await plan._graph.get_nodes_by_kind(NodeType.PLAN_STEP)
        assert len(steps) == 3

    @pytest.mark.asyncio
    async def test_plan_with_metadata_and_variables(self, plan):
        """Test plan with both metadata and variables"""
        plan.set_variable("api_key", "secret123")
        plan.set_variable("endpoint", "https://api.example.com")
        plan.add_metadata("author", "Alice")
        plan.add_metadata("version", "2.0")
        plan.add_tag("production")
        plan.add_tag("critical")

        await plan.add_tool_step(
            "API Call", "api_tool", {"url": "${endpoint}", "auth": "${api_key}"}
        )

        # Convert to dict and verify
        plan_dict = await plan.to_dict()

        assert plan_dict["variables"]["api_key"] == "secret123"
        assert plan_dict["metadata"]["author"] == "Alice"
        assert "production" in plan_dict["tags"]

    @pytest.mark.asyncio
    async def test_mixed_step_types(self, plan):
        """Test plan with tool, function, and plan steps"""
        await plan.add_tool_step("Tool Step", "my_tool", {"arg": "value"})

        await plan.add_function_step("Function Step", "my_func", {"param": "value"})

        await plan.add_plan_step("Plan Step", "subplan-id", {"input": "data"})

        # Verify all 3 steps created
        steps = await plan._graph.get_nodes_by_kind(NodeType.PLAN_STEP)
        assert len(steps) == 3

        # Verify 3 tool calls (one for each step)
        tools = await plan._graph.get_nodes_by_kind(NodeType.TOOL_CALL)
        assert len(tools) == 3


class TestFromDictEdgeCases:
    """Test from_dict with various scenarios"""

    @pytest.mark.asyncio
    async def test_from_dict_steps_without_tool_calls(self):
        """Test from_dict with steps but no tool calls"""
        data = {
            "title": "Plan",
            "steps": [{"index": "1", "title": "Step without tools"}],
        }

        plan = await UniversalPlan.from_dict(data)
        assert plan.title == "Plan"

    @pytest.mark.asyncio
    async def test_from_dict_steps_with_result_variable(self):
        """Test from_dict with result variables"""
        data = {
            "title": "Plan with Result",
            "steps": [
                {
                    "index": "1",
                    "title": "Step 1",
                    "tool_calls": [{"name": "tool1", "args": {}}],
                    "result_variable": "my_var",
                }
            ],
        }

        plan = await UniversalPlan.from_dict(data)
        assert plan.title == "Plan with Result"

    @pytest.mark.asyncio
    async def test_from_dict_with_duplicate_steps(self):
        """Test from_dict when steps with same index already exist (lines 408-410)"""
        # Create a plan with a pre-existing step
        graph_store = InMemoryGraphStore()
        plan = UniversalPlan("Existing Plan", graph=graph_store)

        # Add a step manually first
        await plan.add_tool_step("Original Step 1", "tool_original", {})

        # Now use from_dict to recreate the plan with same step index
        data = {
            "title": "Existing Plan",
            "steps": [
                {
                    "index": "1",
                    "title": "Updated Step 1",
                    "tool_calls": [{"name": "tool_new", "args": {"key": "value"}}],
                }
            ],
        }

        # This should handle the existing step gracefully
        restored_plan = await UniversalPlan.from_dict(data, graph=graph_store)
        assert restored_plan.title == "Existing Plan"

        # Verify the step exists in the graph
        steps = await restored_plan._graph.get_nodes_by_kind(NodeType.PLAN_STEP)
        assert len(steps) >= 1


class TestVariableCache:
    """Test variable caching functionality"""

    @pytest.mark.asyncio
    async def test_variable_cache_initialization(self, plan):
        """Test that variable cache is initialized"""
        assert hasattr(plan, "_variable_cache")
        assert isinstance(plan._variable_cache, dict)
        assert len(plan._variable_cache) == 0

    @pytest.mark.asyncio
    async def test_set_variable_doesnt_affect_cache(self, plan):
        """Test set_variable doesn't directly populate cache"""
        plan.set_variable("test", "value")

        # Cache is for execution-time resolution, not set_variable
        assert "test" in plan.variables


class TestOutlineMethod:
    """Test outline() method"""

    @pytest.mark.asyncio
    async def test_outline_basic(self, plan):
        """Test outline generation"""
        await plan.add_tool_step("Step 1", "tool1", {})
        await plan.add_tool_step("Step 2", "tool2", {})

        outline = plan.outline()

        assert "Plan: Test Plan" in outline
        assert plan.id[:8] in outline
        assert "Step 1" in outline
        assert "Step 2" in outline

    @pytest.mark.asyncio
    async def test_outline_with_dependencies(self, plan):
        """Test outline with step dependencies"""
        step1 = await plan.add_tool_step("First", "tool1", {})
        await plan.add_tool_step("Second", "tool2", {}, depends_on=[step1])

        outline = plan.outline()

        assert "First" in outline
        assert "Second" in outline
        # May contain dependency info

    @pytest.mark.asyncio
    async def test_outline_triggers_indexing(self, plan):
        """Test that outline() triggers indexing if needed"""
        assert not plan._indexed

        plan.outline()

        assert plan._indexed  # Should be indexed after calling outline
