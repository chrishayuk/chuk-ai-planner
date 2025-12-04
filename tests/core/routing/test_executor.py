# tests/core/routing/test_executor.py
"""
Comprehensive tests for routing executor.

Tests all routing types (EXPRESSION, LLM, FUNCTION) and the FunctionRegistry.
Achieves 90%+ coverage of executor.py.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from chuk_ai_planner.core.routing.executor import (
    FunctionRegistry,
    RoutingExecutor,
    RoutingDecision,
)
from chuk_ai_planner.core.graph import RouterStep, RouteEdge
from chuk_ai_planner.core.graph.types import RouterType, EdgeType
from chuk_ai_planner.core.store.memory import InMemoryGraphStore


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def graph_store():
    """Create an in-memory graph store."""
    return InMemoryGraphStore()


@pytest.fixture
def function_registry():
    """Create a function registry."""
    return FunctionRegistry()


@pytest.fixture
def executor(graph_store, function_registry):
    """Create a routing executor."""
    return RoutingExecutor(graph_store, function_registry)


@pytest_asyncio.fixture
async def router_with_routes(graph_store):
    """
    Create a router step with route edges.

    Returns a tuple of (router, high_edge, low_edge).
    """
    router = RouterStep(
        id="router-1",
        router_type=RouterType.EXPRESSION,
        routes=["high", "low"],
        description="Score router",
        condition="${score} > 0.7",
        route_mapping={True: "high", False: "low"},
    )
    await graph_store.add_node(router)

    high_edge = RouteEdge(
        src=router.id,
        dst="step-high",
        route_key="high",
    )
    low_edge = RouteEdge(
        src=router.id,
        dst="step-low",
        route_key="low",
        is_default=True,
    )
    await graph_store.add_edge(high_edge)
    await graph_store.add_edge(low_edge)

    return router, high_edge, low_edge


# ==============================================================================
# FUNCTION REGISTRY TESTS
# ==============================================================================


class TestFunctionRegistry:
    """Test FunctionRegistry class."""

    def test_register_decorator(self, function_registry):
        """Test registering a function using decorator."""

        @function_registry.register("calculate_priority")
        def priority_func(context):
            return "high" if context.get("urgency", 0) > 7 else "low"

        # Should be registered
        assert function_registry.has("calculate_priority")

        # Should be callable
        result = function_registry.call("calculate_priority", {"urgency": 9})
        assert result == "high"

        result = function_registry.call("calculate_priority", {"urgency": 3})
        assert result == "low"

    def test_add_method(self, function_registry):
        """Test manually adding a function."""

        def classify(context):
            score = context.get("score", 0)
            if score > 0.8:
                return "excellent"
            elif score > 0.5:
                return "good"
            else:
                return "poor"

        function_registry.add("classify", classify)

        assert function_registry.has("classify")
        assert function_registry.call("classify", {"score": 0.9}) == "excellent"
        assert function_registry.call("classify", {"score": 0.6}) == "good"
        assert function_registry.call("classify", {"score": 0.3}) == "poor"

    def test_call_nonexistent_function(self, function_registry):
        """Test calling a function that doesn't exist."""
        with pytest.raises(ValueError, match="Function 'missing' not found"):
            function_registry.call("missing", {})

    def test_call_shows_available_functions(self, function_registry):
        """Test that error message shows available functions."""
        function_registry.add("func1", lambda ctx: "a")
        function_registry.add("func2", lambda ctx: "b")

        with pytest.raises(ValueError) as exc_info:
            function_registry.call("missing", {})

        error_msg = str(exc_info.value)
        assert "func1" in error_msg
        assert "func2" in error_msg

    def test_has_method(self, function_registry):
        """Test checking if a function exists."""
        assert not function_registry.has("test_func")

        function_registry.add("test_func", lambda ctx: "result")

        assert function_registry.has("test_func")

    def test_list_method(self, function_registry):
        """Test listing all registered functions."""
        assert function_registry.list() == []

        function_registry.add("func1", lambda ctx: "a")
        function_registry.add("func2", lambda ctx: "b")
        function_registry.add("func3", lambda ctx: "c")

        funcs = function_registry.list()
        assert set(funcs) == {"func1", "func2", "func3"}

    def test_function_receives_context(self, function_registry):
        """Test that registered functions receive the context."""
        captured_context = None

        def capture_context(context):
            nonlocal captured_context
            captured_context = context
            return "result"

        function_registry.add("capture", capture_context)

        test_context = {"key1": "value1", "key2": 42}
        function_registry.call("capture", test_context)

        assert captured_context == test_context


# ==============================================================================
# ROUTING EXECUTOR - INITIALIZATION
# ==============================================================================


class TestRoutingExecutorInit:
    """Test RoutingExecutor initialization."""

    def test_init_with_registry(self, graph_store, function_registry):
        """Test initializing with a function registry."""
        executor = RoutingExecutor(graph_store, function_registry)

        assert executor.graph is graph_store
        assert executor.function_registry is function_registry

    def test_init_without_registry(self, graph_store):
        """Test initializing without a function registry creates empty one."""
        executor = RoutingExecutor(graph_store)

        assert executor.graph is graph_store
        assert executor.function_registry is not None
        assert isinstance(executor.function_registry, FunctionRegistry)
        assert executor.function_registry.list() == []


# ==============================================================================
# ROUTING EXECUTOR - EXPRESSION ROUTING
# ==============================================================================


class TestExpressionRouting:
    """Test expression-based routing."""

    @pytest.mark.asyncio
    async def test_simple_greater_than(self, executor, router_with_routes):
        """Test simple > comparison."""
        router, high_edge, low_edge = router_with_routes

        # High score should route to "high"
        decision = await executor.evaluate_route(router, {"score": 0.85})

        assert decision.route_key == "high"
        assert decision.target_step_id == "step-high"
        assert decision.router_step_id == router.id
        assert decision.evaluation_method == "expression"
        assert decision.skipped_routes == ["low"]
        assert decision.evaluation_details["result"] is True

    @pytest.mark.asyncio
    async def test_simple_less_than(self, executor, router_with_routes):
        """Test condition evaluates to False."""
        router, high_edge, low_edge = router_with_routes

        # Low score should route to "low"
        decision = await executor.evaluate_route(router, {"score": 0.3})

        assert decision.route_key == "low"
        assert decision.target_step_id == "step-low"
        assert decision.evaluation_method == "expression"
        assert decision.skipped_routes == ["high"]
        assert decision.evaluation_details["result"] is False

    @pytest.mark.asyncio
    async def test_boundary_value(self, executor, router_with_routes):
        """Test boundary value (0.7)."""
        router, high_edge, low_edge = router_with_routes

        # Exactly 0.7 should be False (not > 0.7)
        decision = await executor.evaluate_route(router, {"score": 0.7})

        assert decision.route_key == "low"
        assert decision.evaluation_details["result"] is False

    @pytest.mark.asyncio
    async def test_greater_than_or_equal(self, executor, graph_store):
        """Test >= comparison."""
        router = RouterStep(
            id="router-gte",
            router_type=RouterType.EXPRESSION,
            routes=["pass", "fail"],
            condition="${score} >= 0.7",
            route_mapping={True: "pass", False: "fail"},
            description="Test router",
        )
        await graph_store.add_node(router)

        pass_edge = RouteEdge(src=router.id, dst="step-pass", route_key="pass")
        fail_edge = RouteEdge(src=router.id, dst="step-fail", route_key="fail")
        await graph_store.add_edge(pass_edge)
        await graph_store.add_edge(fail_edge)

        # 0.7 should pass with >=
        decision = await executor.evaluate_route(router, {"score": 0.7})
        assert decision.route_key == "pass"
        assert decision.evaluation_details["result"] is True

    @pytest.mark.asyncio
    async def test_less_than_or_equal(self, executor, graph_store):
        """Test <= comparison."""
        router = RouterStep(
            id="router-lte",
            router_type=RouterType.EXPRESSION,
            routes=["low", "high"],
            condition="${count} <= 10",
            route_mapping={True: "low", False: "high"},
            description="Test router",
        )
        await graph_store.add_node(router)

        low_edge = RouteEdge(src=router.id, dst="step-low", route_key="low")
        high_edge = RouteEdge(src=router.id, dst="step-high", route_key="high")
        await graph_store.add_edge(low_edge)
        await graph_store.add_edge(high_edge)

        decision = await executor.evaluate_route(router, {"count": 10})
        assert decision.route_key == "low"

        decision = await executor.evaluate_route(router, {"count": 11})
        assert decision.route_key == "high"

    @pytest.mark.asyncio
    async def test_equality_comparison(self, executor, graph_store):
        """Test == comparison."""
        router = RouterStep(
            id="router-eq",
            router_type=RouterType.EXPRESSION,
            routes=["match", "nomatch"],
            condition="${status} == 'ready'",
            route_mapping={True: "match", False: "nomatch"},
            description="Test router",
        )
        await graph_store.add_node(router)

        match_edge = RouteEdge(src=router.id, dst="step-match", route_key="match")
        nomatch_edge = RouteEdge(src=router.id, dst="step-nomatch", route_key="nomatch")
        await graph_store.add_edge(match_edge)
        await graph_store.add_edge(nomatch_edge)

        decision = await executor.evaluate_route(router, {"status": "ready"})
        assert decision.route_key == "match"

        decision = await executor.evaluate_route(router, {"status": "pending"})
        assert decision.route_key == "nomatch"

    @pytest.mark.asyncio
    async def test_nested_variable_access(self, executor, graph_store):
        """Test nested variable access like ${result.quality_score}."""
        router = RouterStep(
            id="router-nested",
            router_type=RouterType.EXPRESSION,
            routes=["high", "low"],
            condition="${result.quality_score} > 8",
            route_mapping={True: "high", False: "low"},
            description="Test router",
        )
        await graph_store.add_node(router)

        high_edge = RouteEdge(src=router.id, dst="step-high", route_key="high")
        low_edge = RouteEdge(src=router.id, dst="step-low", route_key="low")
        await graph_store.add_edge(high_edge)
        await graph_store.add_edge(low_edge)

        context = {"result": {"quality_score": 9.5, "completeness": 0.8}}

        decision = await executor.evaluate_route(router, context)
        assert decision.route_key == "high"
        assert decision.evaluation_details["result"] is True

    @pytest.mark.asyncio
    async def test_nested_object_attribute_access(self, executor, graph_store):
        """Test nested access on objects with attributes."""
        router = RouterStep(
            id="router-obj",
            router_type=RouterType.EXPRESSION,
            routes=["high", "low"],
            condition="${obj.score} > 5",
            route_mapping={True: "high", False: "low"},
            description="Test router",
        )
        await graph_store.add_node(router)

        high_edge = RouteEdge(src=router.id, dst="step-high", route_key="high")
        low_edge = RouteEdge(src=router.id, dst="step-low", route_key="low")
        await graph_store.add_edge(high_edge)
        await graph_store.add_edge(low_edge)

        # Create an object with attribute access
        class ScoreObject:
            def __init__(self, score):
                self.score = score

        context = {"obj": ScoreObject(7)}

        decision = await executor.evaluate_route(router, context)
        assert decision.route_key == "high"

    @pytest.mark.asyncio
    async def test_missing_variable_uses_original_placeholder(
        self, executor, graph_store
    ):
        """Test that missing variables keep the ${} placeholder."""
        router = RouterStep(
            id="router-missing",
            router_type=RouterType.EXPRESSION,
            routes=["yes", "no"],
            condition="${missing_var} > 0",
            route_mapping={True: "yes", False: "no"},
            description="Test router",
        )
        await graph_store.add_node(router)

        yes_edge = RouteEdge(src=router.id, dst="step-yes", route_key="yes")
        await graph_store.add_edge(yes_edge)

        # Should fail to evaluate because variable is missing
        with pytest.raises(ValueError, match="Failed to evaluate condition"):
            await executor.evaluate_route(router, {})

    @pytest.mark.asyncio
    async def test_route_mapping_with_string_boolean(self, executor, graph_store):
        """Test route_mapping with string 'true'/'false' keys."""
        router = RouterStep(
            id="router-str-bool",
            router_type=RouterType.EXPRESSION,
            routes=["yes", "no"],
            condition="${enabled} == 'yes'",
            route_mapping={"true": "yes", "false": "no"},
            description="Test router",
        )
        await graph_store.add_node(router)

        yes_edge = RouteEdge(src=router.id, dst="step-yes", route_key="yes")
        no_edge = RouteEdge(src=router.id, dst="step-no", route_key="no")
        await graph_store.add_edge(yes_edge)
        await graph_store.add_edge(no_edge)

        # True should map to "true" string
        decision = await executor.evaluate_route(router, {"enabled": "yes"})
        assert decision.route_key == "yes"

        decision = await executor.evaluate_route(router, {"enabled": "no"})
        assert decision.route_key == "no"

    @pytest.mark.asyncio
    async def test_no_route_mapping_uses_direct_result(self, executor, graph_store):
        """Test that without route_mapping, result is used directly."""
        router = RouterStep(
            id="router-direct",
            router_type=RouterType.EXPRESSION,
            routes=["True", "False"],
            condition="${value} > 0",
            # No route_mapping
            description="Test router",
        )
        await graph_store.add_node(router)

        true_edge = RouteEdge(src=router.id, dst="step-true", route_key="True")
        false_edge = RouteEdge(src=router.id, dst="step-false", route_key="False")
        await graph_store.add_edge(true_edge)
        await graph_store.add_edge(false_edge)

        decision = await executor.evaluate_route(router, {"value": 5})
        assert decision.route_key == "True"

    @pytest.mark.asyncio
    async def test_default_route_fallback(self, executor, graph_store):
        """Test that default route is used when no match found."""
        router = RouterStep(
            id="router-default",
            router_type=RouterType.EXPRESSION,
            routes=["specific", "default"],
            condition="${value}",  # Will return the actual value
            description="Test router",
        )
        await graph_store.add_node(router)

        specific_edge = RouteEdge(
            src=router.id, dst="step-specific", route_key="specific"
        )
        default_edge = RouteEdge(
            src=router.id, dst="step-default", route_key="default", is_default=True
        )
        await graph_store.add_edge(specific_edge)
        await graph_store.add_edge(default_edge)

        # "unknown" doesn't match "specific", should use default
        decision = await executor.evaluate_route(router, {"value": "unknown"})
        assert decision.route_key == "default"
        assert decision.target_step_id == "step-default"

    @pytest.mark.asyncio
    async def test_missing_condition_raises_error(self, executor, graph_store):
        """Test that missing condition raises error."""
        router = RouterStep(
            id="router-no-condition",
            router_type=RouterType.EXPRESSION,
            routes=["a", "b"],
            # No condition
            description="Test router",
        )
        await graph_store.add_node(router)

        with pytest.raises(ValueError, match="missing 'condition'"):
            await executor.evaluate_route(router, {})

    @pytest.mark.asyncio
    async def test_invalid_expression_raises_error(self, executor, graph_store):
        """Test that invalid expression raises error."""
        router = RouterStep(
            id="router-invalid",
            router_type=RouterType.EXPRESSION,
            routes=["a", "b"],
            condition="invalid python syntax !!!",
            description="Test router",
        )
        await graph_store.add_node(router)

        a_edge = RouteEdge(src=router.id, dst="step-a", route_key="a")
        await graph_store.add_edge(a_edge)

        with pytest.raises(ValueError, match="Failed to evaluate condition"):
            await executor.evaluate_route(router, {})

    @pytest.mark.asyncio
    async def test_no_matching_route_raises_error(self, executor, graph_store):
        """Test that no matching route raises error."""
        router = RouterStep(
            id="router-nomatch",
            router_type=RouterType.EXPRESSION,
            routes=["high", "low"],
            condition="${value}",
            description="Test router",
        )
        await graph_store.add_node(router)

        high_edge = RouteEdge(src=router.id, dst="step-high", route_key="high")
        low_edge = RouteEdge(src=router.id, dst="step-low", route_key="low")
        await graph_store.add_edge(high_edge)
        await graph_store.add_edge(low_edge)

        # "medium" doesn't match any route and no default
        with pytest.raises(ValueError, match="No route found"):
            await executor.evaluate_route(router, {"value": "medium"})

    @pytest.mark.asyncio
    async def test_evaluation_details_included(self, executor, router_with_routes):
        """Test that evaluation details are included in decision."""
        router, high_edge, low_edge = router_with_routes

        decision = await executor.evaluate_route(router, {"score": 0.9})

        assert "condition" in decision.evaluation_details
        assert decision.evaluation_details["condition"] == "${score} > 0.7"

        assert "resolved_condition" in decision.evaluation_details
        assert decision.evaluation_details["resolved_condition"] == "0.9 > 0.7"

        assert "result" in decision.evaluation_details
        assert decision.evaluation_details["result"] is True

    @pytest.mark.asyncio
    async def test_edge_without_route_key_is_skipped(self, executor, graph_store):
        """Test that edges without route_key attribute are skipped."""
        from chuk_ai_planner.core.graph import NextEdge

        router = RouterStep(
            id="router-mixed",
            router_type=RouterType.EXPRESSION,
            routes=["yes", "no"],
            condition="${value}",
            description="Test router",
        )
        await graph_store.add_node(router)

        # Add a regular NextEdge (no route_key attribute)
        next_edge = NextEdge(src=router.id, dst="step-next")
        await graph_store.add_edge(next_edge)

        # Add a proper RouteEdge
        yes_edge = RouteEdge(
            src=router.id, dst="step-yes", route_key="yes", is_default=True
        )
        await graph_store.add_edge(yes_edge)

        # Even though there's a NextEdge, it should be ignored
        decision = await executor.evaluate_route(router, {"value": "yes"})
        assert decision.route_key == "yes"


# ==============================================================================
# ROUTING EXECUTOR - FUNCTION ROUTING
# ==============================================================================


class TestFunctionRouting:
    """Test function-based routing."""

    @pytest.mark.asyncio
    async def test_string_function_name_lookup(
        self, executor, graph_store, function_registry
    ):
        """Test function routing with string function name."""

        # Register a routing function
        @function_registry.register("priority_router")
        def route_by_priority(context):
            urgency = context.get("urgency", 0)
            if urgency > 7:
                return "urgent"
            else:
                return "normal"

        router = RouterStep(
            id="router-func",
            router_type=RouterType.FUNCTION,
            routes=["urgent", "normal"],
            router_function="priority_router",
            description="Test router",
        )
        await graph_store.add_node(router)

        urgent_edge = RouteEdge(src=router.id, dst="step-urgent", route_key="urgent")
        normal_edge = RouteEdge(src=router.id, dst="step-normal", route_key="normal")
        await graph_store.add_edge(urgent_edge)
        await graph_store.add_edge(normal_edge)

        # High urgency
        decision = await executor.evaluate_route(router, {"urgency": 9})
        assert decision.route_key == "urgent"
        assert decision.target_step_id == "step-urgent"
        assert decision.evaluation_method == "function"
        assert decision.evaluation_details["function_result"] == "urgent"

        # Low urgency
        decision = await executor.evaluate_route(router, {"urgency": 3})
        assert decision.route_key == "normal"
        assert decision.target_step_id == "step-normal"

    @pytest.mark.asyncio
    async def test_function_routing_multiple_outcomes(
        self, executor, graph_store, function_registry
    ):
        """Test function routing with multiple possible outcomes."""

        def quality_router(context):
            score = context.get("quality", 0)
            if score > 8:
                return "excellent"
            elif score > 5:
                return "good"
            else:
                return "poor"

        function_registry.add("quality_router", quality_router)

        router = RouterStep(
            id="router-quality",
            router_type=RouterType.FUNCTION,
            routes=["excellent", "good", "poor"],
            router_function="quality_router",
            description="Test router",
        )
        await graph_store.add_node(router)

        excellent_edge = RouteEdge(
            src=router.id, dst="step-excellent", route_key="excellent"
        )
        good_edge = RouteEdge(src=router.id, dst="step-good", route_key="good")
        poor_edge = RouteEdge(src=router.id, dst="step-poor", route_key="poor")
        await graph_store.add_edge(excellent_edge)
        await graph_store.add_edge(good_edge)
        await graph_store.add_edge(poor_edge)

        decision = await executor.evaluate_route(router, {"quality": 9})
        assert decision.route_key == "excellent"

        decision = await executor.evaluate_route(router, {"quality": 7})
        assert decision.route_key == "good"

        decision = await executor.evaluate_route(router, {"quality": 3})
        assert decision.route_key == "poor"

    @pytest.mark.asyncio
    async def test_function_not_found_raises_error(
        self, executor, graph_store, function_registry
    ):
        """Test that missing function raises error."""
        function_registry.add("func1", lambda ctx: "a")
        function_registry.add("func2", lambda ctx: "b")

        router = RouterStep(
            id="router-missing-func",
            router_type=RouterType.FUNCTION,
            routes=["a", "b"],
            router_function="nonexistent_function",
            description="Test router",
        )
        await graph_store.add_node(router)

        a_edge = RouteEdge(src=router.id, dst="step-a", route_key="a")
        await graph_store.add_edge(a_edge)

        with pytest.raises(
            ValueError, match="Function 'nonexistent_function' not found"
        ):
            await executor.evaluate_route(router, {})

    @pytest.mark.asyncio
    async def test_function_error_shows_available(
        self, executor, graph_store, function_registry
    ):
        """Test that error shows available functions."""
        function_registry.add("router1", lambda ctx: "a")
        function_registry.add("router2", lambda ctx: "b")

        router = RouterStep(
            id="router-err",
            router_type=RouterType.FUNCTION,
            routes=["a", "b"],
            router_function="missing",
            description="Test router",
        )
        await graph_store.add_node(router)

        a_edge = RouteEdge(src=router.id, dst="step-a", route_key="a")
        await graph_store.add_edge(a_edge)

        with pytest.raises(ValueError) as exc_info:
            await executor.evaluate_route(router, {})

        error_msg = str(exc_info.value)
        assert "router1" in error_msg
        assert "router2" in error_msg

    @pytest.mark.asyncio
    async def test_function_execution_failure(
        self, executor, graph_store, function_registry
    ):
        """Test that function execution errors are handled."""

        def failing_function(context):
            raise RuntimeError("Function failed!")

        function_registry.add("failing", failing_function)

        router = RouterStep(
            id="router-fail",
            router_type=RouterType.FUNCTION,
            routes=["a", "b"],
            router_function="failing",
            description="Test router",
        )
        await graph_store.add_node(router)

        a_edge = RouteEdge(src=router.id, dst="step-a", route_key="a")
        await graph_store.add_edge(a_edge)

        with pytest.raises(ValueError, match="Router function 'failing' failed"):
            await executor.evaluate_route(router, {})

    @pytest.mark.asyncio
    async def test_function_context_passed_correctly(
        self, executor, graph_store, function_registry
    ):
        """Test that context is passed correctly to functions."""
        captured_context = {}

        def context_capture(context):
            captured_context.update(context)
            return "route_a"

        function_registry.add("context_capture", context_capture)

        router = RouterStep(
            id="router-context",
            router_type=RouterType.FUNCTION,
            routes=["route_a", "route_b"],
            router_function="context_capture",
            description="Test router",
        )
        await graph_store.add_node(router)

        a_edge = RouteEdge(src=router.id, dst="step-a", route_key="route_a")
        b_edge = RouteEdge(src=router.id, dst="step-b", route_key="route_b")
        await graph_store.add_edge(a_edge)
        await graph_store.add_edge(b_edge)

        test_context = {"key1": "value1", "key2": 42}
        await executor.evaluate_route(router, test_context)

        assert captured_context == test_context

    @pytest.mark.asyncio
    async def test_default_route_fallback(
        self, executor, graph_store, function_registry
    ):
        """Test that default route is used when result doesn't match."""
        function_registry.add("router", lambda ctx: "unknown_result")

        router = RouterStep(
            id="router-default",
            router_type=RouterType.FUNCTION,
            routes=["known", "default"],
            router_function="router",
            description="Test router",
        )
        await graph_store.add_node(router)

        known_edge = RouteEdge(src=router.id, dst="step-known", route_key="known")
        default_edge = RouteEdge(
            src=router.id, dst="step-default", route_key="default", is_default=True
        )
        await graph_store.add_edge(known_edge)
        await graph_store.add_edge(default_edge)

        decision = await executor.evaluate_route(router, {})
        assert decision.route_key == "default"
        assert decision.target_step_id == "step-default"

    @pytest.mark.asyncio
    async def test_no_matching_route_raises_error(
        self, executor, graph_store, function_registry
    ):
        """Test that no matching route (and no default) raises error."""
        function_registry.add("router", lambda ctx: "unexpected")

        router = RouterStep(
            id="router-nomatch",
            router_type=RouterType.FUNCTION,
            routes=["a", "b"],
            router_function="router",
            description="Test router",
        )
        await graph_store.add_node(router)

        a_edge = RouteEdge(src=router.id, dst="step-a", route_key="a")
        b_edge = RouteEdge(src=router.id, dst="step-b", route_key="b")
        await graph_store.add_edge(a_edge)
        await graph_store.add_edge(b_edge)

        with pytest.raises(
            ValueError, match="No route found for function result 'unexpected'"
        ):
            await executor.evaluate_route(router, {})

    @pytest.mark.asyncio
    async def test_missing_router_function_raises_error(self, executor, graph_store):
        """Test that missing router_function raises error."""
        router = RouterStep(
            id="router-no-func",
            router_type=RouterType.FUNCTION,
            routes=["a", "b"],
            # No router_function
            description="Test router",
        )
        await graph_store.add_node(router)

        with pytest.raises(ValueError, match="missing 'router_function'"):
            await executor.evaluate_route(router, {})

    @pytest.mark.asyncio
    async def test_function_skipped_routes_calculation(
        self, executor, graph_store, function_registry
    ):
        """Test that skipped_routes is properly calculated for function routing."""
        function_registry.add("chooser", lambda ctx: "b")

        router = RouterStep(
            id="router-func-skipped",
            router_type=RouterType.FUNCTION,
            routes=["a", "b", "c", "d"],
            router_function="chooser",
            description="Test router",
        )
        await graph_store.add_node(router)

        a_edge = RouteEdge(src=router.id, dst="step-a", route_key="a")
        b_edge = RouteEdge(src=router.id, dst="step-b", route_key="b")
        c_edge = RouteEdge(src=router.id, dst="step-c", route_key="c")
        d_edge = RouteEdge(src=router.id, dst="step-d", route_key="d")
        await graph_store.add_edge(a_edge)
        await graph_store.add_edge(b_edge)
        await graph_store.add_edge(c_edge)
        await graph_store.add_edge(d_edge)

        decision = await executor.evaluate_route(router, {})

        assert decision.route_key == "b"
        assert set(decision.skipped_routes) == {"a", "c", "d"}


# ==============================================================================
# ROUTING EXECUTOR - LLM ROUTING
# ==============================================================================


class TestLLMRouting:
    """Test LLM-based routing."""

    @pytest.mark.asyncio
    async def test_basic_llm_route_selection(self, executor, graph_store):
        """Test basic LLM routing."""
        router = RouterStep(
            id="router-llm",
            router_type=RouterType.LLM,
            routes=["technical", "creative", "balanced"],
            llm_prompt="What writing style fits this content?",
            description="Test router",
        )
        await graph_store.add_node(router)

        tech_edge = RouteEdge(src=router.id, dst="step-tech", route_key="technical")
        creative_edge = RouteEdge(
            src=router.id, dst="step-creative", route_key="creative"
        )
        balanced_edge = RouteEdge(
            src=router.id, dst="step-balanced", route_key="balanced"
        )
        await graph_store.add_edge(tech_edge)
        await graph_store.add_edge(creative_edge)
        await graph_store.add_edge(balanced_edge)

        # Mock the LLM call
        with patch.object(executor, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "technical"

            decision = await executor.evaluate_route(
                router, {"content": "API documentation"}
            )

            assert decision.route_key == "technical"
            assert decision.target_step_id == "step-tech"
            assert decision.evaluation_method == "llm"
            assert "llm_response" in decision.evaluation_details
            assert decision.evaluation_details["llm_response"] == "technical"

            # Verify LLM was called with correct prompts
            mock_llm.assert_called_once()
            call_args = mock_llm.call_args[0]
            system_prompt = call_args[0]
            user_prompt = call_args[1]

            assert "technical" in system_prompt
            assert "creative" in system_prompt
            assert "balanced" in system_prompt
            assert "What writing style" in user_prompt

    @pytest.mark.asyncio
    async def test_llm_response_case_insensitive_matching(self, executor, graph_store):
        """Test that LLM response matching is case-insensitive."""
        router = RouterStep(
            id="router-llm-case",
            router_type=RouterType.LLM,
            routes=["high", "medium", "low"],
            llm_prompt="What priority?",
            description="Test router",
        )
        await graph_store.add_node(router)

        high_edge = RouteEdge(src=router.id, dst="step-high", route_key="high")
        medium_edge = RouteEdge(src=router.id, dst="step-medium", route_key="medium")
        low_edge = RouteEdge(src=router.id, dst="step-low", route_key="low")
        await graph_store.add_edge(high_edge)
        await graph_store.add_edge(medium_edge)
        await graph_store.add_edge(low_edge)

        with patch.object(executor, "_call_llm", new_callable=AsyncMock) as mock_llm:
            # LLM returns "HIGH" but should match "high"
            mock_llm.return_value = "HIGH"

            decision = await executor.evaluate_route(router, {})
            assert decision.route_key == "high"

    @pytest.mark.asyncio
    async def test_llm_response_partial_match(self, executor, graph_store):
        """Test that partial match in LLM response works."""
        router = RouterStep(
            id="router-llm-partial",
            router_type=RouterType.LLM,
            routes=["approve", "reject"],
            llm_prompt="Should we approve?",
            description="Test router",
        )
        await graph_store.add_node(router)

        approve_edge = RouteEdge(src=router.id, dst="step-approve", route_key="approve")
        reject_edge = RouteEdge(src=router.id, dst="step-reject", route_key="reject")
        await graph_store.add_edge(approve_edge)
        await graph_store.add_edge(reject_edge)

        with patch.object(executor, "_call_llm", new_callable=AsyncMock) as mock_llm:
            # LLM returns "I would approve this" - should match "approve"
            mock_llm.return_value = "I would approve this"

            decision = await executor.evaluate_route(router, {})
            assert decision.route_key == "approve"

    @pytest.mark.asyncio
    async def test_llm_fallback_to_default_on_no_match(self, executor, graph_store):
        """Test fallback to default route when LLM response doesn't match."""
        router = RouterStep(
            id="router-llm-default",
            router_type=RouterType.LLM,
            routes=["specific", "default"],
            llm_prompt="Choose a route",
            description="Test router",
        )
        await graph_store.add_node(router)

        specific_edge = RouteEdge(
            src=router.id, dst="step-specific", route_key="specific"
        )
        default_edge = RouteEdge(
            src=router.id, dst="step-default", route_key="default", is_default=True
        )
        await graph_store.add_edge(specific_edge)
        await graph_store.add_edge(default_edge)

        with patch.object(executor, "_call_llm", new_callable=AsyncMock) as mock_llm:
            # LLM returns something that doesn't match
            mock_llm.return_value = "unexpected response"

            decision = await executor.evaluate_route(router, {})
            assert decision.route_key == "default"
            assert decision.target_step_id == "step-default"

    @pytest.mark.asyncio
    async def test_llm_picks_first_route_when_no_default(self, executor, graph_store):
        """Test that first route is picked when no match and no default."""
        router = RouterStep(
            id="router-llm-first",
            router_type=RouterType.LLM,
            routes=["first", "second", "third"],
            llm_prompt="Choose",
            description="Test router",
        )
        await graph_store.add_node(router)

        first_edge = RouteEdge(src=router.id, dst="step-first", route_key="first")
        second_edge = RouteEdge(src=router.id, dst="step-second", route_key="second")
        third_edge = RouteEdge(src=router.id, dst="step-third", route_key="third")
        await graph_store.add_edge(first_edge)
        await graph_store.add_edge(second_edge)
        await graph_store.add_edge(third_edge)

        with patch.object(executor, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "no match"

            decision = await executor.evaluate_route(router, {})
            assert decision.route_key == "first"

    @pytest.mark.asyncio
    async def test_llm_no_routes_available_raises_error(self, executor, graph_store):
        """Test that error is raised when no routes available."""
        router = RouterStep(
            id="router-llm-noroutes",
            router_type=RouterType.LLM,
            routes=["a", "b"],
            llm_prompt="Choose",
            description="Test router",
        )
        await graph_store.add_node(router)
        # No edges added!

        with patch.object(executor, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "a"

            with pytest.raises(ValueError, match="No routes available"):
                await executor.evaluate_route(router, {})

    @pytest.mark.asyncio
    async def test_llm_missing_prompt_raises_error(self, executor, graph_store):
        """Test that missing llm_prompt raises error."""
        router = RouterStep(
            id="router-llm-noprompt",
            router_type=RouterType.LLM,
            routes=["a", "b"],
            # No llm_prompt
            description="Test router",
        )
        await graph_store.add_node(router)

        with pytest.raises(ValueError, match="missing 'llm_prompt'"):
            await executor.evaluate_route(router, {})

    @pytest.mark.asyncio
    async def test_llm_context_passed_to_prompt(self, executor, graph_store):
        """Test that context is passed to LLM prompt."""
        router = RouterStep(
            id="router-llm-context",
            router_type=RouterType.LLM,
            routes=["urgent", "normal"],
            llm_prompt="What priority based on urgency?",
            description="Test router",
        )
        await graph_store.add_node(router)

        urgent_edge = RouteEdge(src=router.id, dst="step-urgent", route_key="urgent")
        normal_edge = RouteEdge(src=router.id, dst="step-normal", route_key="normal")
        await graph_store.add_edge(urgent_edge)
        await graph_store.add_edge(normal_edge)

        context = {"urgency": 10, "deadline": "today"}

        with patch.object(executor, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "urgent"

            await executor.evaluate_route(router, context)

            # Check that context was in the user prompt
            user_prompt = mock_llm.call_args[0][1]
            assert '"urgency": 10' in user_prompt
            assert '"deadline": "today"' in user_prompt

    @pytest.mark.asyncio
    async def test_llm_skipped_routes_calculation(self, executor, graph_store):
        """Test that skipped_routes is properly calculated for LLM routing."""
        router = RouterStep(
            id="router-llm-skipped",
            router_type=RouterType.LLM,
            routes=["a", "b", "c"],
            llm_prompt="Choose",
            description="Test router",
        )
        await graph_store.add_node(router)

        a_edge = RouteEdge(src=router.id, dst="step-a", route_key="a")
        b_edge = RouteEdge(src=router.id, dst="step-b", route_key="b")
        c_edge = RouteEdge(src=router.id, dst="step-c", route_key="c")
        await graph_store.add_edge(a_edge)
        await graph_store.add_edge(b_edge)
        await graph_store.add_edge(c_edge)

        with patch.object(executor, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "b"

            decision = await executor.evaluate_route(router, {})

            assert decision.route_key == "b"
            assert set(decision.skipped_routes) == {"a", "c"}


# ==============================================================================
# ROUTING EXECUTOR - UNKNOWN TYPE
# ==============================================================================


class TestUnknownRouterType:
    """Test handling of unknown router types."""

    @pytest.mark.asyncio
    async def test_unknown_router_type_raises_error(self, executor, graph_store):
        """Test that unknown router type raises error."""
        # Pydantic will validate the router_type, so we can't create an invalid one
        # This test demonstrates that Pydantic protects us from invalid router types
        with pytest.raises(ValueError, match="Input should be"):
            RouterStep(
                id="router-unknown",
                router_type="unknown_type",  # Invalid type - Pydantic will reject this
                routes=["a", "b"],
                description="Test router",
            )

    @pytest.mark.asyncio
    async def test_string_router_types_work(self, executor, graph_store):
        """Test that string versions of router types work."""
        # Test string "expression" instead of RouterType.EXPRESSION
        router = RouterStep(
            id="router-string-type",
            router_type="expression",  # String version
            routes=["yes", "no"],
            condition="${value} > 0",
            route_mapping={True: "yes", False: "no"},
            description="Test router",
        )
        await graph_store.add_node(router)

        yes_edge = RouteEdge(src=router.id, dst="step-yes", route_key="yes")
        no_edge = RouteEdge(src=router.id, dst="step-no", route_key="no")
        await graph_store.add_edge(yes_edge)
        await graph_store.add_edge(no_edge)

        decision = await executor.evaluate_route(router, {"value": 1})
        assert decision.route_key == "yes"


# ==============================================================================
# ROUTING EXECUTOR - HELPER METHODS
# ==============================================================================


class TestHelperMethods:
    """Test helper methods."""

    @pytest.mark.asyncio
    async def test_get_route_edges(self, executor, graph_store):
        """Test _get_route_edges helper method."""
        router_id = "router-test"

        edge1 = RouteEdge(src=router_id, dst="step-1", route_key="route1")
        edge2 = RouteEdge(src=router_id, dst="step-2", route_key="route2")
        edge3 = RouteEdge(src=router_id, dst="step-3", route_key="route3")

        await graph_store.add_edge(edge1)
        await graph_store.add_edge(edge2)
        await graph_store.add_edge(edge3)

        # Add some non-route edges that should be filtered out
        from chuk_ai_planner.core.graph import NextEdge

        next_edge = NextEdge(src=router_id, dst="other-step")
        await graph_store.add_edge(next_edge)

        edges = await executor._get_route_edges(router_id)

        assert len(edges) == 3
        assert all(e.kind == EdgeType.ROUTE for e in edges)

    def test_resolve_variables_simple(self, executor):
        """Test simple variable resolution."""
        context = {"score": 0.85, "count": 10}

        expr = "${score} > 0.7"
        resolved = executor._resolve_variables(expr, context)
        assert resolved == "0.85 > 0.7"

        expr = "${count} >= 10"
        resolved = executor._resolve_variables(expr, context)
        assert resolved == "10 >= 10"

    def test_resolve_variables_string_values(self, executor):
        """Test variable resolution with string values."""
        context = {"status": "ready", "mode": "production"}

        expr = "${status} == 'ready'"
        resolved = executor._resolve_variables(expr, context)
        # Strings get quoted
        assert resolved == "'ready' == 'ready'"

    def test_resolve_variables_nested_dict(self, executor):
        """Test nested dictionary access."""
        context = {"result": {"score": 9.5, "quality": "high"}}

        expr = "${result.score} > 8"
        resolved = executor._resolve_variables(expr, context)
        assert resolved == "9.5 > 8"

        expr = "${result.quality} == 'high'"
        resolved = executor._resolve_variables(expr, context)
        assert resolved == "'high' == 'high'"

    def test_resolve_variables_nested_object(self, executor):
        """Test nested object attribute access."""

        class Inner:
            def __init__(self):
                self.value = 42

        class Outer:
            def __init__(self):
                self.inner = Inner()

        context = {"obj": Outer()}

        expr = "${obj.inner.value} == 42"
        resolved = executor._resolve_variables(expr, context)
        assert resolved == "42 == 42"

    def test_resolve_variables_missing_variable(self, executor):
        """Test that missing variables keep placeholder."""
        context = {"existing": 5}

        expr = "${missing} > 0"
        resolved = executor._resolve_variables(expr, context)
        # Should keep the placeholder
        assert resolved == "${missing} > 0"

    def test_resolve_variables_missing_nested_property(self, executor):
        """Test missing nested property."""
        context = {"obj": {"a": 1}}

        expr = "${obj.b} > 0"
        resolved = executor._resolve_variables(expr, context)
        # Should keep the placeholder
        assert resolved == "${obj.b} > 0"

    def test_safe_eval_comparisons(self, executor):
        """Test safe evaluation of comparison expressions."""
        assert executor._safe_eval("5 > 3", {}) is True
        assert executor._safe_eval("5 < 3", {}) is False
        assert executor._safe_eval("5 >= 5", {}) is True
        assert executor._safe_eval("5 <= 5", {}) is True
        assert executor._safe_eval("5 == 5", {}) is True
        assert executor._safe_eval("5 != 3", {}) is True

    def test_safe_eval_arithmetic(self, executor):
        """Test safe evaluation with arithmetic."""
        assert executor._safe_eval("2 + 3", {}) == 5
        assert executor._safe_eval("10 - 3", {}) == 7
        assert executor._safe_eval("4 * 5", {}) == 20
        assert executor._safe_eval("10 / 2", {}) == 5.0

    def test_safe_eval_string_comparison(self, executor):
        """Test safe evaluation with strings."""
        assert executor._safe_eval("'hello' == 'hello'", {}) is True
        assert executor._safe_eval("'hello' != 'world'", {}) is True

    def test_safe_eval_invalid_syntax(self, executor):
        """Test that invalid syntax raises error."""
        with pytest.raises(ValueError, match="Invalid expression"):
            executor._safe_eval("invalid syntax !!!", {})

    def test_safe_eval_no_builtins(self, executor):
        """Test that built-in functions are not available."""
        # This should fail because print is not available
        with pytest.raises(ValueError):
            executor._safe_eval("print('hello')", {})


# ==============================================================================
# ROUTING DECISION DATACLASS
# ==============================================================================


class TestRoutingDecision:
    """Test RoutingDecision dataclass."""

    def test_create_basic_decision(self):
        """Test creating a basic routing decision."""
        decision = RoutingDecision(
            route_key="high",
            router_step_id="router-1",
            target_step_id="step-high",
            skipped_routes=["low", "medium"],
            evaluation_method="expression",
        )

        assert decision.route_key == "high"
        assert decision.router_step_id == "router-1"
        assert decision.target_step_id == "step-high"
        assert decision.skipped_routes == ["low", "medium"]
        assert decision.evaluation_method == "expression"
        assert decision.evaluation_details is None

    def test_create_decision_with_details(self):
        """Test creating a decision with evaluation details."""
        decision = RoutingDecision(
            route_key="urgent",
            router_step_id="router-2",
            target_step_id="step-urgent",
            skipped_routes=["normal"],
            evaluation_method="function",
            evaluation_details={"function_result": "urgent", "urgency": 9},
        )

        assert decision.evaluation_details["function_result"] == "urgent"
        assert decision.evaluation_details["urgency"] == 9
