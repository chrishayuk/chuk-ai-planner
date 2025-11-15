# tests/graph/test_types.py
"""
Tests for graph type enums.

Verifies that all enums are properly defined and have expected values.
"""


from chuk_ai_planner.graph.types import (
    NodeType,
    EdgeType,
    RouterType,
    StepStatus,
)


class TestNodeType:
    """Test NodeType enum."""

    def test_all_node_types_exist(self):
        """Test that all expected node types are defined."""
        assert NodeType.SESSION == "session"
        assert NodeType.PLAN == "plan"
        assert NodeType.PLAN_STEP == "plan_step"
        assert NodeType.ROUTER_STEP == "router_step"
        assert NodeType.TOOL_CALL == "tool_call"
        assert NodeType.TASK_RUN == "task_run"
        assert NodeType.SUMMARY == "summary"

    def test_node_type_count(self):
        """Test that we have the expected number of node types."""
        # Domain-agnostic types + workflow types (approval, artifact)
        assert len(NodeType) == 9

    def test_node_type_is_string_enum(self):
        """Test that NodeType is a string enum."""
        assert issubclass(NodeType, str)
        assert isinstance(NodeType.PLAN, str)

    def test_node_type_values_are_lowercase(self):
        """Test that all enum values are lowercase."""
        for node_type in NodeType:
            assert node_type.value == node_type.value.lower()


class TestEdgeType:
    """Test EdgeType enum."""

    def test_all_edge_types_exist(self):
        """Test that all expected edge types are defined."""
        assert EdgeType.PARENT_CHILD == "parent_child"
        assert EdgeType.NEXT == "next"
        assert EdgeType.PLAN_LINK == "plan_link"
        assert EdgeType.STEP_ORDER == "step_order"
        assert EdgeType.ROUTE == "route"
        assert EdgeType.CUSTOM == "custom"

    def test_edge_type_count(self):
        """Test that we have the expected number of edge types."""
        # Core types + workflow types (approval, fallback, artifact_dependency)
        assert len(EdgeType) == 9

    def test_edge_type_is_string_enum(self):
        """Test that EdgeType is a string enum."""
        assert issubclass(EdgeType, str)
        assert isinstance(EdgeType.ROUTE, str)

    def test_edge_type_values_are_lowercase(self):
        """Test that all enum values are lowercase."""
        for edge_type in EdgeType:
            assert edge_type.value == edge_type.value.lower()


class TestRouterType:
    """Test RouterType enum."""

    def test_all_router_types_exist(self):
        """Test that all expected router types are defined."""
        assert RouterType.EXPRESSION == "expression"
        assert RouterType.LLM == "llm"
        assert RouterType.FUNCTION == "function"

    def test_router_type_count(self):
        """Test that we have exactly 3 router types."""
        assert len(RouterType) == 3

    def test_router_type_is_string_enum(self):
        """Test that RouterType is a string enum."""
        assert issubclass(RouterType, str)
        assert isinstance(RouterType.EXPRESSION, str)


class TestStepStatus:
    """Test StepStatus enum."""

    def test_all_step_statuses_exist(self):
        """Test that all expected statuses are defined."""
        assert StepStatus.PENDING == "pending"
        assert StepStatus.RUNNING == "running"
        assert StepStatus.COMPLETED == "completed"
        assert StepStatus.FAILED == "failed"
        assert StepStatus.SKIPPED == "skipped"

    def test_step_status_count(self):
        """Test that we have all step statuses including workflow states."""
        # Core + workflow states (blocked, paused, waiting_approval, cancelled, timeout, retrying)
        assert len(StepStatus) == 11

    def test_step_status_is_string_enum(self):
        """Test that StepStatus is a string enum."""
        assert issubclass(StepStatus, str)
        assert isinstance(StepStatus.PENDING, str)

    def test_step_status_lifecycle(self):
        """Test that status values represent a logical lifecycle."""
        # Just verify the values exist in a sensible order
        statuses = [s.value for s in StepStatus]
        assert "pending" in statuses
        assert "running" in statuses
        assert "completed" in statuses
        assert "failed" in statuses
        assert "skipped" in statuses


class TestEnumComparisons:
    """Test enum comparisons and usage."""

    def test_enum_equality(self):
        """Test that enum values can be compared."""
        assert NodeType.PLAN == NodeType.PLAN
        assert NodeType.PLAN != NodeType.PLAN_STEP
        assert EdgeType.ROUTE == EdgeType.ROUTE
        assert EdgeType.ROUTE != EdgeType.NEXT

    def test_enum_string_comparison(self):
        """Test that enums can be compared with strings."""
        assert NodeType.PLAN == "plan"
        assert NodeType.PLAN != "plan_step"
        assert RouterType.EXPRESSION == "expression"

    def test_enum_in_collections(self):
        """Test that enums work in collections."""
        node_types = {NodeType.PLAN, NodeType.PLAN_STEP}
        assert NodeType.PLAN in node_types
        assert NodeType.ROUTER_STEP not in node_types

        edge_list = [EdgeType.ROUTE, EdgeType.NEXT, EdgeType.CUSTOM]
        assert EdgeType.ROUTE in edge_list
        assert EdgeType.PARENT_CHILD not in edge_list
