# tests/graph/test_edges.py
"""
Tests for pure Pydantic graph edges.

Tests all edge types with typed fields - no dictionary goop!
"""

import re

import pytest
from pydantic import ValidationError

from chuk_ai_planner.graph import (
    GraphEdge,
    ParentChildEdge,
    PlanLinkEdge,
    StepEdge,
    RouteEdge,
    NextEdge,
    CustomEdge,
)
from chuk_ai_planner.graph.types import EdgeType

UUID_V4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.I,
)


class TestGraphEdgeBase:
    """Test base GraphEdge class."""

    def test_cannot_instantiate_base_directly(self):
        """GraphEdge requires a kind, so can't instantiate without subclassing."""
        with pytest.raises(ValidationError):
            GraphEdge(src="node1", dst="node2")  # No kind provided

    def test_hash_and_equality(self):
        """Test that edges can be hashed and compared."""
        edge1 = RouteEdge(src="a", dst="b", route_key="route1")
        edge2 = RouteEdge(src="a", dst="b", route_key="route2")

        # Should be hashable
        edge_set = {edge1, edge2}
        assert len(edge_set) == 2

        # Should be unequal (different IDs)
        assert edge1 != edge2

        # Same ID should be equal
        edge3 = RouteEdge(id=edge1.id, src="c", dst="d", route_key="route3")
        assert edge1 == edge3

    def test_equality_with_non_edge(self):
        """Test comparing edge with non-edge objects."""
        edge = RouteEdge(src="a", dst="b", route_key="route1")

        # Should not equal non-edge objects
        assert edge != "not an edge"
        assert edge != 123
        assert edge is not None
        assert edge != {"id": edge.id}


class TestParentChildEdge:
    """Test ParentChildEdge typed fields."""

    def test_create_parent_child_edge(self):
        """Test creating a parent-child edge."""
        edge = ParentChildEdge(src="parent_id", dst="child_id")

        assert edge.kind == EdgeType.PARENT_CHILD
        assert edge.src == "parent_id"
        assert edge.dst == "child_id"
        assert UUID_V4_RE.match(edge.id)

    def test_immutable(self):
        """Test that edge is immutable."""
        edge = ParentChildEdge(src="a", dst="b")

        with pytest.raises(ValidationError):
            edge.src = "new_src"

    def test_repr(self):
        """Test string representation."""
        edge = ParentChildEdge(src="parent_id_123", dst="child_id_456")
        repr_str = repr(edge)

        assert repr_str.startswith("<parent_child:")
        assert "→" in repr_str  # Shows direction


class TestPlanLinkEdge:
    """Test PlanLinkEdge typed fields."""

    def test_create_plan_link(self):
        """Test creating a plan link edge."""
        edge = PlanLinkEdge(src="plan_id", dst="step_id")

        assert edge.kind == EdgeType.PLAN_LINK
        assert edge.src == "plan_id"
        assert edge.dst == "step_id"


class TestStepEdge:
    """Test StepEdge typed fields."""

    def test_create_with_defaults(self):
        """Test creating a step edge with default values."""
        edge = StepEdge(src="step1", dst="step2")

        assert edge.kind == EdgeType.STEP_ORDER
        assert edge.src == "step1"
        assert edge.dst == "step2"
        assert edge.dependency is True
        assert edge.condition is None

    def test_create_with_all_fields(self):
        """Test creating a step edge with all fields."""
        edge = StepEdge(
            src="step1",
            dst="step2",
            dependency=False,
            condition="${step1.success} == true",
        )

        assert edge.dependency is False
        assert edge.condition == "${step1.success} == true"

    def test_non_dependency_edge(self):
        """Test creating an edge that's not a dependency."""
        edge = StepEdge(src="step1", dst="step2", dependency=False)

        assert edge.dependency is False


class TestRouteEdge:
    """Test RouteEdge typed fields."""

    def test_create_with_required_fields(self):
        """Test creating a route edge with required fields."""
        edge = RouteEdge(src="router_id", dst="target_id", route_key="high_quality")

        assert edge.kind == EdgeType.ROUTE
        assert edge.src == "router_id"
        assert edge.dst == "target_id"
        assert edge.route_key == "high_quality"
        assert edge.is_default is False

    def test_create_default_route(self):
        """Test creating a default route edge."""
        edge = RouteEdge(
            src="router_id",
            dst="fallback_id",
            route_key="default",
            is_default=True,
        )

        assert edge.route_key == "default"
        assert edge.is_default is True

    def test_route_key_is_typed(self):
        """Test that route_key is a string field."""
        edge = RouteEdge(src="a", dst="b", route_key="test_route")

        assert isinstance(edge.route_key, str)
        assert edge.route_key == "test_route"


class TestNextEdge:
    """Test NextEdge typed fields."""

    def test_create_without_weight(self):
        """Test creating a next edge without weight."""
        edge = NextEdge(src="step1", dst="step2")

        assert edge.kind == EdgeType.NEXT
        assert edge.src == "step1"
        assert edge.dst == "step2"
        assert edge.weight is None

    def test_create_with_weight(self):
        """Test creating a next edge with weight."""
        edge = NextEdge(src="step1", dst="step2", weight=5.0)

        assert edge.weight == 5.0

    def test_weight_type(self):
        """Test that weight is a float field."""
        edge = NextEdge(src="a", dst="b", weight=3.14)

        assert isinstance(edge.weight, float)
        assert edge.weight == 3.14


class TestCustomEdge:
    """Test CustomEdge typed fields."""

    def test_create_with_required_fields(self):
        """Test creating a custom edge with required fields."""
        edge = CustomEdge(src="node1", dst="node2", custom_type="approval")

        assert edge.kind == EdgeType.CUSTOM
        assert edge.src == "node1"
        assert edge.dst == "node2"
        assert edge.custom_type == "approval"
        assert edge.properties == {}

    def test_create_with_properties(self):
        """Test creating a custom edge with properties."""
        edge = CustomEdge(
            src="node1",
            dst="node2",
            custom_type="workflow_transition",
            properties={
                "requires_approval": True,
                "approver_role": "manager",
                "timeout_hours": 24,
            },
        )

        assert edge.custom_type == "workflow_transition"
        assert edge.properties["requires_approval"] is True
        assert edge.properties["approver_role"] == "manager"
        assert edge.properties["timeout_hours"] == 24

    def test_properties_type(self):
        """Test that properties is a dict."""
        edge = CustomEdge(
            src="a",
            dst="b",
            custom_type="test",
            properties={"key": "value"},
        )

        assert isinstance(edge.properties, dict)


class TestEdgeImmutability:
    """Test that all edges are immutable."""

    @pytest.mark.parametrize(
        "edge",
        [
            ParentChildEdge(src="a", dst="b"),
            PlanLinkEdge(src="a", dst="b"),
            StepEdge(src="a", dst="b"),
            RouteEdge(src="a", dst="b", route_key="route"),
            NextEdge(src="a", dst="b"),
            CustomEdge(src="a", dst="b", custom_type="custom"),
        ],
    )
    def test_edge_is_frozen(self, edge):
        """Test that edges cannot be mutated."""
        with pytest.raises(ValidationError):
            edge.src = "new_src"

        with pytest.raises(ValidationError):
            edge.dst = "new_dst"

    def test_metadata_is_mutable_dict(self):
        """Test that metadata can be used for extensibility."""
        edge = RouteEdge(
            src="a",
            dst="b",
            route_key="route",
            metadata={"custom_field": "value", "priority": 5},
        )

        assert edge.metadata["custom_field"] == "value"
        assert edge.metadata["priority"] == 5


class TestEdgeCollections:
    """Test working with collections of edges."""

    def test_edges_in_sets(self):
        """Test that edges work in sets."""
        edge1 = RouteEdge(src="a", dst="b", route_key="route1")
        edge2 = RouteEdge(src="a", dst="c", route_key="route2")
        edge3 = ParentChildEdge(src="x", dst="y")

        edge_set = {edge1, edge2, edge3}
        assert len(edge_set) == 3
        assert edge1 in edge_set

    def test_edges_as_dict_keys(self):
        """Test that edges can be used as dict keys."""
        edge1 = StepEdge(src="a", dst="b")
        edge2 = StepEdge(src="b", dst="c")

        edge_dict = {edge1: "first", edge2: "second"}
        assert edge_dict[edge1] == "first"
        assert edge_dict[edge2] == "second"

    def test_filtering_by_kind(self):
        """Test filtering edges by type."""
        edges = [
            ParentChildEdge(src="a", dst="b"),
            RouteEdge(src="r", dst="t1", route_key="route1"),
            RouteEdge(src="r", dst="t2", route_key="route2"),
            NextEdge(src="x", dst="y"),
        ]

        route_edges = [e for e in edges if e.kind == EdgeType.ROUTE]
        assert len(route_edges) == 2
        assert all(isinstance(e, RouteEdge) for e in route_edges)

    def test_filtering_by_source(self):
        """Test filtering edges by source node."""
        src_id = "router_123"
        edges = [
            RouteEdge(src=src_id, dst="target1", route_key="route1"),
            RouteEdge(src=src_id, dst="target2", route_key="route2"),
            RouteEdge(src="other", dst="target3", route_key="route3"),
        ]

        from_router = [e for e in edges if e.src == src_id]
        assert len(from_router) == 2


class TestEdgeDirection:
    """Test that edges properly represent direction."""

    def test_src_and_dst_are_distinct(self):
        """Test that source and destination can be different."""
        edge = ParentChildEdge(src="parent", dst="child")

        assert edge.src != edge.dst
        assert edge.src == "parent"
        assert edge.dst == "child"

    def test_can_have_same_src_and_dst(self):
        """Test that self-loops are allowed (if needed)."""
        # Some graph algorithms might need self-loops
        edge = NextEdge(src="node1", dst="node1")

        assert edge.src == edge.dst == "node1"

    def test_repr_shows_direction(self):
        """Test that repr shows the direction with arrow."""
        edge = StepEdge(src="step_abc123", dst="step_def456")
        repr_str = repr(edge)

        assert "→" in repr_str
        assert "step_abc123"[:6] in repr_str
        assert "step_def456"[:6] in repr_str
