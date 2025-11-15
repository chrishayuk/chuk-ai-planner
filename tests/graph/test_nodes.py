# tests/graph/test_nodes.py
"""
Tests for pure Pydantic graph nodes.

Tests all node types with typed fields - no dictionary goop!
"""

import re
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from chuk_ai_planner.graph import (
    GraphNode,
    PlanNode,
    PlanStep,
    RouterStep,
    ToolCall,
    TaskRun,
    SessionNode,
    SummaryNode,
)
from chuk_ai_planner.graph.types import (
    NodeType,
    RouterType,
    StepStatus,
)

UUID_V4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.I,
)


class TestGraphNodeBase:
    """Test base GraphNode class."""

    def test_cannot_instantiate_base_directly(self):
        """GraphNode requires a kind, so can't instantiate without subclassing."""
        # This should fail because GraphNode is abstract
        with pytest.raises(ValidationError):
            GraphNode()  # No kind provided

    def test_hash_and_equality(self):
        """Test that nodes can be hashed and compared."""
        plan = PlanNode(title="Test")
        step = PlanStep(description="Test Step")

        # Should be hashable
        node_set = {plan, step}
        assert len(node_set) == 2

        # Should be unequal (different IDs)
        assert plan != step

        # Same ID should be equal
        plan2 = PlanNode(id=plan.id, title="Different Title")
        assert plan == plan2

    def test_equality_with_non_node(self):
        """Test comparing node with non-node objects."""
        node = PlanNode(title="Test")

        # Should not equal non-node objects
        assert node != "not a node"
        assert node != 123
        assert node is not None
        assert node != {"id": node.id}


class TestPlanNode:
    """Test PlanNode typed fields."""

    def test_create_with_required_fields(self):
        """Test creating a plan with required fields only."""
        plan = PlanNode(title="My Plan")

        assert plan.kind == NodeType.PLAN
        assert plan.title == "My Plan"
        assert plan.description is None
        assert plan.variables == {}
        assert plan.tags == []

    def test_create_with_all_fields(self):
        """Test creating a plan with all fields."""
        plan = PlanNode(
            title="Complete Plan",
            description="A fully specified plan",
            variables={"user_id": "123", "priority": "high"},
            tags=["important", "urgent"],
        )

        assert plan.title == "Complete Plan"
        assert plan.description == "A fully specified plan"
        assert plan.variables["user_id"] == "123"
        assert plan.variables["priority"] == "high"
        assert "important" in plan.tags
        assert "urgent" in plan.tags

    def test_fields_are_typed(self):
        """Test that fields have correct types."""
        plan = PlanNode(
            title="Test",
            variables={"key": "value"},
            tags=["tag1"],
        )

        assert isinstance(plan.title, str)
        assert isinstance(plan.variables, dict)
        assert isinstance(plan.tags, list)

    def test_immutable(self):
        """Test that plan is immutable."""
        plan = PlanNode(title="Test")

        with pytest.raises(ValidationError):
            plan.title = "New Title"  # Cannot mutate frozen model

    def test_auto_generated_fields(self):
        """Test that ID and timestamp are auto-generated."""
        plan = PlanNode(title="Test")

        assert UUID_V4_RE.match(plan.id)
        assert plan.ts.tzinfo == timezone.utc
        assert isinstance(plan.ts, datetime)

    def test_repr(self):
        """Test string representation."""
        plan = PlanNode(title="Test Plan")
        repr_str = repr(plan)

        assert repr_str.startswith("<plan:")
        assert plan.id[:8] in repr_str


class TestPlanStep:
    """Test PlanStep typed fields."""

    def test_create_with_required_fields(self):
        """Test creating a step with required fields."""
        step = PlanStep(description="Do something")

        assert step.kind == NodeType.PLAN_STEP
        assert step.description == "Do something"
        assert step.index is None
        assert step.status == StepStatus.PENDING
        assert step.result_variable is None

    def test_create_with_all_fields(self):
        """Test creating a step with all fields."""
        step = PlanStep(
            description="Execute task",
            index="1.2.3",
            status=StepStatus.RUNNING,
            result_variable="task_result",
        )

        assert step.description == "Execute task"
        assert step.index == "1.2.3"
        assert step.status == StepStatus.RUNNING
        assert step.result_variable == "task_result"

    def test_status_enum(self):
        """Test that status is a proper enum."""
        step = PlanStep(description="Test", status=StepStatus.COMPLETED)

        # Pydantic serializes enum to string value (use_enum_values=True)
        assert step.status == "completed"
        assert step.status == StepStatus.COMPLETED  # Can compare with enum

    def test_status_lifecycle(self):
        """Test changing status through new instances."""
        step1 = PlanStep(description="Test")
        assert step1.status == StepStatus.PENDING

        # Can't mutate, must create new instance
        with pytest.raises(ValidationError):
            step1.status = StepStatus.RUNNING


class TestRouterStep:
    """Test RouterStep typed fields."""

    def test_create_expression_router(self):
        """Test creating an expression-based router."""
        router = RouterStep(
            router_type=RouterType.EXPRESSION,
            routes=["high", "low"],
            description="Quality check",
            condition="${score} > 0.7",
            route_mapping={True: "high", False: "low"},
        )

        assert router.kind == NodeType.ROUTER_STEP
        assert router.router_type == RouterType.EXPRESSION
        assert router.routes == ["high", "low"]
        assert router.description == "Quality check"
        assert router.condition == "${score} > 0.7"
        assert router.route_mapping[True] == "high"

    def test_create_llm_router(self):
        """Test creating an LLM-based router."""
        router = RouterStep(
            router_type=RouterType.LLM,
            routes=["technical", "creative", "balanced"],
            description="Choose style",
            llm_prompt="What writing style fits best?",
        )

        assert router.router_type == RouterType.LLM
        assert len(router.routes) == 3
        assert router.llm_prompt == "What writing style fits best?"
        assert router.condition is None  # LLM router doesn't use condition

    def test_routes_validation(self):
        """Test that routers must have at least 2 routes."""
        with pytest.raises(ValidationError):
            RouterStep(
                router_type=RouterType.EXPRESSION,
                routes=["only_one"],  # Too few!
                description="Invalid router",
            )

    def test_router_type_enum(self):
        """Test that router_type is a proper enum."""
        router = RouterStep(
            router_type=RouterType.FUNCTION,
            routes=["a", "b"],
            description="Test",
            router_function="my_function",
        )

        # Pydantic serializes enum to string value (use_enum_values=True)
        assert router.router_type == "function"
        assert router.router_type == RouterType.FUNCTION  # Can compare with enum


class TestToolCall:
    """Test ToolCall typed fields."""

    def test_create_with_required_fields(self):
        """Test creating a tool call with required fields."""
        tool = ToolCall(name="calculate_sum")

        assert tool.kind == NodeType.TOOL_CALL
        assert tool.name == "calculate_sum"
        assert tool.args == {}
        assert tool.result_variable is None

    def test_create_with_args(self):
        """Test creating a tool call with arguments."""
        tool = ToolCall(
            name="fetch_data",
            args={"url": "https://api.example.com", "timeout": 30},
            result_variable="api_response",
        )

        assert tool.name == "fetch_data"
        assert tool.args["url"] == "https://api.example.com"
        assert tool.args["timeout"] == 30
        assert tool.result_variable == "api_response"

    def test_args_type(self):
        """Test that args is a dict."""
        tool = ToolCall(name="test", args={"key": "value"})

        assert isinstance(tool.args, dict)


class TestTaskRun:
    """Test TaskRun typed fields."""

    def test_create_successful_run(self):
        """Test creating a successful task run."""
        task = TaskRun(
            tool_call_id="tool-123",
            status="success",
            result={"data": [1, 2, 3]},
        )

        assert task.kind == NodeType.TASK_RUN
        assert task.tool_call_id == "tool-123"
        assert task.status == "success"
        assert task.result["data"] == [1, 2, 3]
        assert task.error is None

    def test_create_failed_run(self):
        """Test creating a failed task run."""
        task = TaskRun(
            tool_call_id="tool-456",
            status="failure",
            error="Connection timeout",
        )

        assert task.status == "failure"
        assert task.error == "Connection timeout"
        assert task.result is None

    def test_with_timing(self):
        """Test task run with timing information."""
        start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 12, 0, 5, tzinfo=timezone.utc)

        task = TaskRun(
            tool_call_id="tool-789",
            status="success",
            started_at=start,
            completed_at=end,
        )

        assert task.started_at == start
        assert task.completed_at == end
        assert task.duration_seconds == 5.0

    def test_duration_calculation(self):
        """Test duration property."""
        task1 = TaskRun(
            tool_call_id="test",
            status="success",
        )
        assert task1.duration_seconds is None  # No timing info

        start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 12, 0, 10, 500000, tzinfo=timezone.utc)
        task2 = TaskRun(
            tool_call_id="test",
            status="success",
            started_at=start,
            completed_at=end,
        )
        assert task2.duration_seconds == 10.5


class TestSessionNode:
    """Test SessionNode typed fields."""

    def test_create_with_required_fields(self):
        """Test creating a session with required fields."""
        session = SessionNode(name="Analysis Session")

        assert session.kind == NodeType.SESSION
        assert session.name == "Analysis Session"
        assert session.description is None
        assert session.user_id is None
        assert session.context == {}
        assert session.status == "active"

    def test_create_with_all_fields(self):
        """Test creating a session with all fields."""
        start = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        session = SessionNode(
            name="Data Processing",
            description="Process customer data",
            user_id="user-123",
            context={"dataset": "customers", "format": "csv"},
            status="completed",
            started_at=start,
        )

        assert session.name == "Data Processing"
        assert session.description == "Process customer data"
        assert session.user_id == "user-123"
        assert session.context["dataset"] == "customers"
        assert session.status == "completed"
        assert session.started_at == start


class TestSummaryNode:
    """Test SummaryNode typed fields."""

    def test_create_with_required_fields(self):
        """Test creating a summary with required fields."""
        summary = SummaryNode(
            title="Execution Complete",
            content="All tasks completed successfully",
        )

        assert summary.kind == NodeType.SUMMARY
        assert summary.title == "Execution Complete"
        assert summary.content == "All tasks completed successfully"
        assert summary.summary_type == "checkpoint"
        assert summary.execution_state == {}

    def test_create_error_summary(self):
        """Test creating an error summary."""
        summary = SummaryNode(
            title="Execution Failed",
            content="Task failed due to timeout",
            summary_type="error",
            execution_state={"last_step": "fetch_data", "error_code": 408},
        )

        assert summary.summary_type == "error"
        assert summary.execution_state["last_step"] == "fetch_data"
        assert summary.execution_state["error_code"] == 408


class TestNodeImmutability:
    """Test that all nodes are immutable."""

    @pytest.mark.parametrize(
        "node",
        [
            PlanNode(title="Test"),
            PlanStep(description="Test"),
            RouterStep(
                router_type=RouterType.EXPRESSION,
                routes=["a", "b"],
                description="Test",
            ),
            ToolCall(name="test"),
            TaskRun(tool_call_id="test", status="success"),
            SessionNode(name="test"),
            SummaryNode(title="test", content="test"),
        ],
    )
    def test_node_is_frozen(self, node):
        """Test that nodes cannot be mutated."""
        with pytest.raises(ValidationError):
            node.id = "new_id"

    def test_metadata_is_mutable_dict(self):
        """Test that metadata can be used for extensibility."""
        plan = PlanNode(
            title="Test",
            metadata={"custom_field": "value", "priority": 5},
        )

        assert plan.metadata["custom_field"] == "value"
        assert plan.metadata["priority"] == 5


class TestNodeCollections:
    """Test working with collections of nodes."""

    def test_nodes_in_sets(self):
        """Test that nodes work in sets."""
        plan = PlanNode(title="Plan")
        step1 = PlanStep(description="Step 1")
        step2 = PlanStep(description="Step 2")

        node_set = {plan, step1, step2}
        assert len(node_set) == 3
        assert plan in node_set

    def test_nodes_as_dict_keys(self):
        """Test that nodes can be used as dict keys."""
        plan = PlanNode(title="Plan")
        step = PlanStep(description="Step")

        node_dict = {plan: "plan_value", step: "step_value"}
        assert node_dict[plan] == "plan_value"
        assert node_dict[step] == "step_value"

    def test_filtering_by_kind(self):
        """Test filtering nodes by type."""
        nodes = [
            PlanNode(title="Plan"),
            PlanStep(description="Step 1"),
            PlanStep(description="Step 2"),
            RouterStep(
                router_type=RouterType.EXPRESSION,
                routes=["a", "b"],
                description="Router",
            ),
        ]

        steps = [n for n in nodes if n.kind == NodeType.PLAN_STEP]
        assert len(steps) == 2
        assert all(isinstance(n, PlanStep) for n in steps)
