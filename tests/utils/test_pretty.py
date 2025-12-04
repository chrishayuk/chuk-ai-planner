"""Tests for utils/pretty.py"""

import pytest

from chuk_ai_planner.utils.pretty import clr, pretty_print_plan, PlanRunLogger
from chuk_ai_planner.core.graph import PlanNode, PlanStep, ToolCall
from chuk_session_manager.models.event_type import EventType


class TestClr:
    """Test the clr color helper function."""

    def test_clr_with_color(self):
        """Test color formatting with color enabled."""
        import os

        # Remove NO_COLOR if set
        old_value = os.environ.get("NO_COLOR")
        if "NO_COLOR" in os.environ:
            del os.environ["NO_COLOR"]

        result = clr("test", "1;33")
        assert result == "\033[1;33mtest\033[0m"

        # Restore NO_COLOR
        if old_value is not None:
            os.environ["NO_COLOR"] = old_value

    def test_clr_no_color(self):
        """Test color formatting with NO_COLOR set."""
        import os

        # Set NO_COLOR
        os.environ["NO_COLOR"] = "1"

        result = clr("test", "1;33")
        assert result == "test"

        # Clean up
        del os.environ["NO_COLOR"]


class TestPrettyPrintPlan:
    """Test the pretty_print_plan function."""

    @pytest.mark.asyncio
    async def test_pretty_print_plan_simple(self, graph_store, capsys):
        """Test printing a simple plan."""
        # Create a plan node
        plan = PlanNode(id="plan_1", title="Test Plan")
        await graph_store.add_node(plan)

        # Add a single step
        step = PlanStep(id="step_1", description="Step 1", index="1")
        await graph_store.add_node(step)
        from chuk_ai_planner.core.graph import ParentChildEdge

        await graph_store.add_edge(ParentChildEdge(src=plan.id, dst=step.id))

        # Print the plan
        await pretty_print_plan(graph_store, plan)

        # Capture output
        captured = capsys.readouterr()
        assert "Test Plan" in captured.out
        assert "Step 1" in captured.out

    @pytest.mark.asyncio
    async def test_pretty_print_plan_nested(self, graph_store, capsys):
        """Test printing a nested plan."""
        # Create a plan node
        plan = PlanNode(id="plan_1", title="Nested Plan")
        await graph_store.add_node(plan)

        # Add parent step
        step1 = PlanStep(id="step_1", description="Parent Step", index="1")
        await graph_store.add_node(step1)
        from chuk_ai_planner.core.graph import ParentChildEdge

        await graph_store.add_edge(ParentChildEdge(src=plan.id, dst=step1.id))

        # Add child step
        step2 = PlanStep(id="step_2", description="Child Step", index="1.1")
        await graph_store.add_node(step2)
        await graph_store.add_edge(ParentChildEdge(src=step1.id, dst=step2.id))

        # Print the plan
        await pretty_print_plan(graph_store, plan)

        # Capture output
        captured = capsys.readouterr()
        assert "Nested Plan" in captured.out
        assert "Parent Step" in captured.out
        assert "Child Step" in captured.out

    @pytest.mark.asyncio
    async def test_pretty_print_plan_not_plan_node(self, graph_store):
        """Test printing with a non-plan node raises error."""
        # Create a non-plan node
        step = PlanStep(id="step_1", description="Not a plan")
        await graph_store.add_node(step)

        with pytest.raises(ValueError, match="expected a PlanNode"):
            await pretty_print_plan(graph_store, step)

    @pytest.mark.asyncio
    async def test_pretty_print_plan_with_multiple_steps(self, graph_store, capsys):
        """Test printing a plan with multiple steps at the same level."""
        # Create a plan node
        plan = PlanNode(id="plan_1", title="Multi-step Plan")
        await graph_store.add_node(plan)

        # Add multiple steps
        from chuk_ai_planner.core.graph import ParentChildEdge

        for i in range(3):
            step = PlanStep(
                id=f"step_{i}", description=f"Step {i + 1}", index=str(i + 1)
            )
            await graph_store.add_node(step)
            await graph_store.add_edge(ParentChildEdge(src=plan.id, dst=step.id))

        # Print the plan
        await pretty_print_plan(graph_store, plan)

        # Capture output
        captured = capsys.readouterr()
        assert "Multi-step Plan" in captured.out
        assert "Step 1" in captured.out
        assert "Step 2" in captured.out
        assert "Step 3" in captured.out


class TestPlanRunLogger:
    """Test the PlanRunLogger class."""

    @pytest.mark.asyncio
    async def test_create_logger(self, graph_store):
        """Test creating a logger from a plan."""
        # Create a plan with steps
        plan = PlanNode(id="plan_1", title="Test Plan")
        await graph_store.add_node(plan)

        step = PlanStep(id="step_1", description="Test Step", index="1")
        await graph_store.add_node(step)
        from chuk_ai_planner.core.graph import ParentChildEdge

        await graph_store.add_edge(ParentChildEdge(src=plan.id, dst=step.id))

        # Create logger
        logger = await PlanRunLogger.create(graph_store, plan.id)

        assert logger is not None
        assert "step_1" in logger.label
        assert "Test Step" in logger.label["step_1"]

    @pytest.mark.asyncio
    async def test_create_logger_with_tool_calls(self, graph_store):
        """Test creating a logger with tool calls."""
        # Create a plan with steps and tool calls
        plan = PlanNode(id="plan_1", title="Test Plan")
        await graph_store.add_node(plan)

        step = PlanStep(id="step_1", description="Test Step", index="1")
        await graph_store.add_node(step)
        from chuk_ai_planner.core.graph import ParentChildEdge, PlanLinkEdge

        await graph_store.add_edge(ParentChildEdge(src=plan.id, dst=step.id))

        # Add a tool call
        tool = ToolCall(id="tool_1", name="test_tool", args={"arg": "value"})
        await graph_store.add_node(tool)
        await graph_store.add_edge(PlanLinkEdge(src=step.id, dst=tool.id))

        # Create logger
        logger = await PlanRunLogger.create(graph_store, plan.id)

        assert logger is not None
        assert "step_1" in logger.label
        assert "tool_1" in logger.label  # Tool should inherit step label

    @pytest.mark.asyncio
    async def test_logger_evt(self, graph_store, capsys):
        """Test logging step summary events."""
        # Create a simple plan
        plan = PlanNode(id="plan_1", title="Test Plan")
        await graph_store.add_node(plan)

        step = PlanStep(id="step_1", description="Test Step", index="1")
        await graph_store.add_node(step)
        from chuk_ai_planner.core.graph import ParentChildEdge

        await graph_store.add_edge(ParentChildEdge(src=plan.id, dst=step.id))

        # Create logger
        logger = await PlanRunLogger.create(graph_store, plan.id)

        # Log an event
        msg = {"step_id": "step_1", "status": "completed"}
        logger.evt(EventType.SUMMARY, msg, "parent_id")

        # Capture output
        captured = capsys.readouterr()
        assert "[step]" in captured.out
        assert "completed" in captured.out

    @pytest.mark.asyncio
    async def test_logger_evt_unknown_step(self, graph_store, capsys):
        """Test logging event for unknown step."""
        # Create empty logger
        logger = PlanRunLogger()

        # Log event for unknown step
        msg = {"step_id": "unknown_step", "status": "completed"}
        logger.evt(EventType.SUMMARY, msg, "parent_id")

        # Capture output
        captured = capsys.readouterr()
        assert "[step]" in captured.out
        assert "<?>" in captured.out

    @pytest.mark.asyncio
    async def test_logger_proc(self, graph_store, capsys):
        """Test logging tool call processing."""
        # Create a plan with tool
        plan = PlanNode(id="plan_1", title="Test Plan")
        await graph_store.add_node(plan)

        step = PlanStep(id="step_1", description="Test Step", index="1")
        await graph_store.add_node(step)
        from chuk_ai_planner.core.graph import ParentChildEdge, PlanLinkEdge

        await graph_store.add_edge(ParentChildEdge(src=plan.id, dst=step.id))

        tool = ToolCall(id="tool_1", name="test_tool", args={"arg": "value"})
        await graph_store.add_node(tool)
        await graph_store.add_edge(PlanLinkEdge(src=step.id, dst=tool.id))

        # Create logger
        logger = await PlanRunLogger.create(graph_store, plan.id)

        # Mock processor function
        async def mock_proc(tc, start_evt_id, assistant_id):
            return {"result": "success"}

        # Log tool call
        tc = {
            "id": "tool_1",
            "function": {"name": "test_tool", "arguments": '{"arg": "value"}'},
        }
        result = await logger.proc(tc, None, None, mock_proc)

        # Capture output
        captured = capsys.readouterr()
        assert "[tool]" in captured.out
        assert "test_tool" in captured.out
        assert "✓" in captured.out
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_logger_with_nested_steps(self, graph_store):
        """Test logger with nested steps."""
        # Create a nested plan
        plan = PlanNode(id="plan_1", title="Nested Plan")
        await graph_store.add_node(plan)

        step1 = PlanStep(id="step_1", description="Parent Step", index="1")
        await graph_store.add_node(step1)
        from chuk_ai_planner.core.graph import ParentChildEdge

        await graph_store.add_edge(ParentChildEdge(src=plan.id, dst=step1.id))

        step2 = PlanStep(id="step_2", description="Child Step", index="1.1")
        await graph_store.add_node(step2)
        await graph_store.add_edge(ParentChildEdge(src=step1.id, dst=step2.id))

        # Create logger
        logger = await PlanRunLogger.create(graph_store, plan.id)

        assert "step_1" in logger.label
        assert "step_2" in logger.label
