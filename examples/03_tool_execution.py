#!/usr/bin/env python3
"""
Tool Execution Example
=======================

This example demonstrates tool execution in plans:
- ToolCall nodes for defining tool invocations
- TaskRun nodes for tracking results
- PlanLinkEdge to connect steps to tools
- Result tracking with typed fields

Key Takeaway: Execute tools and track results with type safety.
"""

from datetime import datetime, timezone
from chuk_ai_planner.graph import (
    PlanNode,
    PlanStep,
    ToolCall,
    TaskRun,
    ParentChildEdge,
    PlanLinkEdge,
)
from chuk_ai_planner.graph.types import NodeType
from chuk_ai_planner.store.memory import InMemoryGraphStore


def main():
    print("=" * 70)
    print("Tool Execution Example")
    print("=" * 70)

    graph = InMemoryGraphStore()

    # 1. Create a Plan
    plan = PlanNode(
        title="Data Processing Pipeline",
        description="Fetch, process, and analyze data"
    )
    graph.add_node(plan)
    print(f"\n✅ Created Plan: {plan.title}")

    # 2. Create Step 1: Fetch Data
    fetch_step = PlanStep(
        description="Fetch data from API",
        index="1"
    )
    graph.add_node(fetch_step)
    graph.add_edge(ParentChildEdge(src=plan.id, dst=fetch_step.id))

    # 3. Create Tool Call for fetching (typed fields!)
    fetch_tool = ToolCall(
        name="fetch_api_data",
        args={
            "url": "https://api.example.com/data",
            "method": "GET",
            "timeout": 30
        }
    )
    graph.add_node(fetch_tool)

    # Link step to tool
    graph.add_edge(PlanLinkEdge(src=fetch_step.id, dst=fetch_tool.id))
    print(f"\n✅ Step 1: {fetch_step.description}")
    print(f"   Tool: {fetch_tool.name}")
    print(f"   Args: {fetch_tool.args}")

    # 4. Create TaskRun for the result (typed fields!)
    fetch_result = TaskRun(
        tool_call_id=fetch_tool.id,
        status="success",  # One of: success, failure, running, pending
        result={
            "data": [1, 2, 3, 4, 5],
            "count": 5
        },
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc)
    )
    graph.add_node(fetch_result)
    graph.add_edge(ParentChildEdge(src=fetch_tool.id, dst=fetch_result.id))
    print(f"   Result Status: {fetch_result.status}")
    print(f"   Result Data: {fetch_result.result}")

    # 5. Create Step 2: Process Data
    process_step = PlanStep(
        description="Process fetched data",
        index="2"
    )
    graph.add_node(process_step)
    graph.add_edge(ParentChildEdge(src=plan.id, dst=process_step.id))

    # Tool for processing
    process_tool = ToolCall(
        name="transform_data",
        args={
            "input_data": "${fetch_result}",  # Reference to previous result
            "transformation": "normalize"
        }
    )
    graph.add_node(process_tool)
    graph.add_edge(PlanLinkEdge(src=process_step.id, dst=process_tool.id))
    print(f"\n✅ Step 2: {process_step.description}")
    print(f"   Tool: {process_tool.name}")

    # 6. Create Step 3: Analyze with multiple tools
    analyze_step = PlanStep(
        description="Analyze processed data",
        index="3"
    )
    graph.add_node(analyze_step)
    graph.add_edge(ParentChildEdge(src=plan.id, dst=analyze_step.id))

    # Multiple tools for one step
    stats_tool = ToolCall(name="calculate_statistics", args={"data": "${processed_data}"})
    plot_tool = ToolCall(name="generate_plot", args={"data": "${processed_data}"})

    graph.add_node(stats_tool)
    graph.add_node(plot_tool)

    graph.add_edge(PlanLinkEdge(src=analyze_step.id, dst=stats_tool.id))
    graph.add_edge(PlanLinkEdge(src=analyze_step.id, dst=plot_tool.id))

    print(f"\n✅ Step 3: {analyze_step.description}")
    print(f"   Tool 1: {stats_tool.name}")
    print(f"   Tool 2: {plot_tool.name}")

    # 7. Query the Graph
    print("\n" + "=" * 70)
    print("Graph Structure")
    print("=" * 70)

    # Find all tool calls
    tools = graph.get_nodes_by_kind(NodeType.TOOL_CALL)
    print(f"\n📊 Total Tools: {len(tools)}")
    for tool in tools:
        print(f"   - {tool.name}: {tool.args}")

    # Find task results
    task_runs = graph.get_nodes_by_kind(NodeType.TASK_RUN)
    print(f"\n📊 Total Task Runs: {len(task_runs)}")
    for task in task_runs:
        print(f"   - Status: {task.status}")
        if task.duration_seconds:
            print(f"     Duration: {task.duration_seconds}s")

    # Find tools for a specific step
    step_tools = graph.get_edges(src=analyze_step.id, kind=EdgeType.PLAN_LINK)
    print(f"\n📊 Tools for Step 3: {len(step_tools)}")
    for edge in step_tools:
        tool_node = graph.get_node(edge.dst)
        print(f"   - {tool_node.name}")

    # 8. Demonstrate Typed Fields
    print("\n" + "=" * 70)
    print("Typed Fields (No Dictionary Goop!)")
    print("=" * 70)

    print("\n🔍 ToolCall typed fields:")
    print(f"   name: str = '{fetch_tool.name}'")
    print(f"   args: dict = {fetch_tool.args}")
    print("   (Not .data.get('name')!)")

    print("\n🔍 TaskRun typed fields:")
    print(f"   status: str = '{fetch_result.status}'")
    print(f"   result: Optional[Any] = {fetch_result.result}")
    print(f"   started_at: datetime = {fetch_result.started_at}")
    print(f"   completed_at: Optional[datetime] = {fetch_result.completed_at}")
    print("   (All properly typed!)")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✅ ToolCall nodes with typed name and args")
    print("✅ TaskRun nodes with status, result, and timing")
    print("✅ PlanLinkEdge connects steps to tools")
    print("✅ Multiple tools per step supported")
    print("✅ Result tracking with proper types")
    print("\n")


if __name__ == "__main__":
    from chuk_ai_planner.graph.types import EdgeType  # For query
    main()
