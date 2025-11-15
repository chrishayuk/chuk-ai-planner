#!/usr/bin/env python3
"""
Basic Graph Structure Example
==============================

This example demonstrates the fundamentals of the pure Pydantic graph:
- Creating typed nodes with fields (no .data dictionaries!)
- Creating typed edges
- Using the InMemoryGraphStore
- Querying the graph

Key Takeaway: Everything is type-safe with Pydantic models.
"""

from chuk_ai_planner.graph import (
    PlanNode,
    PlanStep,
    ParentChildEdge,
    StepEdge,
)
from chuk_ai_planner.graph.types import NodeType, EdgeType, StepStatus
from chuk_ai_planner.store.memory import InMemoryGraphStore


def main():
    print("=" * 70)
    print("Basic Graph Structure Example")
    print("=" * 70)

    # Create an in-memory graph store
    graph = InMemoryGraphStore()

    # 1. Create a Plan Node (typed fields, no data dict!)
    plan = PlanNode(
        title="Content Publishing Workflow",
        description="A simple workflow for creating and publishing content",
        variables={"author": "Alice", "topic": "AI Planning"}
    )
    graph.add_node(plan)
    print(f"\n✅ Created Plan: {plan.title}")
    print(f"   ID: {plan.id}")
    print(f"   Kind: {plan.kind}")  # Automatically set to NodeType.PLAN

    # 2. Create Plan Steps (typed fields!)
    step1 = PlanStep(
        description="Research the topic",
        index="1",
        status=StepStatus.PENDING
    )
    graph.add_node(step1)

    step2 = PlanStep(
        description="Write draft",
        index="2",
        status=StepStatus.PENDING
    )
    graph.add_node(step2)

    step3 = PlanStep(
        description="Review and edit",
        index="3",
        status=StepStatus.PENDING
    )
    graph.add_node(step3)

    print(f"\n✅ Created {3} steps")

    # 3. Create Parent-Child Edges (Plan → Steps)
    graph.add_edge(ParentChildEdge(src=plan.id, dst=step1.id))
    graph.add_edge(ParentChildEdge(src=plan.id, dst=step2.id))
    graph.add_edge(ParentChildEdge(src=plan.id, dst=step3.id))
    print("✅ Linked steps to plan")

    # 4. Create Step Edges (Dependencies)
    # Step 2 depends on Step 1
    graph.add_edge(StepEdge(src=step1.id, dst=step2.id))
    # Step 3 depends on Step 2
    graph.add_edge(StepEdge(src=step2.id, dst=step3.id))
    print("✅ Created step dependencies")

    # 5. Query the Graph
    print("\n" + "=" * 70)
    print("Querying the Graph")
    print("=" * 70)

    # Get all plan nodes
    plans = graph.get_nodes_by_kind(NodeType.PLAN)
    print(f"\n📊 Found {len(plans)} plan(s)")

    # Get all step nodes
    steps = graph.get_nodes_by_kind(NodeType.PLAN_STEP)
    print(f"📊 Found {len(steps)} step(s)")

    # Get children of plan (using ParentChild edges)
    plan_children = graph.get_edges(src=plan.id, kind=EdgeType.PARENT_CHILD)
    print(f"📊 Plan has {len(plan_children)} child steps")

    # Get step dependencies (StepEdge)
    step2_deps = graph.get_edges(dst=step2.id, kind=EdgeType.STEP_ORDER)
    print(f"📊 Step 2 depends on {len(step2_deps)} other step(s)")

    # 6. Access Typed Fields (No .data.get()!)
    print("\n" + "=" * 70)
    print("Accessing Typed Fields")
    print("=" * 70)

    print("\n🔍 Plan Details:")
    print(f"   Title: {plan.title}")  # Direct field access!
    print(f"   Description: {plan.description}")
    print(f"   Variables: {plan.variables}")

    print("\n🔍 Step 1 Details:")
    print(f"   Description: {step1.description}")  # Not step1.data.get("description")!
    print(f"   Index: {step1.index}")
    print(f"   Status: {step1.status}")  # Enum value
    print(f"   Status is enum: {isinstance(step1.status, str)}")  # True (use_enum_values)

    # 7. Demonstrate Immutability
    print("\n" + "=" * 70)
    print("Immutability (Frozen Models)")
    print("=" * 70)

    try:
        step1.description = "Changed"  # This will fail!
    except Exception as e:
        print(f"✅ Cannot modify frozen model: {type(e).__name__}")

    # To update, create a new node
    updated_step1 = step1.model_copy(update={"status": StepStatus.RUNNING})
    print(f"✅ Created updated copy with new status: {updated_step1.status}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✅ Pure Pydantic graph - no dictionary goop!")
    print("✅ Type-safe enums - NodeType.PLAN, not 'plan'")
    print("✅ Direct field access - plan.title, not plan.data.get('title')")
    print("✅ Immutable nodes - use model_copy() to update")
    print("\n")


if __name__ == "__main__":
    main()
