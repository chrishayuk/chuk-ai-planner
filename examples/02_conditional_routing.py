#!/usr/bin/env python3
"""
Conditional Routing Example
============================

This example demonstrates conditional routing in plans:
- RouterStep nodes with expression-based routing
- RouteEdge connections
- Multiple execution paths based on conditions
- Variable substitution

Key Takeaway: Build dynamic workflows with conditional logic.
"""

from chuk_ai_planner.graph import (
    PlanNode,
    PlanStep,
    RouterStep,
    ParentChildEdge,
    StepEdge,
    RouteEdge,
)
from chuk_ai_planner.graph.types import NodeType, EdgeType, RouterType
from chuk_ai_planner.store.memory import InMemoryGraphStore


def main():
    print("=" * 70)
    print("Conditional Routing Example")
    print("=" * 70)

    graph = InMemoryGraphStore()

    # 1. Create a Plan for content quality routing
    plan = PlanNode(
        title="Quality-Based Publishing",
        description="Route content based on quality score",
        variables={"quality_threshold": 0.7}
    )
    graph.add_node(plan)
    print(f"\n✅ Created Plan: {plan.title}")

    # 2. Create analysis step
    analyze_step = PlanStep(
        description="Analyze content quality",
        index="1"
    )
    graph.add_node(analyze_step)
    graph.add_edge(ParentChildEdge(src=plan.id, dst=analyze_step.id))
    print(f"✅ Created Step 1: {analyze_step.description}")

    # 3. Create Router Step with typed fields (no data dict!)
    router = RouterStep(
        router_type=RouterType.EXPRESSION,  # Enum, not string!
        description="Route based on quality score",
        condition="${quality_score} > 0.7",  # Expression with variable
        routes=["high_quality", "low_quality"],
        route_mapping={
            True: "high_quality",   # If condition is true
            False: "low_quality"    # If condition is false
        }
    )
    graph.add_node(router)
    graph.add_edge(ParentChildEdge(src=plan.id, dst=router.id))
    graph.add_edge(StepEdge(src=analyze_step.id, dst=router.id))
    print("\n✅ Created Router:")
    print(f"   Type: {router.router_type}")  # RouterType.EXPRESSION
    print(f"   Condition: {router.condition}")
    print(f"   Routes: {router.routes}")

    # 4. Create High Quality Path
    publish_step = PlanStep(
        description="Publish immediately",
        index="2a"
    )
    graph.add_node(publish_step)

    # Create RouteEdge with typed fields
    graph.add_edge(RouteEdge(
        src=router.id,
        dst=publish_step.id,
        route_key="high_quality"  # Matches route name
    ))
    print(f"\n✅ High Quality Route → {publish_step.description}")

    # 5. Create Low Quality Path
    revise_step = PlanStep(
        description="Send for revision",
        index="2b"
    )
    graph.add_node(revise_step)

    graph.add_edge(RouteEdge(
        src=router.id,
        dst=revise_step.id,
        route_key="low_quality",
        is_default=True  # Typed boolean field
    ))
    print(f"✅ Low Quality Route (default) → {revise_step.description}")

    # 6. Add follow-up step after revision
    reanalyze_step = PlanStep(
        description="Re-analyze revised content",
        index="3"
    )
    graph.add_node(reanalyze_step)
    graph.add_edge(StepEdge(src=revise_step.id, dst=reanalyze_step.id))
    print(f"✅ Follow-up: {reanalyze_step.description}")

    # 7. Query the Graph Structure
    print("\n" + "=" * 70)
    print("Graph Structure")
    print("=" * 70)

    # Find all routes from router
    routes = graph.get_edges(src=router.id, kind=EdgeType.ROUTE)
    print(f"\n📊 Router has {len(routes)} routes:")
    for edge in routes:
        target = graph.get_node(edge.dst)
        print(f"   → {edge.route_key}: {target.description}")
        if edge.is_default:
            print("      (default route)")

    # Find router steps
    routers = graph.get_nodes_by_kind(NodeType.ROUTER_STEP)
    print(f"\n📊 Found {len(routers)} router step(s)")
    for r in routers:
        print(f"   Router: {r.description}")
        print(f"   Type: {r.router_type}")
        print(f"   Condition: {r.condition}")

    # 8. Demonstrate Router Types
    print("\n" + "=" * 70)
    print("Router Types")
    print("=" * 70)

    # Expression router (already created above)
    print("\n🎯 Expression Router:")
    print(f"   Evaluates: {router.condition}")

    # LLM router example (conceptual)
    llm_router = RouterStep(
        router_type=RouterType.LLM,  # Enum!
        description="LLM decides the route",
        routes=["approve", "reject"],
        llm_prompt="Analyze this content and decide if it should be approved or rejected."
    )
    print("\n🤖 LLM Router:")
    print(f"   Type: {llm_router.router_type}")
    print(f"   Prompt: {llm_router.llm_prompt}")

    # Function router example (conceptual)
    function_router = RouterStep(
        router_type=RouterType.FUNCTION,  # Enum!
        description="Custom function routing",
        routes=["urgent", "normal"],
        router_function="calculate_priority"
    )
    print("\n⚡ Function Router:")
    print(f"   Type: {function_router.router_type}")
    print(f"   Function: {function_router.router_function}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✅ RouterStep with typed router_type enum")
    print("✅ RouteEdge with route_key for matching")
    print("✅ Multiple execution paths based on conditions")
    print("✅ Support for EXPRESSION, LLM, and FUNCTION routing")
    print("\n")


if __name__ == "__main__":
    main()
