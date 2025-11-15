"""
Simple Conditional Routing Example
===================================

Demonstrates pure Pydantic graph API with conditional routing.

This example shows routing based on a quality score:
- If quality > 0.7 → publish
- If quality <= 0.7 → revise

Super clean typed API - no dictionary goop!
"""

import asyncio

from chuk_ai_planner.graph import (
    PlanNode,
    PlanStep,
    RouterStep,
    ParentChildEdge,
    RouteEdge,
    StepEdge,
)
from chuk_ai_planner.graph.types import RouterType
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.routing import RoutingExecutor


async def main():
    print("=" * 70)
    print("CONDITIONAL ROUTING EXAMPLE - Pure Pydantic API")
    print("=" * 70)

    # Create graph store
    graph = InMemoryGraphStore()

    # Create plan - typed fields, no dictionary!
    plan = PlanNode(
        title="Quality-based Publishing",
        description="Route content based on quality score",
    )
    graph.add_node(plan)

    # Step 1: Analyze content - clean typed API
    step1 = PlanStep(description="Analyze content quality", index="1")
    graph.add_node(step1)
    graph.add_edge(ParentChildEdge(src=plan.id, dst=step1.id))

    # Step 2: Router - pure Pydantic with enums!
    router = RouterStep(
        router_type=RouterType.EXPRESSION,  # Enum, not string!
        condition="${quality_score} > 0.7",
        routes=["high_quality", "low_quality"],
        description="Route based on quality score",
        route_mapping={
            True: "high_quality",  # When condition is True
            False: "low_quality",  # When condition is False
        },
    )
    graph.add_node(router)
    graph.add_edge(StepEdge(src=step1.id, dst=router.id))

    # Step 3a: Publish (high quality route)
    step_publish = PlanStep(description="Publish content", index="3a")
    graph.add_node(step_publish)

    # Step 3b: Revise (low quality route)
    step_revise = PlanStep(description="Revise and improve content", index="3b")
    graph.add_node(step_revise)

    # Create route edges - clean typed fields
    route_high = RouteEdge(src=router.id, dst=step_publish.id, route_key="high_quality")
    graph.add_edge(route_high)

    route_low = RouteEdge(src=router.id, dst=step_revise.id, route_key="low_quality")
    graph.add_edge(route_low)

    print(f"\n✅ Plan created with {len(graph.nodes)} nodes")
    print(f"   - Plan: {plan.id[:8]} (title: {plan.title})")
    print(f"   - Analysis step: {step1.id[:8]} (desc: {step1.description})")
    print(f"   - Router: {router.id[:8]} (type: {router.router_type})")
    print(f"   - Publish step: {step_publish.id[:8]}")
    print(f"   - Revise step: {step_revise.id[:8]}")

    # Test routing with high quality score
    print("\n" + "=" * 70)
    print("TEST 1: High Quality Score (0.85)")
    print("=" * 70)

    routing_executor = RoutingExecutor(graph)
    context_high = {
        "quality_score": 0.85,
        "content": "This is excellent content!",
    }

    decision_high = await routing_executor.evaluate_route(router, context_high)

    print("\n🔀 Routing Decision:")
    print(f"   Route chosen: {decision_high.route_key}")
    print(f"   Target step: {decision_high.target_step_id[:8]}")
    print(f"   Skipped routes: {', '.join(decision_high.skipped_routes)}")
    print(f"   Method: {decision_high.evaluation_method}")
    print(f"   Details: {decision_high.evaluation_details}")

    assert decision_high.route_key == "high_quality"
    assert decision_high.target_step_id == step_publish.id
    print("\n✅ High quality route correctly chosen!")

    # Test routing with low quality score
    print("\n" + "=" * 70)
    print("TEST 2: Low Quality Score (0.45)")
    print("=" * 70)

    context_low = {
        "quality_score": 0.45,
        "content": "This needs work...",
    }

    decision_low = await routing_executor.evaluate_route(router, context_low)

    print("\n🔀 Routing Decision:")
    print(f"   Route chosen: {decision_low.route_key}")
    print(f"   Target step: {decision_low.target_step_id[:8]}")
    print(f"   Skipped routes: {', '.join(decision_low.skipped_routes)}")
    print(f"   Method: {decision_low.evaluation_method}")
    print(f"   Details: {decision_low.evaluation_details}")

    assert decision_low.route_key == "low_quality"
    assert decision_low.target_step_id == step_revise.id
    print("\n✅ Low quality route correctly chosen!")

    # Test routing with boundary value
    print("\n" + "=" * 70)
    print("TEST 3: Boundary Value (0.7)")
    print("=" * 70)

    context_boundary = {
        "quality_score": 0.7,
        "content": "Right on the edge",
    }

    decision_boundary = await routing_executor.evaluate_route(router, context_boundary)

    print("\n🔀 Routing Decision:")
    print(f"   Route chosen: {decision_boundary.route_key}")
    print(f"   Target step: {decision_boundary.target_step_id[:8]}")
    print(f"   Method: {decision_boundary.evaluation_method}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✅")
    print("=" * 70)
    print("\n✨ Pure Pydantic API is working perfectly!")
    print("   - Type-safe fields (no dictionary goop)")
    print("   - Enums for constants (no hardcoded strings)")
    print("   - Clean, maintainable code")
    print("\nNext: Update routing executor and UniversalExecutor")


if __name__ == "__main__":
    asyncio.run(main())
