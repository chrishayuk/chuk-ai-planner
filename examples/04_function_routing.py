#!/usr/bin/env python3
"""
Function-Based Routing Example
===============================

This example demonstrates function-based routing:
- FunctionRegistry for registering custom routing functions
- RouterStep with function routing
- Both decorator and manual registration
- Context-based routing decisions

Key Takeaway: Custom logic for complex routing decisions.
"""

import asyncio

from chuk_ai_planner.core.graph import (
    PlanNode,
    PlanStep,
    RouterStep,
    ParentChildEdge,
    StepEdge,
    RouteEdge,
)
from chuk_ai_planner.core.graph.types import RouterType
from chuk_ai_planner.core.store.memory import InMemoryGraphStore
from chuk_ai_planner.core.routing import RoutingExecutor, FunctionRegistry


async def main():
    print("=" * 70)
    print("Function-Based Routing Example")
    print("=" * 70)

    graph = InMemoryGraphStore()

    # Create a function registry
    registry = FunctionRegistry()

    # Register routing functions
    @registry.register("priority_router")
    def calculate_priority(context):
        """Route based on urgency score."""
        urgency = context.get("urgency", 0)
        if urgency >= 8:
            return "critical"
        elif urgency >= 5:
            return "urgent"
        else:
            return "normal"

    @registry.register("quality_router")
    def check_quality(context):
        """Route based on quality metrics."""
        quality = context.get("quality_score", 0)
        errors = context.get("error_count", 0)

        if quality > 0.9 and errors == 0:
            return "excellent"
        elif quality > 0.7:
            return "good"
        else:
            return "needs_work"

    print(f"\n✅ Created function registry with {len(registry.list())} functions:")
    for func_name in registry.list():
        print(f"   - {func_name}")

    # 1. Create a Plan
    plan = PlanNode(
        title="Task Priority Management",
        description="Route tasks based on priority",
    )
    await graph.add_node(plan)
    print(f"\n✅ Created Plan: {plan.title}")

    # 2. Create analysis step
    analyze_step = PlanStep(description="Analyze task", index="1")
    await graph.add_node(analyze_step)
    await graph.add_edge(ParentChildEdge(src=plan.id, dst=analyze_step.id))

    # 3. Create Router Step with function routing
    router = RouterStep(
        router_type=RouterType.FUNCTION,
        description="Route based on priority",
        routes=["critical", "urgent", "normal"],
        router_function="priority_router",  # Reference to registered function
    )
    await graph.add_node(router)
    await graph.add_edge(ParentChildEdge(src=plan.id, dst=router.id))
    await graph.add_edge(StepEdge(src=analyze_step.id, dst=router.id))

    print("\n✅ Created Router:")
    print(f"   Type: {router.router_type}")
    print(f"   Function: {router.router_function}")
    print(f"   Routes: {router.routes}")

    # 4. Create route paths
    critical_step = PlanStep(description="Handle critical task immediately", index="2a")
    urgent_step = PlanStep(description="Handle urgent task soon", index="2b")
    normal_step = PlanStep(description="Queue normal task", index="2c")

    await graph.add_node(critical_step)
    await graph.add_node(urgent_step)
    await graph.add_node(normal_step)

    await graph.add_edge(
        RouteEdge(src=router.id, dst=critical_step.id, route_key="critical")
    )
    await graph.add_edge(
        RouteEdge(src=router.id, dst=urgent_step.id, route_key="urgent")
    )
    await graph.add_edge(
        RouteEdge(
            src=router.id, dst=normal_step.id, route_key="normal", is_default=True
        )
    )

    print("\n✅ Created 3 route paths:")
    print(f"   → critical: {critical_step.description}")
    print(f"   → urgent: {urgent_step.description}")
    print(f"   → normal: {normal_step.description} (default)")

    # 5. Test routing with different priority levels
    print("\n" + "=" * 70)
    print("Testing Function Routing")
    print("=" * 70)

    routing_executor = RoutingExecutor(graph, function_registry=registry)

    # Test 1: Critical urgency
    print("\n🔥 TEST 1: Critical Urgency (9)")
    context_critical = {"urgency": 9, "task": "System down"}
    decision = await routing_executor.evaluate_route(router, context_critical)
    print(f"   Route chosen: {decision.route_key}")
    print(f"   Target: {decision.target_step_id[:8]}")
    assert decision.route_key == "critical"
    print("   ✅ Correctly routed to critical!")

    # Test 2: Urgent
    print("\n⚡ TEST 2: Urgent (6)")
    context_urgent = {"urgency": 6, "task": "Bug fix needed"}
    decision = await routing_executor.evaluate_route(router, context_urgent)
    print(f"   Route chosen: {decision.route_key}")
    print(f"   Target: {decision.target_step_id[:8]}")
    assert decision.route_key == "urgent"
    print("   ✅ Correctly routed to urgent!")

    # Test 3: Normal
    print("\n📋 TEST 3: Normal (3)")
    context_normal = {"urgency": 3, "task": "Documentation update"}
    decision = await routing_executor.evaluate_route(router, context_normal)
    print(f"   Route chosen: {decision.route_key}")
    print(f"   Target: {decision.target_step_id[:8]}")
    assert decision.route_key == "normal"
    print("   ✅ Correctly routed to normal!")

    # 6. Test the quality_router function
    print("\n" + "=" * 70)
    print("Testing Quality Router")
    print("=" * 70)

    quality_router = RouterStep(
        router_type=RouterType.FUNCTION,
        description="Route based on quality",
        routes=["excellent", "good", "needs_work"],
        router_function="quality_router",
    )
    await graph.add_node(quality_router)

    excellent_step = PlanStep(description="Deploy to production", index="3a")
    good_step = PlanStep(description="Manual review", index="3b")
    needs_work_step = PlanStep(description="Send back for revision", index="3c")

    await graph.add_node(excellent_step)
    await graph.add_node(good_step)
    await graph.add_node(needs_work_step)

    await graph.add_edge(
        RouteEdge(src=quality_router.id, dst=excellent_step.id, route_key="excellent")
    )
    await graph.add_edge(
        RouteEdge(src=quality_router.id, dst=good_step.id, route_key="good")
    )
    await graph.add_edge(
        RouteEdge(
            src=quality_router.id,
            dst=needs_work_step.id,
            route_key="needs_work",
            is_default=True,
        )
    )

    print("\n✅ Created quality router")

    # Test excellent quality
    print("\n⭐ TEST: Excellent Quality (score=0.95, errors=0)")
    decision = await routing_executor.evaluate_route(
        quality_router, {"quality_score": 0.95, "error_count": 0}
    )
    print(f"   Route: {decision.route_key}")
    assert decision.route_key == "excellent"
    print("   ✅ Correctly routed to excellent!")

    # Test good quality
    print("\n👍 TEST: Good Quality (score=0.8, errors=1)")
    decision = await routing_executor.evaluate_route(
        quality_router, {"quality_score": 0.8, "error_count": 1}
    )
    print(f"   Route: {decision.route_key}")
    assert decision.route_key == "good"
    print("   ✅ Correctly routed to good!")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✅ FunctionRegistry for managing routing functions")
    print("✅ Decorator-based registration (@registry.register)")
    print("✅ String-based function references in RouterStep")
    print("✅ Multiple routing functions in one plan")
    print("✅ Custom routing logic with full context access")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
