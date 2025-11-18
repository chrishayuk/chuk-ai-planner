#!/usr/bin/env python3
"""
Example 17: chuk-tool-processor Integration (v0.2 CTP-First)
============================================================

This example demonstrates the CTP-first architecture in v0.2, where ALL tools
(Python functions, MCP, ACP, containers) execute via chuk-tool-processor.

Key Features:
- Direct function registration (no @tool decorator needed!)
- Automatic retries, caching, rate limiting for ALL tools
- Type-safe function signatures with **kwargs unpacking
- Universal tool execution via CTP

The planner becomes pure orchestration while CTP handles all execution concerns.
"""

import asyncio
from chuk_ai_planner.core.planner.universal_plan import UniversalPlan
from chuk_ai_planner.core.planner.universal_plan_executor import UniversalExecutor
from chuk_ai_planner.core.store.memory import InMemoryGraphStore


# ============================================================================
# Define tools as simple Python functions (CTP-first pattern!)
# ============================================================================


async def fetch_user_data(user_id: int) -> dict:
    """Fetch user data from a mock API."""
    return {
        "user_id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com",
        "role": "admin" if user_id == 1 else "user",
    }


async def validate_permissions(role: str, action: str) -> dict:
    """Validate user permissions."""
    permissions = {
        "admin": ["read", "write", "delete"],
        "user": ["read"],
    }

    allowed = action in permissions.get(role, [])

    return {
        "role": role,
        "action": action,
        "allowed": allowed,
    }


async def execute_action(action: str, user_name: str, allowed: bool) -> dict:
    """Execute an action if permitted."""
    if not allowed:
        return {
            "status": "denied",
            "message": f"User '{user_name}' does not have permission for '{action}'",
        }

    return {
        "status": "success",
        "message": f"User '{user_name}' successfully executed '{action}'",
        "action": action,
    }


# ============================================================================
# Example 1: Basic CTP Integration
# ============================================================================


async def example_basic_ctp():
    """Basic example: Execute a plan using chuk-tool-processor (v0.2 CTP-first)."""
    print("\n" + "=" * 70)
    print("Example 1: Basic CTP Integration (v0.2)")
    print("=" * 70)

    # Create graph store
    graph = InMemoryGraphStore()

    # Create executor (uses CTP by default in v0.2!)
    executor = UniversalExecutor(graph_store=graph)

    # Register tools - they automatically get retries, caching, rate limiting!
    await executor.register_tool("fetch_user_data", fetch_user_data)
    await executor.register_tool("validate_permissions", validate_permissions)
    await executor.register_tool("execute_action", execute_action)

    print("✅ Tools registered with CTP (automatic reliability enabled!)")

    # Create plan
    plan = UniversalPlan(title="User Permission Check", graph=graph)

    # Step 1: Fetch user data
    await plan.add_tool_step(
        title="Fetch user data",
        tool="fetch_user_data",
        args={"user_id": 1},
        result_variable="user_data",
    )

    # Step 2: Validate permissions (depends on step 1)
    await plan.add_tool_step(
        title="Check permissions",
        tool="validate_permissions",
        args={
            "role": "${user_data.role}",
            "action": "delete",
        },
        result_variable="permission_check",
    )

    # Step 3: Execute action (depends on steps 1 and 2)
    await plan.add_tool_step(
        title="Execute action",
        tool="execute_action",
        args={
            "action": "delete",
            "user_name": "${user_data.name}",
            "allowed": "${permission_check.allowed}",
        },
        result_variable="result",
    )

    plan_id = await plan.save()

    print("\n📋 Executing plan with CTP backend...")
    results = await executor.execute_plan_by_id(plan_id)

    print("\n✅ Plan execution completed!")
    print(f"\nExecution successful: {results.get('success', False)}")

    if results.get("success"):
        # Access variables from results
        variables = results.get("variables", {})
        print(f"\nUser data: {variables.get('user_data')}")
        print(f"Permission check: {variables.get('permission_check')}")
        print(f"Final result: {variables.get('result')}")
    else:
        print(f"\n❌ Execution failed: {results.get('error')}")


# ============================================================================
# Example 2: Multiple Plans with Shared ToolProcessor
# ============================================================================


async def example_shared_executor():
    """Advanced example: Reusable executor for multiple plans (v0.2 pattern)."""
    print("\n" + "=" * 70)
    print("Example 2: Reusable Executor (v0.2)")
    print("=" * 70)

    # Create executor once (CTP-first!)
    executor = UniversalExecutor()

    # Register tools once - they're available for all plans
    await executor.register_tool("fetch_user_data", fetch_user_data)
    await executor.register_tool("validate_permissions", validate_permissions)

    print("✅ Tools registered once - reusable across all plans!")

    # Execute multiple independent plans with the same executor
    # Each plan gets its own graph to avoid variable conflicts
    for user_id in [1, 2, 3]:
        graph = InMemoryGraphStore()
        plan = UniversalPlan(title=f"Check User {user_id} Permissions", graph=graph)

        await plan.add_tool_step(
            title="Fetch user",
            tool="fetch_user_data",
            args={"user_id": user_id},
            result_variable="user",
        )

        await plan.add_tool_step(
            title="Check write permission",
            tool="validate_permissions",
            args={
                "role": "${user.role}",
                "action": "write",
            },
            result_variable="can_write",
        )

        await plan.save()

        print(f"\n🔄 Executing plan for user {user_id}...")
        result = await executor.execute_plan(plan)

        if result.get("success"):
            variables = result.get("variables", {})
            user = variables.get("user")
            can_write = variables.get("can_write")

            print(f"   User: {user['name']} (Role: {user['role']})")
            print(f"   Can write: {can_write['allowed']}")
        else:
            print(f"   ❌ Failed: {result.get('error')}")


# ============================================================================
# Example 3: Parallel Execution
# ============================================================================


async def example_parallel_execution():
    """Example: Parallel tool execution with CTP (v0.2)."""
    print("\n" + "=" * 70)
    print("Example 3: Parallel Execution (v0.2)")
    print("=" * 70)

    graph = InMemoryGraphStore()
    executor = UniversalExecutor(graph_store=graph)

    # Register tool
    await executor.register_tool("fetch_user_data", fetch_user_data)

    plan = UniversalPlan(title="Parallel User Fetching", graph=graph)

    # Add parallel steps (no dependencies between them)
    await plan.add_tool_step(
        title="Fetch user 1",
        tool="fetch_user_data",
        args={"user_id": 1},
        result_variable="user1",
    )

    await plan.add_tool_step(
        title="Fetch user 2",
        tool="fetch_user_data",
        args={"user_id": 2},
        result_variable="user2",
    )

    await plan.add_tool_step(
        title="Fetch user 3",
        tool="fetch_user_data",
        args={"user_id": 3},
        result_variable="user3",
    )

    plan_id = await plan.save()

    print("\n🚀 Executing parallel plan...")
    results = await executor.execute_plan_by_id(plan_id)

    if results.get("success"):
        variables = results.get("variables", {})
        print("\n✅ All users fetched in parallel!")
        print(f"   User 1: {variables['user1']['name']}")
        print(f"   User 2: {variables['user2']['name']}")
        print(f"   User 3: {variables['user3']['name']}")
    else:
        print(f"\n❌ Failed: {results.get('error')}")


# ============================================================================
# Main Entry Point
# ============================================================================


async def main():
    """Run all examples."""
    await example_basic_ctp()
    await example_shared_executor()
    await example_parallel_execution()

    print("\n" + "=" * 70)
    print("🎉 All examples completed successfully!")
    print("=" * 70)
    print(
        """
Key Takeaways (v0.2 CTP-First):
--------------------------------
1. chuk-ai-planner uses CTP-first architecture by default
2. ALL tools (Python, MCP, ACP, containers) execute via chuk-tool-processor
3. Simple: await executor.register_tool(name, func) - that's it!
4. Automatic reliability: retries, caching, rate limiting for ALL tools
5. No @tool decorator needed - just register Python functions directly
6. The planner remains pure orchestration

Next Steps:
-----------
- Add MCP tools via setup_mcp_stdio() / setup_mcp_sse()
- Add ACP agents
- Use container-based execution
- Leverage CTP's advanced features (circuit breakers, distributed execution)

See MIGRATION_V0.2.md for more details on the CTP-first architecture!
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
