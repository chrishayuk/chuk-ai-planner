#!/usr/bin/env python3
"""
Example 18: Simple CTP Integration Example
==========================================

Demonstrates the new Pydantic-native, async-native execution backend
with chuk-tool-processor integration.

This example shows:
1. How to use the default LocalFunctionBackend
2. How to use the ToolProcessorBackend with CTP
3. The Pydantic-native execution flow
4. Async-native tool execution
"""

import asyncio


# ============================================================================
# Example 1: Default LocalFunctionBackend (Backward Compatible)
# ============================================================================


async def example_1_local_backend():
    """Example 1: Using the default LocalFunctionBackend."""
    print("\n" + "=" * 70)
    print("Example 1: Default LocalFunctionBackend")
    print("=" * 70)

    from chuk_ai_planner.core.planner.universal_plan import UniversalPlan
    from chuk_ai_planner.core.planner.universal_plan_executor import (
        UniversalExecutor,
    )
    from chuk_ai_planner.core.store.memory import InMemoryGraphStore

    # Define a simple tool
    async def add_numbers(args):
        """Add two numbers."""
        a = args.get("a", 0)
        b = args.get("b", 0)
        return {"sum": a + b, "operation": "add"}

    # Create executor (uses LocalFunctionBackend by default)
    graph = InMemoryGraphStore()
    executor = UniversalExecutor(graph_store=graph)
    await executor.register_tool("add_numbers", add_numbers)

    # Create plan
    plan = UniversalPlan(title="Simple Math", graph=graph)
    await plan.add_tool_step(
        title="Add 5 and 3",
        tool="add_numbers",
        args={"a": 5, "b": 3},
        result_variable="result",
    )
    plan_id = await plan.save()

    # Execute
    print("\n📋 Executing plan with LocalFunctionBackend...")
    results = await executor.execute_plan_by_id(plan_id)

    print(f"✅ Success: {results.get('success')}")
    print(f"Result: {results.get('variables', {}).get('result')}")


# ============================================================================
# Example 2: ToolProcessorBackend with Local Tools
# ============================================================================


async def example_2_ctp_backend():
    """Example 2: Using ToolProcessorBackend with local CTP tools."""
    print("\n" + "=" * 70)
    print("Example 2: ToolProcessorBackend with Local Tools")
    print("=" * 70)

    from chuk_tool_processor import ToolProcessor, tool
    from chuk_ai_planner.core.planner.universal_plan import UniversalPlan
    from chuk_ai_planner.core.planner.universal_plan_executor import (
        UniversalExecutor,
    )
    from chuk_ai_planner.core.store.memory import InMemoryGraphStore

    # Define tools using CTP's @tool decorator
    @tool(name="multiply")
    class Multiply:
        """Multiply two numbers."""

        async def execute(self, a: int, b: int) -> dict:
            return {"product": a * b, "operation": "multiply"}

    @tool(name="power")
    class Power:
        """Raise a to the power of b."""

        async def execute(self, base: int, exponent: int) -> dict:
            return {"result": base**exponent, "operation": "power"}

    # Create ToolProcessor
    processor = ToolProcessor()

    # Create executor with ToolProcessorBackend
    graph = InMemoryGraphStore()
    executor = await UniversalExecutor.with_tool_processor(
        graph_store=graph, processor=processor
    )

    # Create plan with dependencies
    plan = UniversalPlan(title="Math Pipeline", graph=graph)

    # Step 1: Multiply 4 * 5
    await plan.add_tool_step(
        title="Multiply 4 * 5",
        tool="multiply",
        args={"a": 4, "b": 5},
        result_variable="mult_result",
    )

    # Step 2: Raise result to power 2
    await plan.add_tool_step(
        title="Square the result",
        tool="power",
        args={
            "base": "${mult_result.product}",  # Variable reference
            "exponent": 2,
        },
        result_variable="final_result",
    )

    plan_id = await plan.save()

    # Execute
    print("\n📋 Executing plan with ToolProcessorBackend...")
    results = await executor.execute_plan_by_id(plan_id)

    print(f"✅ Success: {results.get('success')}")
    variables = results.get("variables", {})
    print(f"Multiplication result: {variables.get('mult_result')}")
    print(f"Final result: {variables.get('final_result')}")
    print(
        f"\nCalculation: (4 * 5)² = 20² = {variables.get('final_result', {}).get('result')}"
    )


# ============================================================================
# Example 3: Pydantic Models in Action
# ============================================================================


async def example_3_pydantic_models():
    """Example 3: Direct use of Pydantic execution models."""
    print("\n" + "=" * 70)
    print("Example 3: Pydantic-Native Execution")
    print("=" * 70)

    from chuk_ai_planner.execution import (
        LocalFunctionBackend,
        ToolExecutionRequest,
    )

    # Create backend
    backend = LocalFunctionBackend()

    # Define and register a tool
    async def greet(args):
        name = args.get("name", "World")
        return {"message": f"Hello, {name}!"}

    backend.register_tool("greet", greet)

    # Create a Pydantic request (type-safe, validated, immutable)
    request = ToolExecutionRequest(
        tool_name="greet",
        args={"name": "Claude"},
        step_id="step-001",
        session_id="session-123",
    )

    print("\n📤 Request (Pydantic model):")
    print(f"   Tool: {request.tool_name}")
    print(f"   Args: {request.args}")
    print(f"   Step ID: {request.step_id}")
    print(f"   Immutable: {request.model_config['frozen']}")

    # Execute (async-native)
    result = await backend.execute_tool(request)

    print("\n📥 Result (Pydantic model):")
    print(f"   Tool: {result.tool_name}")
    print(f"   Success: {result.success}")  # Property, not dict key!
    print(f"   Result: {result.result}")
    print(f"   Error: {result.error}")
    print(f"   Duration: {result.duration:.4f}s")
    print(f"   Cached: {result.cached}")
    print(f"   Immutable: {result.model_config['frozen']}")


# ============================================================================
# Example 4: Error Handling with Pydantic Models
# ============================================================================


async def example_4_error_handling():
    """Example 4: Error handling with Pydantic models."""
    print("\n" + "=" * 70)
    print("Example 4: Error Handling")
    print("=" * 70)

    from chuk_ai_planner.execution import (
        LocalFunctionBackend,
        ToolExecutionRequest,
    )

    backend = LocalFunctionBackend()

    # Tool that raises an error
    async def failing_tool(args):
        raise ValueError("Something went wrong!")

    backend.register_tool("failing_tool", failing_tool)

    # Execute
    request = ToolExecutionRequest(
        tool_name="failing_tool", args={}, step_id="step-error"
    )

    result = await backend.execute_tool(request)

    print("\n📥 Result:")
    print(f"   Success: {result.success}")  # False
    print(f"   Error: {result.error}")  # Error message
    print(f"   Result: {result.result}")  # None

    # Clean error checking with Pydantic property
    if not result.success:
        print(f"\n❌ Tool failed: {result.error}")


# ============================================================================
# Main Entry Point
# ============================================================================


async def main():
    """Run all examples."""
    await example_1_local_backend()
    await example_2_ctp_backend()
    await example_3_pydantic_models()
    await example_4_error_handling()

    print("\n" + "=" * 70)
    print("🎉 All examples completed!")
    print("=" * 70)
    print(
        """
Key Takeaways:
--------------
1. ✅ Pydantic-native: All requests/results use Pydantic models
2. ✅ Async-native: All execution is async
3. ✅ Type-safe: Models are validated and immutable
4. ✅ No dictionary goop: Access via properties, not dict keys
5. ✅ Backward compatible: LocalFunctionBackend is the default
6. ✅ CTP integration: Use with_tool_processor() for CTP tools

Next Steps:
-----------
- Add MCP tools via setup_mcp_stdio() / setup_mcp_sse()
- Use @tool decorator for local CTP tools
- Leverage CTP's built-in retries, caching, rate limiting
- Build complex multi-step plans with variable flow
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
