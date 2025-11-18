# chuk-ai-planner

**The Universal Agent Runtime for the CHUK Ecosystem**

A powerful, graph-based planning and execution framework for AI agents with pluggable tool execution backends supporting MCP, ACP, and local tools.

## Overview

`chuk-ai-planner` is a production-ready Python package that provides a flexible, graph-based approach to planning and executing AI agent workflows. It allows you to define plans composed of hierarchical steps, execute them with full traceability, and leverage any tool from the CHUK ecosystem—whether it's a local Python function, an MCP server, or an ACP agent.

The package models plans, steps, tools, results, and other components as nodes in a directed graph, with edges representing relationships between them. This approach enables complex workflows with dependency management, parallel execution, conditional routing, and detailed visualization.

## ✨ What's New in v0.2

- **🎯 CTP-First Architecture**: Single unified registry via chuk-tool-processor
- **📦 Pydantic-Native**: Type-safe, validated data structures throughout
- **⚡ Async-Native**: Built for high-performance asynchronous execution
- **🌐 Universal Tool Support**: Python functions, MCP (Notion, GitHub), ACP agents, containers
- **🛡️ Built-in Reliability**: Automatic retries, caching, rate limiting for ALL tools
- **🔧 Simpler API**: One way to register tools, one execution path

## Key Features

### Orchestration & Planning
- **Graph-based Plan Representation**: Model plans as interconnected nodes and edges
- **Hierarchical Planning**: Create nested steps and sub-steps with dependencies
- **Conditional Routing**: Expression-based, function-based, and LLM-based routing
- **Parallel Execution**: Automatic parallelization of independent steps
- **Variable Substitution**: Dynamic value resolution with `${variable}` syntax
- **Dependency Management**: Automatic dependency resolution from variable flow

### Tool Execution
- **Universal Execution**: All tools (Python, MCP, ACP, containers) via chuk-tool-processor
- **Automatic Reliability**: Retries, caching, rate limiting built-in
- **Pydantic Models**: Type-safe request/result models
- **Error Handling**: Graceful error recovery and reporting
- **OpenTelemetry Ready**: Built-in tracing and observability

### Integration & Extensibility
- **UniversalPlan API**: Modern async-first interface for plan creation and execution
- **JobManager Orchestration**: High-level API for managing AI-powered workflows
- **LLM Integration**: Generate plans from natural language using gpt-5-mini
- **Session Tracing**: Track execution with detailed event logs
- **Visualization**: Console-based and graphical visualizations
- **Flexible Storage**: In-memory storage with extensible interfaces

## Installation

```bash
pip install chuk-ai-planner

# Or with uv
uv add chuk-ai-planner
```

## Quick Start

### Example 1: Simple Python Functions

```python
import asyncio
from chuk_ai_planner.core.planner.universal_plan import UniversalPlan
from chuk_ai_planner.core.planner.universal_plan_executor import UniversalExecutor
from chuk_ai_planner.core.store.memory import InMemoryGraphStore

async def add_numbers(a: int, b: int) -> dict:
    """Add two numbers - gets automatic retries, caching, rate limiting."""
    return {"sum": a + b}

async def main():
    # Create executor
    graph = InMemoryGraphStore()
    executor = UniversalExecutor(graph_store=graph)

    # Register tool (async now!)
    await executor.register_tool("add", add_numbers)

    # Create and execute plan
    plan = UniversalPlan(title="Simple Math", graph=graph)
    await plan.add_tool_step(
        title="Add 5 + 3",
        tool="add",
        args={"a": 5, "b": 3},
        result_variable="result",
    )
    plan_id = await plan.save()

    results = await executor.execute_plan_by_id(plan_id)
    print(f"Result: {results['variables']['result']}")  # {"sum": 8}

asyncio.run(main())
```

### Example 2: Using chuk-tool-processor (MCP/ACP Support)

```python
import asyncio
from chuk_tool_processor import ToolProcessor, tool
from chuk_ai_planner.core.planner.universal_plan import UniversalPlan
from chuk_ai_planner.core.planner.universal_plan_executor import UniversalExecutor
from chuk_ai_planner.core.store.memory import InMemoryGraphStore

# Define tool using CTP's @tool decorator
@tool(name="multiply")
class Multiply:
    """Multiply two numbers."""
    async def execute(self, a: int, b: int) -> dict:
        return {"product": a * b}

async def main():
    # Create executor with ToolProcessor backend (CTP-first!)
    async with ToolProcessor() as processor:
        graph = InMemoryGraphStore()
        executor = UniversalExecutor(
            graph_store=graph,
            processor=processor,
        )

        # Create and execute plan
        plan = UniversalPlan(title="Math Pipeline", graph=graph)
        await plan.add_tool_step(
            title="Multiply 4 * 5",
            tool="multiply",
            args={"a": 4, "b": 5},
            result_variable="result",
        )
        plan_id = await plan.save()

        results = await executor.execute_plan_by_id(plan_id)
        print(f"Result: {results['variables']['result']}")  # {"product": 20}

asyncio.run(main())
```

### Example 3: MCP Tools (Notion, GitHub, etc.)

```python
from chuk_tool_processor import ToolProcessor
from chuk_tool_processor.mcp import setup_mcp_stdio
from chuk_ai_planner.core.planner.universal_plan_executor import UniversalExecutor

async def main():
    # Setup MCP server
    processor = ToolProcessor()
    await setup_mcp_stdio(
        processor,
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": token},
    )

    # Create executor with MCP namespace (CTP-first!)
    executor = UniversalExecutor(
        processor=processor,
        namespace="github",  # Tools prefixed as "github:*"
    )

    # Use GitHub MCP tools
    plan = UniversalPlan(title="GitHub Operations", graph=executor.graph_store)
    await plan.add_tool_step(
        title="Create issue",
        tool="create_issue",  # Resolved as "github:create_issue"
        args={"repo": "org/repo", "title": "Bug", "body": "Description"},
    )
    plan_id = await plan.save()

    results = await executor.execute_plan_by_id(plan_id)
```

### Example 4: Variable Flow & Dependencies

```python
plan = UniversalPlan(title="Data Pipeline", graph=graph)

# Step 1: Fetch data
await plan.add_tool_step(
    title="Fetch data from API",
    tool="api_fetch",
    args={"endpoint": "/data"},
    result_variable="raw_data",
)

# Step 2: Transform (depends on step 1 via variable reference)
await plan.add_tool_step(
    title="Transform data",
    tool="transform",
    args={"data": "${raw_data}"},  # Variable reference
    result_variable="transformed",
)

# Step 3: Save (depends on step 2)
await plan.add_tool_step(
    title="Save to database",
    tool="db_save",
    args={"data": "${transformed}", "table": "results"},
)

plan_id = await plan.save()
results = await executor.execute_plan_by_id(plan_id)
```

## Core Concepts

### Execution Backends

The planner uses **CTP-first architecture** for universal tool execution:

#### ToolProcessorBackend (Default in v0.2+)
- Executes ALL tools via chuk-tool-processor
- Supports Python functions, MCP servers, ACP agents, containers
- Built-in retries, caching, rate limiting for EVERYTHING
- Universal, production-ready tool execution
- Type-safe with Pydantic models

```python
# All tools get automatic reliability features!
executor = UniversalExecutor()  # Uses ToolProcessorBackend by default
await executor.register_tool("my_tool", my_function)

# Or provide a custom processor
executor = UniversalExecutor(processor=ToolProcessor())
```

**Migration from v0.1:** LocalFunctionBackend has been removed. All tools now execute via CTP, which provides the same local execution plus MCP/ACP support and automatic reliability. See [MIGRATION_V0.2.md](MIGRATION_V0.2.md) for details.

### Pydantic Models

All execution uses type-safe Pydantic models:

```python
from chuk_ai_planner.execution import ToolExecutionRequest, ToolExecutionResult

# Request (immutable, validated)
request = ToolExecutionRequest(
    tool_name="greet",
    args={"name": "Claude"},
    step_id="step-001",
    session_id="session-123",
)

# Result (immutable, validated)
result = await backend.execute_tool(request)

# Clean property access (not dict keys!)
if result.success:
    data = result.result
else:
    error = result.error
```

### Best Practices

#### Graph Store Isolation

When executing multiple plans with a shared executor, **each plan should have its own graph store** to avoid variable conflicts:

```python
# ✅ RECOMMENDED: One graph per plan
executor = UniversalExecutor()
await executor.register_tool("my_tool", my_function)

for i in range(3):
    graph = InMemoryGraphStore()  # Fresh graph for each plan
    plan = UniversalPlan(title=f"Plan {i}", graph=graph)
    await plan.add_tool_step(...)
    result = await executor.execute_plan(plan)
    # Each plan has isolated variables and state
```

```python
# ❌ AVOID: Sharing graph across plans
graph = InMemoryGraphStore()  # Shared graph
executor = UniversalExecutor(graph_store=graph)

for i in range(3):
    plan = UniversalPlan(title=f"Plan {i}", graph=graph)  # Same graph!
    # Can cause variable conflicts between plans
```

**Why?** Each plan maintains its own variable context and execution state. Sharing a graph store can lead to:
- Variable name conflicts between plans
- Steps from one plan being executed in another
- Unexpected behavior in parallel execution scenarios

**When to share:** You can share an executor instance (and its registered tools) across multiple plans - just give each plan its own graph store.

#### Tool Function Signatures

In v0.2, tools registered with `register_tool()` receive arguments as **direct parameters** (not an args dict):

```python
# ✅ CORRECT: Direct parameters with type hints
async def my_tool(name: str, age: int = 0) -> dict:
    """A well-typed tool function."""
    return {"greeting": f"Hello {name}, age {age}"}

# Tool is called with: my_tool(name="Alice", age=30)
```

```python
# ❌ OLD v0.1 PATTERN: Args dictionary
async def my_tool(args: Dict[str, Any]) -> dict:
    name = args.get("name", "")
    age = args.get("age", 0)
    return {"greeting": f"Hello {name}, age {age}"}

# This won't work in v0.2 - CTP unpacks as **kwargs
```

**Why?** CTP's `register_fn_tool` unpacks arguments as `**kwargs`, so your function signature should match what callers will pass. This provides:
- Better type checking and IDE autocomplete
- Automatic validation of required parameters
- Self-documenting tool interfaces
- Pydantic schema generation for tool discovery

See [MIGRATION_V0.2.md](MIGRATION_V0.2.md) for migration details.

### Execution Flow

#### How Plans Execute

When you call `await executor.execute_plan(plan)`, here's what happens:

1. **Graph Loading** - Plan structure loaded from graph store (steps, dependencies, tool calls)
2. **Topological Sort** - Steps ordered based on dependencies for parallel execution
3. **Variable Resolution** - `${variable}` placeholders resolved from context
4. **Tool Execution** - Each step's tools executed via CTP with automatic retries
5. **Result Storage** - Results stored in variables for downstream steps
6. **Parallel Batching** - Independent steps execute in parallel batches

```python
# Example execution flow
plan = UniversalPlan(title="Data Pipeline", graph=graph)

# Step 1: Fetch data (no dependencies - runs first)
await plan.add_tool_step(
    title="Fetch users",
    tool="fetch_users",
    args={},
    result_variable="users"
)

# Step 2 & 3: Process data (depend on step 1 - run in parallel)
await plan.add_tool_step(
    title="Validate users",
    tool="validate",
    args={"data": "${users}"},  # Variable substitution!
    result_variable="validated"
)

await plan.add_tool_step(
    title="Enrich users",
    tool="enrich",
    args={"data": "${users}"},  # Same variable, parallel execution!
    result_variable="enriched"
)

# Step 4: Combine (depends on 2 & 3 - runs after both complete)
await plan.add_tool_step(
    title="Combine results",
    tool="combine",
    args={
        "validated": "${validated}",
        "enriched": "${enriched}"
    },
    result_variable="final"
)

# Execute: Steps 1 → [2, 3 in parallel] → 4
result = await executor.execute_plan(plan)
```

#### Execution Results

The `execute_plan()` method returns a dict with:

```python
{
    "success": True,  # Overall success/failure
    "variables": {    # All result variables from steps
        "users": [...],
        "validated": {...},
        "enriched": {...},
        "final": {...}
    },
    "results": [...],  # Raw tool execution results
    "executed_steps": set(...),  # Step IDs that executed
}
```

#### Error Handling

CTP provides automatic retry and error handling:

```python
# Tool fails? CTP retries automatically!
result = await executor.execute_plan(plan)

if not result["success"]:
    # Check error details
    error = result.get("error")
    print(f"Plan failed: {error}")

    # Partial results still available
    completed_vars = result.get("variables", {})
    print(f"Completed before failure: {list(completed_vars.keys())}")
```

**Automatic Retry Behavior:**
- **Network errors**: 3 retries with exponential backoff (1s, 2s, 4s)
- **Rate limits**: Automatic backoff and retry
- **Timeouts**: Configurable per-tool timeout with retries
- **Circuit breaking**: Fast-fail after repeated failures

#### Parallel Execution

The executor automatically parallelizes independent steps:

```python
# These steps have no dependencies - execute in parallel!
await plan.add_tool_step("Fetch users", tool="fetch_users", result_variable="users")
await plan.add_tool_step("Fetch products", tool="fetch_products", result_variable="products")
await plan.add_tool_step("Fetch orders", tool="fetch_orders", result_variable="orders")

# Executor runs all 3 simultaneously, then proceeds to dependent steps
result = await executor.execute_plan(plan)
```

**Parallelization Rules:**
- Steps with no dependencies → Execute immediately in parallel
- Steps with dependencies → Wait for all dependencies to complete
- Tool calls within a step → Execute sequentially (by design)
- Multiple plans → Can execute concurrently with shared executor

### Conditional Routing

The framework supports three types of conditional routing:

#### 1. Expression-Based Routing

```python
from chuk_ai_planner.core.graph import RouterStep, RouteEdge
from chuk_ai_planner.core.graph.types import RouterType

router = RouterStep(
    router_type=RouterType.EXPRESSION,
    description="Route based on priority",
    routes=["high", "medium", "low"],
    router_expression="priority"
)

await graph.add_edge(RouteEdge(src=router.id, dst=high_step.id, route_key="high"))
```

#### 2. Function-Based Routing

```python
from chuk_ai_planner.core.routing import FunctionRegistry

registry = FunctionRegistry()

@registry.register("priority_router")
def calculate_priority(context):
    urgency = context.get("urgency", 0)
    return "critical" if urgency >= 8 else "normal"

router = RouterStep(
    router_type=RouterType.FUNCTION,
    routes=["critical", "normal"],
    router_function="priority_router"
)
```

#### 3. LLM-Based Routing

```python
router = RouterStep(
    router_type=RouterType.LLM,
    description="Classify support request",
    routes=["technical", "billing", "general"],
    router_prompt="Classify this request: {user_message}",
    router_model="gpt-5-mini"
)
```

### JobManager API (High-Level Orchestration)

```python
from chuk_ai_planner.jobs import JobManager
from chuk_ai_planner.agents.graph_plan_agent import GraphPlanAgent

# Set up job manager
planner = GraphPlanAgent(graph=graph, system_prompt="...", validate_step=lambda s: (True, ""))
executor = UniversalExecutor(graph_store=graph)
manager = JobManager(planner=planner, executor=executor, graph_store=graph)

# One-shot execution
run = await manager.run_job("Analyze customer feedback and generate report")

# Or step-by-step
job = await manager.create_job("Deploy to production", tags=["deployment"])
run = await manager.plan_job(job.id)
result = await manager.start_job(job.id)

# Monitor and control
jobs = await manager.list_jobs(status=[JobStatus.RUNNING])
await manager.resume_job(job.id)
```

## Graph Model

The framework models entities as nodes in a graph:

- **PlanNode**: Overall plan
- **PlanStep**: Individual steps
- **RouterStep**: Conditional routing points
- **ToolCall**: Tool invocations
- **TaskRun**: Execution results
- **SessionNode**: Execution sessions

Edges represent relationships:
- **ParentChildEdge**: Hierarchical structure
- **NextEdge**: Sequential ordering
- **PlanLinkEdge**: Plan-to-step connections
- **RouteEdge**: Conditional routing paths
- **StepEdge**: Step dependencies

## Architecture (v0.2 CTP-First)

```
┌─────────────────────────────────────────────────┐
│  UniversalPlan                                  │
│  - Plan creation & validation                   │
│  - Dependency resolution                        │
│  - Variable flow tracking                       │
│  - Graph-based storage                          │
└────────────────┬────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────┐
│  UniversalExecutor                              │
│  - Plan orchestration                           │
│  - Parallel execution                           │
│  - CTP-first (ToolProcessorBackend by default)  │
│  - Async registration (await register_tool)     │
└────────────────┬────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────┐
│  ToolProcessorBackend (chuk-tool-processor)     │
│  ┌───────────────────────────────────────────┐  │
│  │ Universal Tool Execution                  │  │
│  │ - Python functions (register_fn_tool)     │  │
│  │ - MCP servers (Notion, GitHub, etc.)      │  │
│  │ - ACP agents (distributed)                │  │
│  │ - Container-based tools                   │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │ Automatic Reliability (for ALL tools)     │  │
│  │ - Retries with exponential backoff        │  │
│  │ - Caching with TTL                        │  │
│  │ - Rate limiting (global + per-tool)       │  │
│  │ - Circuit breaking                        │  │
│  │ - OpenTelemetry tracing                   │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

**Key Benefits:**
- **Single execution path** - All tools (Python, MCP, ACP, containers) use the same reliable infrastructure
- **Automatic reliability** - Every tool gets retries, caching, rate limiting without any extra code
- **Type-safe** - Pydantic models throughout (ToolExecutionRequest → ToolExecutionResult)
- **Production-ready** - Built-in observability, circuit breakers, and distributed execution support

## Examples

See the `examples/` directory for complete working examples:

- **18_simple_ctp_example.py**: Pydantic-native execution with CTP
- **16_universal_with_tools.py**: UniversalPlan with tool execution
- **15_conditional_routing_*.py**: Expression, function, and LLM routing
- **14_job_manager.py**: High-level job orchestration
- And many more!

## Advanced Features

### Custom Graph Stores

Implement the `GraphStore` interface for custom persistence:

```python
from chuk_ai_planner.core.store.base import GraphStore

class MyDatabaseGraphStore(GraphStore):
    async def add_node(self, node):
        # Your implementation
        ...
```

### GraphAwareToolProcessor (LLM Integration)

Process LLM messages with tool_calls:

```python
from chuk_ai_planner.processor import GraphAwareToolProcessor

processor = GraphAwareToolProcessor(session_id, graph_store)
results = await processor.process_llm_message(
    llm_response,
    llm_call_fn=llm.call,
)
```

### Visualization

```python
from chuk_ai_planner.utils.visualization import (
    print_graph_structure,
    print_session_events,
)

print_graph_structure(graph)
print_session_events(session)
```

## Documentation

- **CTP_INTEGRATION.md**: Comprehensive CTP integration guide
- **EXECUTION_ARCHITECTURE_UPDATE.md**: Architecture overview
- **EXAMPLES_GUIDE.md**: Example documentation
- **QUICK_REFERENCE.md**: API quick reference

## Testing

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=src/chuk_ai_planner

# Type checking
make typecheck

# Linting
make lint
```

## Requirements

- Python 3.11+
- chuk-session-manager
- chuk-tool-processor (for MCP/ACP support)
- Pydantic 2.0+
- OpenAI SDK (for LLM features)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`make test`)
5. Run type checking (`make typecheck`)
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Roadmap

See `ROADMAP_TO_EXCELLENCE.md` for the complete roadmap to making this the world's best LLM planner.

**Highlights:**
- ✅ Pluggable execution backends (COMPLETE)
- ✅ Pydantic-native models (COMPLETE)
- ✅ Async-native architecture (COMPLETE)
- ✅ MCP/ACP integration (COMPLETE)
- 🚧 Tool schema discovery
- 🚧 Advanced optimization engine
- 🚧 Distributed execution
- 🚧 Plan marketplace

---

**Built with ❤️ for the AI agent community**
