# chuk-ai-planner

A powerful graph-based planning and execution framework for AI agents.

## Overview

`chuk-ai-planner` is a Python package that provides a flexible, graph-based approach to planning and executing AI agent workflows. It allows you to define plans composed of hierarchical steps, link them to tool calls, and execute them with full traceability.

The package models plans, steps, tools, results, and other components as nodes in a directed graph, with edges representing relationships between them. This approach enables complex workflows with dependency management, parallel execution, and detailed visualization.

## Key Features

- **Graph-based Plan Representation**: Model plans as interconnected nodes and edges
- **Hierarchical Planning**: Create nested steps and sub-steps with dependencies
- **Conditional Routing**: Expression-based, function-based, and LLM-based routing
- **UniversalPlan API**: Modern async-first interface for plan creation and execution
- **JobManager Orchestration**: High-level API for managing AI-powered workflows
- **Tool Execution Framework**: Clean abstraction for executing tools within plans
- **LLM Integration**: Generate plans from natural language using gpt-5-mini
- **Parallel Execution**: Automatic parallelization of independent steps
- **Variable Substitution**: Dynamic value resolution with {variable} syntax
- **Visualization Utilities**: Console-based and graphical visualizations
- **Session Tracing**: Track execution with detailed event logs
- **Flexible Storage**: In-memory storage with extensible interfaces

## Installation

```bash
pip install chuk-ai-planner
```

## Quick Start

Here's a simple example of creating and executing a plan:

```python
from chuk_ai_planner import Plan, GraphAwareToolProcessor
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.utils.visualization import print_graph_structure

# Create a plan with some steps
graph = InMemoryGraphStore()
plan = (
    Plan("Weather and calculation", graph=graph)
      .step("Check weather in New York").up()
      .step("Multiply 235.5 × 18.75").up()
)
plan_id = plan.save()

# Print the plan outline
print(plan.outline())

# Link tools to steps
# ... (code to link tool calls to steps)

# Execute the plan
processor = GraphAwareToolProcessor(session_id="session123", graph_store=graph)
# ... (code to register tools)
results = await processor.process_plan(plan_id, "assistant", lambda _: None)

# Visualize the executed plan
print_graph_structure(graph)
```

## Core Components

### UniversalPlan API (Recommended)

The modern UniversalPlan API provides a clean, async-first interface for creating and executing plans:

```python
from chuk_ai_planner.core.planner import UniversalPlan
from chuk_ai_planner.core.planner.universal_plan_executor import UniversalExecutor
from chuk_ai_planner.core.store.memory import InMemoryGraphStore

# Create a plan with tool and function steps
graph = InMemoryGraphStore()
plan = UniversalPlan(title="Data Processing Pipeline", graph=graph)

# Add steps with dependencies
await plan.add_tool_step(
    title="Fetch data from API",
    tool_name="api_fetch",
    args={"endpoint": "/data"},
    result_variable="raw_data"
)

await plan.add_function_step(
    title="Process data",
    function="process_data",
    args={"data": "{raw_data}"},  # Variable substitution
    depends_on=["1"],
    result_variable="processed_data"
)

# Save and execute
plan_id = await plan.save()
executor = UniversalExecutor(graph_store=graph)
results = await executor.execute(plan_id)
```

### Conditional Routing

The framework supports three types of conditional routing for dynamic plan execution:

#### 1. Expression-Based Routing

```python
from chuk_ai_planner.core.graph import RouterStep, RouteEdge
from chuk_ai_planner.core.graph.types import RouterType

# Create a router that evaluates expressions
router = RouterStep(
    router_type=RouterType.EXPRESSION,
    description="Route based on priority",
    routes=["high", "medium", "low"],
    router_expression="priority"  # Variable name
)

# Add route edges to different steps
await graph.add_edge(RouteEdge(src=router.id, dst=high_step.id, route_key="high"))
await graph.add_edge(RouteEdge(src=router.id, dst=medium_step.id, route_key="medium"))
await graph.add_edge(RouteEdge(src=router.id, dst=low_step.id, route_key="low", is_default=True))
```

#### 2. Function-Based Routing

```python
from chuk_ai_planner.core.routing import FunctionRegistry, RoutingExecutor

# Create and register a routing function
registry = FunctionRegistry()

@registry.register("priority_router")
def calculate_priority(context):
    urgency = context.get("urgency", 0)
    if urgency >= 8:
        return "critical"
    elif urgency >= 5:
        return "urgent"
    return "normal"

# Use in a router step
router = RouterStep(
    router_type=RouterType.FUNCTION,
    description="Route based on urgency",
    routes=["critical", "urgent", "normal"],
    router_function="priority_router"
)

# Execute with the function registry
executor = RoutingExecutor(graph, function_registry=registry)
decision = await executor.evaluate_route(router, {"urgency": 9})
```

#### 3. LLM-Based Routing

```python
# Create a router that uses LLM to make decisions
router = RouterStep(
    router_type=RouterType.LLM,
    description="Classify user request",
    routes=["technical", "billing", "general"],
    router_prompt="Classify this support request: {user_message}",
    router_model="gpt-5-mini"
)

# The router will call the LLM to decide which route to take
executor = RoutingExecutor(graph)
decision = await executor.evaluate_route(router, {"user_message": "My API key isn't working"})
# Returns: "technical"
```

### Plan DSL (Classic API)

The Plan Domain-Specific Language (DSL) allows you to define hierarchical plans with steps and dependencies:

```python
plan = (
    Plan("My Plan")
      .step("Step 1").up()
      .step("Step 2")
        .step("Step 2.1").up()
        .step("Step 2.2").up()
      .up()
      .step("Step 3", after=["1", "2"]).up()
)
```

### Graph Model

The framework models various entities as nodes in a graph:

- **PlanNode**: Represents the overall plan
- **PlanStep**: Individual steps in the plan
- **ToolCall**: Calls to external tools/functions
- **TaskRun**: Execution results of tool calls
- **SessionNode**: Represents a session
- **UserMessage/AssistantMessage**: Conversation messages

Edges represent relationships like parent-child, next, plan links, and step ordering.

### Execution

The `GraphAwareToolProcessor` handles plan execution:

- Processes plans in dependency order
- Executes tool calls
- Records results
- Generates session events

### Visualization

Visualization utilities help understand and debug plans:

- Text-based plan outlines
- Hierarchical session event display
- Graph structure visualization
- SVG graph generation

## Examples

### Creating a Plan with Dependencies

```python
plan = Plan("Research Task")
plan.step("Gather information").up()
plan.step("Analyze data", after=["1"]).up()
plan.step("Write report", after=["2"]).up()
plan_id = plan.save()
```

### Executing Tools in Parallel

```python
# Define a plan with parallel steps
plan = (
    Plan("Parallel Processing")
      .step("Process File A").up()
      .step("Process File B").up()
      .step("Combine Results", after=["1", "2"]).up()
)

# Execute with automatic parallelization
results = await processor.process_plan(plan_id, "assistant", lambda _: None)
```

## Advanced Usage

### Custom Graph Stores

Create custom graph stores by implementing the `GraphStore` interface:

```python
from chuk_ai_planner.store.base import GraphStore

class MyDatabaseGraphStore(GraphStore):
    # Implement required methods
    ...
```

### Plan Agents

Use the provided agents to generate plans from natural language:

```python
from chuk_ai_planner.agents.graph_plan_agent import GraphPlanAgent

agent = GraphPlanAgent(
    graph=graph,
    system_prompt="You are a planning assistant...",
    validate_step=lambda step: (True, ""),
    model="gpt-5-mini",  # Default model
    temperature=1.0  # Required for gpt-5-mini
)

# Generate a plan from a prompt
plan, plan_id, graph = await agent.plan_into_graph("Research the history of AI")
```

### JobManager API (High-Level Orchestration)

The JobManager provides a Manus-style orchestration API for managing AI-powered workflows:

```python
from chuk_ai_planner.jobs import JobManager
from chuk_ai_planner.agents.graph_plan_agent import GraphPlanAgent
from chuk_ai_planner.core.planner.universal_plan_executor import UniversalExecutor
from chuk_ai_planner.core.store.memory import InMemoryGraphStore

# Set up the job manager
graph = InMemoryGraphStore()
planner = GraphPlanAgent(graph=graph, system_prompt="...", validate_step=lambda s: (True, ""))
executor = UniversalExecutor(graph_store=graph)
manager = JobManager(planner=planner, executor=executor, graph_store=graph)

# One-shot execution: describe what you want and run it
run = await manager.run_job("Analyze customer feedback and generate report")

# Or step-by-step with more control
job = await manager.create_job("Deploy to production", tags=["deployment"])
run = await manager.plan_job(job.id)  # Generate execution plan
# ... review plan, get approvals ...
result = await manager.start_job(job.id)  # Execute

# Monitor jobs
jobs = await manager.list_jobs(status=[JobStatus.RUNNING])
status = await manager.get_job_status(job.id)

# Resume failed jobs
await manager.resume_job(job.id)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.