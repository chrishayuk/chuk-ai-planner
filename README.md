# chuk-ai-planner

A powerful graph-based planning and execution framework for AI agents.

## Overview

`chuk-ai-planner` is a Python package that provides a flexible, graph-based approach to planning and executing AI agent workflows. It allows you to define plans composed of hierarchical steps, link them to tool calls, and execute them with full traceability.

The package models plans, steps, tools, results, and other components as nodes in a directed graph, with edges representing relationships between them. This approach enables complex workflows with dependency management, parallel execution, and detailed visualization.

## Key Features

- **Graph-based Plan Representation**: Model plans as interconnected nodes and edges
- **Hierarchical Planning**: Create nested steps and sub-steps with dependencies
- **Tool Execution Framework**: Clean abstraction for executing tools within plans
- **Parallel Execution**: Automatic parallelization of independent steps
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

### Plan DSL

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
    validate_step=my_validator
)

# Generate a plan from a prompt
plan, plan_id, graph = await agent.plan_into_graph("Research the history of AI")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.