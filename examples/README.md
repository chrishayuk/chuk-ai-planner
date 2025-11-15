# Pure Pydantic Graph Examples

This directory contains examples demonstrating the **pure Pydantic graph structure** in chuk-ai-planner.

## Key Concepts

The graph is built on **typed Pydantic models** with:
- ✅ **No dictionary goop** - Direct field access, not `.data.get()`
- ✅ **Type-safe enums** - `NodeType.PLAN`, not `"plan"`
- ✅ **Immutable nodes** - Frozen Pydantic models
- ✅ **Domain-agnostic** - Works for any planning workflow

## Quick Start Examples

### 1. Basic Graph Structure
**File:** `01_basic_graph.py`

Learn the fundamentals:
- Creating nodes (PlanNode, PlanStep)
- Creating edges (ParentChildEdge, StepEdge)
- Using typed fields instead of dictionaries

### 2. Conditional Routing
**File:** `02_conditional_routing.py`

Build plans with conditional logic:
- RouterStep nodes
- RouteEdge connections
- Expression-based routing

### 3. Tool Execution
**File:** `03_tool_execution.py`

Execute tools within plans:
- ToolCall nodes
- TaskRun nodes for results
- Linking tools to plan steps

### 4. Complete Workflow
**File:** `04_complete_workflow.py`

End-to-end example:
- Multi-step plans
- Conditional routing
- Tool execution
- Result variables

## Graph Structure

### Node Types

```python
from chuk_ai_planner.graph import (
    SessionNode,    # Top-level session/conversation
    PlanNode,       # A plan with steps
    PlanStep,       # A single step in a plan
    RouterStep,     # Conditional routing step
    ToolCall,       # A tool to execute
    TaskRun,        # Result of tool execution
    SummaryNode,    # Summary/checkpoint
)
```

### Edge Types

```python
from chuk_ai_planner.graph import (
    ParentChildEdge,  # Hierarchical relationship
    StepEdge,         # Step dependency (execution order)
    PlanLinkEdge,     # Plan to step connection
    RouteEdge,        # Conditional route
    NextEdge,         # Sequential flow
    CustomEdge,       # Custom relationships
)
```

### Type Enums

```python
from chuk_ai_planner.graph.types import (
    NodeType,      # Node kind enumeration
    EdgeType,      # Edge kind enumeration
    RouterType,    # Router types (EXPRESSION, LLM, FUNCTION)
    StepStatus,    # Step status (PENDING, RUNNING, COMPLETED, etc.)
)
```

## Running Examples

```bash
# Basic examples
python examples/01_basic_graph.py
python examples/02_conditional_routing.py
python examples/03_tool_execution.py
python examples/04_complete_workflow.py
```

## Legacy Examples

The `legacy/` directory contains older examples using deprecated patterns.
These are kept for reference but should not be used as templates for new code.
