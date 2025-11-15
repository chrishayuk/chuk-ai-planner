# Examples Guide

## Pure Pydantic Graph Examples ✨

All examples demonstrate the **clean, type-safe, pure Pydantic graph structure** with:
- ✅ No dictionary goop
- ✅ Typed fields throughout
- ✅ Enums instead of strings
- ✅ Immutable, validated models

## Quick Start Examples

### Running the Examples

```bash
# Basic graph structure
python examples/01_basic_graph.py

# Conditional routing
python examples/02_conditional_routing.py

# Tool execution
python examples/03_tool_execution.py

# LLM extension (domain-specific nodes)
python examples/05_llm_extension.py
```

## Example Overview

### 1. Basic Graph Structure (`01_basic_graph.py`)

**Learn:**
- Creating nodes with typed fields
- Creating edges
- Querying the graph
- Immutability

**Key Code:**
```python
# Typed fields, not data={}!
plan = PlanNode(
    title="My Plan",
    description="A clean plan",
    variables={"key": "value"}
)

# Direct field access!
print(plan.title)  # Not plan.data.get("title")
```

### 2. Conditional Routing (`02_conditional_routing.py`)

**Learn:**
- RouterStep with expression-based routing
- RouteEdge connections
- Multiple execution paths
- RouterType enum

**Key Code:**
```python
# Typed router
router = RouterStep(
    router_type=RouterType.EXPRESSION,  # Enum!
    condition="${score} > 0.7",
    routes=["high", "low"],
    route_mapping={True: "high", False: "low"}
)

# Typed route edge
graph.add_edge(RouteEdge(
    src=router.id,
    dst=next_step.id,
    route_key="high"
))
```

### 3. Tool Execution (`03_tool_execution.py`)

**Learn:**
- ToolCall nodes
- TaskRun nodes for results
- PlanLinkEdge connections
- Result tracking

**Key Code:**
```python
# Typed tool call
tool = ToolCall(
    name="my_tool",
    args={"input": "value"}
)

# Typed task result
result = TaskRun(
    tool_call_id=tool.id,
    status="success",
    result={"output": "data"},
    started_at=datetime.now(),
    completed_at=datetime.now()
)
```

### 5. LLM Extension (`05_llm_extension.py`)

**Learn:**
- Domain-specific extensions
- UserMessage, AssistantMessage, SystemMessage
- Keeping core graph clean
- Extension pattern for other projects

**Key Code:**
```python
# Core graph (domain-agnostic)
from chuk_ai_planner.graph import PlanNode, PlanStep

# LLM extension (domain-specific)
from chuk_ai_planner.graph.nodes.llm import UserMessage, AssistantMessage

# Use both together
user_msg = UserMessage(
    content="What's the weather?",
    role="user"
)

plan = PlanNode(title="Weather Plan")
```

## Graph Node Types

### Core Nodes (Domain-Agnostic)

```python
from chuk_ai_planner.graph import (
    SessionNode,    # Top-level session
    PlanNode,       # A plan with steps
    PlanStep,       # A single step
    RouterStep,     # Conditional routing
    ToolCall,       # Tool to execute
    TaskRun,        # Tool execution result
    SummaryNode,    # Summary/checkpoint
)
```

### LLM Extension Nodes

```python
from chuk_ai_planner.graph.nodes.llm import (
    UserMessage,      # User message in chat
    AssistantMessage, # Assistant response
    SystemMessage,    # System prompt
)
```

## Graph Edge Types

```python
from chuk_ai_planner.graph import (
    ParentChildEdge,  # Hierarchical relationship
    StepEdge,         # Step dependency
    PlanLinkEdge,     # Plan to step link
    RouteEdge,        # Conditional route
    NextEdge,         # Sequential flow
    CustomEdge,       # Custom relationships
)
```

## Type Enums

```python
from chuk_ai_planner.graph.types import (
    NodeType,      # Node kinds
    EdgeType,      # Edge kinds
    RouterType,    # EXPRESSION, LLM, FUNCTION
    StepStatus,    # PENDING, RUNNING, COMPLETED, etc.
)
```

## Extension Pattern

### For LLM Projects

```python
# Use core + LLM extension
from chuk_ai_planner.graph import PlanNode, PlanStep
from chuk_ai_planner.graph.nodes.llm import UserMessage
```

### For Other Domains (e.g., chuk-motion)

```python
# Create your own extension module
# chuk_motion/graph/nodes/video.py

from chuk_ai_planner.graph.nodes.base import GraphNode

class VideoNode(GraphNode):
    kind: Literal["video"] = "video"
    filename: str
    duration: float
    codec: str

# Then use it
from chuk_ai_planner.graph import PlanNode
from chuk_motion.graph.nodes import VideoNode

video = VideoNode(filename="intro.mp4", duration=5.0, codec="h264")
plan = PlanNode(title="Video Processing")
```

## Legacy Examples

The `examples/legacy/` directory contains older examples that:
- Use the old `models` module (deprecated)
- Use `.data.get()` patterns (avoid)
- Are kept for reference only

**Do not use legacy examples as templates for new code!**

## Key Principles

1. **No Dictionary Goop**
   ```python
   # ❌ Old way
   node.data.get("title")

   # ✅ New way
   node.title
   ```

2. **Type-Safe Enums**
   ```python
   # ❌ Old way
   kind="plan"

   # ✅ New way
   kind=NodeType.PLAN
   ```

3. **Typed Fields**
   ```python
   # ❌ Old way
   PlanStep(data={"description": "Do thing", "index": "1"})

   # ✅ New way
   PlanStep(description="Do thing", index="1")
   ```

4. **Immutable Models**
   ```python
   # ❌ Cannot do this
   node.title = "New Title"

   # ✅ Create updated copy
   updated = node.model_copy(update={"title": "New Title"})
   ```

5. **Domain Extensions**
   ```python
   # ✅ Core is domain-agnostic
   from chuk_ai_planner.graph import PlanNode

   # ✅ Extensions are domain-specific
   from chuk_ai_planner.graph.nodes.llm import UserMessage
   from chuk_motion.graph.nodes import VideoNode  # Future
   ```

## Next Steps

- Read the examples in order (01, 02, 03, 05)
- Run each example and study the output
- Look at the source code to see the patterns
- Build your own plans using these patterns
- Create domain extensions for your specific use case

## Questions?

- Check `examples/README.md` for more details
- Look at the test files in `tests/graph/` for more examples
- Read the source in `src/chuk_ai_planner/graph/`
