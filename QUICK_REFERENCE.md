# Quick Reference: Chuk-AI-Planner New Architecture

## What Changed?

The codebase transitioned from a monolithic structure to a **modular, Pydantic-based architecture**:

### Before: Old Structure (Deleted)
```
src/chuk_ai_planner/
  planner/                    # Mixed domain code
  graph/                      # Mixed domain code
  routing/                    # Mixed domain code
  store/                      # Mixed domain code
```

### After: New Structure (Clear Separation)
```
src/chuk_ai_planner/
  core/                       # Domain-agnostic infrastructure
    graph/                    # Pure Pydantic models (nodes + edges)
    planner/                  # Planning DSL + execution engine
    routing/                  # Conditional routing logic
    store/                    # Storage abstraction
  extensions/                 # Domain-specific features
    llm/                      # Chat/LLM nodes (optional)
```

---

## Core Principle: "No Dictionary Goop"

Every single field is explicitly typed and validated:

### Node Definition (Example)
```python
class PlanStep(GraphNode):
    kind: Literal[NodeType.PLAN_STEP] = NodeType.PLAN_STEP
    description: str
    index: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    max_retries: int = 0
    timeout_seconds: Optional[int] = None
    # ... 10+ more explicitly typed fields
```

NOT stored as `{"step_data": {...}}` with string keys. Every field is discoverable and type-checked.

---

## Module Breakdown

### 1. Graph System (`core/graph/`)
- **Purpose**: Type-safe node and edge definitions
- **Key Files**:
  - `types.py` - All enums (NodeType, EdgeType, RouterType, StepStatus, etc.)
  - `nodes/` - 7 node types organized by category
  - `edges/` - 8 edge types organized by relationship type
- **Total**: ~600 lines of pure Pydantic models

### 2. Planner (`core/planner/`)
- **Purpose**: Planning DSL and execution engine
- **Key Classes**:
  - `Plan` - Author-facing fluent builder API
  - `PlanExecutor` - Internal helper (NOT author-facing)
  - `PlanRegistry` - Plan storage/retrieval
  - `UniversalPlan` / `UniversalPlanExecutor` - Generic representations
- **Total**: ~2100 lines

### 3. Routing (`core/routing/`)
- **Purpose**: Conditional execution decision-making
- **Key Classes**:
  - `RoutingExecutor` - Evaluates conditions
  - `RoutingDecision` - Result of routing evaluation
- **Supports**: Expression-based, LLM-based, function-based routing

### 4. Storage (`core/store/`)
- **Purpose**: Abstract storage interface
- **Key Classes**:
  - `GraphStore` - Abstract base (fully async)
  - `InMemoryGraphStore` - Reference implementation
- **Operations**: add_node, get_node, add_edge, get_edges, etc.

### 5. LLM Extension (`extensions/llm/`)
- **Purpose**: Domain-specific chat/LLM nodes
- **Key Classes**:
  - `UserMessage` - User chat message
  - `AssistantMessage` - LLM response
  - `SystemMessage` - System prompt
  - `LLMNodeType` - Enum for types
- **Design**: Completely separate from core (optional import)

---

## Node Types at a Glance

| Category | Nodes | Purpose |
|----------|-------|---------|
| **Planning** | PlanNode, PlanStep, RouterStep | Workflow structure and routing decisions |
| **Execution** | ToolCall, TaskRun | Tool invocation and results |
| **Session** | SessionNode, SummaryNode | Execution context and checkpoints |
| **Workflow** | ApprovalNode | Human-in-the-loop approval gates |
| **Artifacts** | ArtifactNode | Lineage tracking for multi-artifact workflows |
| **Jobs** | JobNode, JobRunNode | High-level task tracking |
| **LLM (ext)** | UserMessage, AssistantMessage, SystemMessage | Chat workflow support |

---

## Edge Types at a Glance

| Category | Edges | Purpose |
|----------|-------|---------|
| **Hierarchy** | ParentChildEdge | Containment relationships |
| **Planning** | PlanLinkEdge, StepEdge | Plan structure and dependencies |
| **Routing** | RouteEdge | Routing paths |
| **Ordering** | NextEdge, CustomEdge | Sequential and custom relationships |
| **Workflow** | ApprovalEdge, FallbackEdge, ArtifactDependencyEdge | Complex workflow patterns |

---

## Pydantic Features Used

### 1. Literal Types (Discriminated Unions)
```python
class PlanNode(GraphNode):
    kind: Literal[NodeType.PLAN] = NodeType.PLAN  # Type-safe!
```
Not just a string - the type checker knows the exact value.

### 2. Field Validators
```python
@field_validator("routes")
def validate_routes(cls, v: list[str]) -> list[str]:
    if len(v) < 2:
        raise ValueError("Router must have at least 2 routes")
    return v
```

### 3. Immutability
```python
class GraphNode(BaseModel):
    model_config = ConfigDict(frozen=True)  # Can't mutate after creation
```

### 4. Computed Properties
```python
@property
def duration_seconds(self) -> Optional[float]:
    if self.started_at and self.completed_at:
        return (self.completed_at - self.started_at).total_seconds()
    return None
```

### 5. Default Factories
```python
id: str = Field(default_factory=lambda: str(uuid4()))  # Generate on creation
```

---

## Key Design Patterns

### Pattern 1: Fluent Builder
```python
plan = (Plan("My Plan")
    .step("Research")
        .step("Find sources")
        .step("Read papers")
    .up()
    .step("Write"))
```
Clean, ergonomic API for creating hierarchical plans.

### Pattern 2: Lazy Indexing
```python
step.index = "1.2.3"  # Assigned only when needed
```
Steps numbered as "1", "1.1", "1.2.3" reflecting hierarchy.

### Pattern 3: Graph Persistence
```python
# In-memory tree → saved to graph as nodes + edges
_Step tree  →  PlanNode + PlanStep nodes + edges
```

### Pattern 4: Extensibility
- **Core** knows nothing about LLM
- **LLM extension** depends on core
- **User code** imports what it needs
- Future audio/video extensions follow same pattern

---

## Import Paths

### Core Graph Types
```python
from chuk_ai_planner.core.graph import (
    # Enums
    NodeType, EdgeType, RouterType, StepStatus, ApprovalStatus,
    # Base classes
    GraphNode, GraphEdge,
    # Specific nodes
    PlanNode, PlanStep, RouterStep, ToolCall, TaskRun,
    SessionNode, ApprovalNode, ArtifactNode,
    # Specific edges
    ParentChildEdge, PlanLinkEdge, StepEdge, RouteEdge,
)
```

### Planning
```python
from chuk_ai_planner.core.planner import Plan, PlanExecutor
from chuk_ai_planner.core.store import GraphStore, InMemoryGraphStore
```

### Routing
```python
from chuk_ai_planner.core.routing import RoutingExecutor, RoutingDecision
```

### LLM Extension (Optional)
```python
from chuk_ai_planner.extensions.llm import (
    UserMessage, AssistantMessage, SystemMessage, LLMNodeType
)
```

---

## Creating Custom Extensions

Template for adding your own domain (audio, video, etc.):

```python
# extensions/multimedia/types.py
class MultimediaNodeType(str, Enum):
    VIDEO_NODE = "video_node"
    AUDIO_NODE = "audio_node"

# extensions/multimedia/nodes.py
from chuk_ai_planner.core.graph.nodes.base import GraphNode

class VideoNode(GraphNode):
    kind: Literal[MultimediaNodeType.VIDEO_NODE] = MultimediaNodeType.VIDEO_NODE
    duration_seconds: float
    resolution: str
    codec: str

# extensions/multimedia/__init__.py
from .types import MultimediaNodeType
from .nodes import VideoNode, AudioNode

__all__ = ["MultimediaNodeType", "VideoNode", "AudioNode"]
```

Users import it as:
```python
from chuk_ai_planner.extensions.multimedia import VideoNode
```

---

## File Locations Reference

| What | Where |
|------|-------|
| All node types | `/core/graph/nodes/` |
| All edge types | `/core/graph/edges/` |
| All enums | `/core/graph/types.py` |
| Planning DSL | `/core/planner/plan.py` |
| Plan execution | `/core/planner/plan_executor.py` |
| Routing logic | `/core/routing/executor.py` |
| Storage interface | `/core/store/base.py` |
| In-memory store | `/core/store/memory.py` |
| LLM nodes | `/extensions/llm/nodes.py` |

---

## Key Statistics

- **Total Core Lines**: ~3500+ lines of code
- **Pure Pydantic Models**: ~700 lines (fully typed, zero goop)
- **Planning Engine**: ~2100 lines
- **Storage + Routing**: ~300 lines
- **Extensions**: ~120 lines (LLM module)

---

## Architecture Principles

1. **Domain-Agnostic Core** - No assumptions about use case
2. **Modular Extensions** - Domain code in `/extensions/`, not core
3. **Pure Pydantic** - Everything typed, nothing stringly-typed
4. **Async-Native** - GraphStore fully async from the ground up
5. **Type-Safe** - Literal types, validators, computed properties
6. **Immutable** - Models frozen, can't accidentally mutate
7. **Hashable** - Nodes/edges hashable by ID for sets/dicts
8. **Composable** - Core + any extensions, no forced dependencies

---

## Next Steps for Using This

1. **Read** `CODEBASE_STRUCTURE.md` for deep dive on each module
2. **Review** `ARCHITECTURE_OVERVIEW.txt` for visual diagrams
3. **Look at** actual node definitions in `/core/graph/nodes/`
4. **Try** creating a simple Plan with the fluent builder
5. **Extend** by adding new node types in an extension

---

Generated: 2025-11-15
Context: Very thorough exploration of new codebase structure
