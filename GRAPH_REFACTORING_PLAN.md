# Graph Models Refactoring Plan

## Problems Identified

### 1. **Folder Structure is Backwards** ❌
```
models/           ← Contains graph nodes, edges, planning
  ├── base.py
  ├── edges/
  ├── planning.py
  └── ...

graph/            ← Only has node_manager.py!
  └── node_manager.py
```

**Should be:**
```
graph/            ← All graph-related code
  ├── nodes/
  │   ├── base.py
  │   ├── plan.py
  │   ├── execution.py
  │   └── ...
  ├── edges/
  ├── store/
  └── ...
```

### 2. **Dictionary Goop** ❌

**Current (BAD):**
```python
class RouterStep(GraphNode):
    data: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, router_type, routes, description, **kwargs):
        data = {
            "router_type": router_type,  # Hardcoded string!
            "routes": routes,
            "description": description,
            **kwargs
        }
        super().__init__(kind=NodeKind.ROUTER_STEP, data=data)

    @property
    def router_type(self) -> str:
        return self.data.get("router_type", "expression")  # More strings!
```

**Should be (GOOD):**
```python
class RouterStep(GraphNode):
    kind: Literal[NodeKind.ROUTER_STEP] = NodeKind.ROUTER_STEP

    # Typed fields!
    router_type: RouterType  # Enum, not string
    routes: List[str]
    description: str
    condition: Optional[str] = None
    llm_prompt: Optional[str] = None
    route_mapping: Optional[Dict[bool, str]] = None
```

### 3. **Hardcoded Strings Everywhere** ❌

**Current:**
```python
# In code:
step.data.get("description")
step.data.get("index")
step.data.get("router_type")
tool.data.get("name")
tool.data.get("args")

# What if we typo?
step.data.get("descripion")  # Silent failure!
```

**Should be:**
```python
# Type-safe access:
step.description
step.index
router.router_type
tool.name
tool.args

# Typos caught by IDE/mypy:
step.descripion  # AttributeError!
```

### 4. **Not Leveraging Pydantic** ❌

Missing out on:
- Field validation
- Computed fields
- Type coercion
- JSON schema generation
- IDE autocomplete
- Static type checking

---

## Proposed Solution

### Phase 1: Define Enums & Constants

```python
# graph/types.py

from enum import Enum
from typing import Literal

class NodeType(str, Enum):
    """All node types in the graph."""
    SESSION = "session"
    PLAN = "plan"
    PLAN_STEP = "plan_step"
    ROUTER_STEP = "router_step"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALL = "tool_call"
    TASK_RUN = "task_run"
    SUMMARY = "summary"

class EdgeType(str, Enum):
    """All edge types in the graph."""
    PARENT_CHILD = "parent_child"
    NEXT = "next"
    PLAN_LINK = "plan_link"
    STEP_ORDER = "step_order"
    ROUTE = "route"
    CUSTOM = "custom"

class RouterType(str, Enum):
    """Router strategy types."""
    EXPRESSION = "expression"
    LLM = "llm"
    FUNCTION = "function"

class StepStatus(str, Enum):
    """Step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
```

### Phase 2: Pure Pydantic Base Node

```python
# graph/nodes/base.py

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict

from chuk_ai_planner.graph.types import NodeType

class GraphNode(BaseModel):
    """
    Base class for all graph nodes.

    Pure Pydantic - no dictionary goop!
    """
    model_config = ConfigDict(
        frozen=True,  # Immutable
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    # Core fields - every node has these
    id: str = Field(default_factory=lambda: str(uuid4()))
    kind: NodeType  # Enforced by subclasses
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional metadata (for flexibility)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphNode):
            return NotImplemented
        return self.id == other.id
```

### Phase 3: Typed Node Classes

```python
# graph/nodes/plan.py

from typing import Literal, Optional
from pydantic import Field

from .base import GraphNode
from chuk_ai_planner.graph.types import NodeType

class PlanNode(GraphNode):
    """A plan - top-level workflow container."""

    kind: Literal[NodeType.PLAN] = NodeType.PLAN

    # Typed fields
    title: str
    description: Optional[str] = None
    variables: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

class PlanStep(GraphNode):
    """A single step in a plan."""

    kind: Literal[NodeType.PLAN_STEP] = NodeType.PLAN_STEP

    # Typed fields
    description: str
    index: Optional[str] = None  # Hierarchical index like "1.2.3"
    status: StepStatus = StepStatus.PENDING
    result_variable: Optional[str] = None

class RouterStep(GraphNode):
    """A routing decision point."""

    kind: Literal[NodeType.ROUTER_STEP] = NodeType.ROUTER_STEP

    # Typed fields
    router_type: RouterType
    routes: list[str]
    description: str

    # Type-specific optional fields
    condition: Optional[str] = None  # For expression routing
    llm_prompt: Optional[str] = None  # For LLM routing
    router_function: Optional[str] = None  # For function routing
    route_mapping: Optional[dict[bool | str, str]] = None

    # Validation
    @field_validator('routes')
    @classmethod
    def validate_routes(cls, v):
        if len(v) < 2:
            raise ValueError("Router must have at least 2 routes")
        return v
```

```python
# graph/nodes/execution.py

from typing import Literal, Optional, Any
from pydantic import Field

from .base import GraphNode
from chuk_ai_planner.graph.types import NodeType

class ToolCall(GraphNode):
    """A tool invocation."""

    kind: Literal[NodeType.TOOL_CALL] = NodeType.TOOL_CALL

    # Typed fields
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result_variable: Optional[str] = None

class TaskRun(GraphNode):
    """Result of a tool execution."""

    kind: Literal[NodeType.TASK_RUN] = NodeType.TASK_RUN

    # Typed fields
    tool_call_id: str
    status: Literal["success", "failure", "running"]
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
```

### Phase 4: Clean Edges

```python
# graph/edges/base.py

from typing import Optional, Any
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict

from chuk_ai_planner.graph.types import EdgeType

class GraphEdge(BaseModel):
    """Base class for all graph edges."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    kind: EdgeType
    src: str  # Source node ID
    dst: str  # Destination node ID

    # Optional metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
```

```python
# graph/edges/routing.py

from typing import Literal, Optional
from pydantic import Field

from .base import GraphEdge
from chuk_ai_planner.graph.types import EdgeType

class RouteEdge(GraphEdge):
    """Edge from router to target step."""

    kind: Literal[EdgeType.ROUTE] = EdgeType.ROUTE

    # Typed fields
    route_key: str  # Which route this edge represents
    is_default: bool = False
```

### Phase 5: Folder Reorganization

**New structure:**
```
src/chuk_ai_planner/
├── graph/                    ← All graph code
│   ├── __init__.py
│   ├── types.py             ← Enums & constants
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── base.py          ← GraphNode
│   │   ├── plan.py          ← PlanNode, PlanStep, RouterStep
│   │   ├── execution.py     ← ToolCall, TaskRun
│   │   ├── messages.py      ← UserMessage, AssistantMessage
│   │   └── session.py       ← SessionNode
│   ├── edges/
│   │   ├── __init__.py
│   │   ├── base.py          ← GraphEdge
│   │   ├── hierarchy.py     ← ParentChildEdge
│   │   ├── planning.py      ← PlanEdge, StepEdge
│   │   ├── routing.py       ← RouteEdge
│   │   └── ordering.py      ← NextEdge
│   └── store/
│       ├── __init__.py
│       ├── base.py          ← GraphStore interface
│       └── memory.py        ← InMemoryGraphStore
├── planner/
├── routing/
├── agents/
└── ...
```

---

## Migration Strategy

### Step 1: Create New Structure (Parallel)
- Create `graph/` folder with new code
- Keep old `models/` folder temporarily
- Both can coexist during migration

### Step 2: Update Imports Gradually
```python
# Old
from chuk_ai_planner.models.planning import PlanStep
from chuk_ai_planner.models.base import NodeKind

# New
from chuk_ai_planner.graph.nodes import PlanStep
from chuk_ai_planner.graph.types import NodeType
```

### Step 3: Update Code Patterns

**Old pattern:**
```python
# Creating a node
step = PlanStep(data={
    "description": "Do something",
    "index": "1",
    "status": "pending"
})

# Accessing fields
desc = step.data.get("description")
status = step.data.get("status", "pending")
```

**New pattern:**
```python
# Creating a node
step = PlanStep(
    description="Do something",
    index="1",
    status=StepStatus.PENDING
)

# Accessing fields
desc = step.description  # Type-safe!
status = step.status  # Returns StepStatus enum
```

### Step 4: Update Tests
- Rewrite tests to use typed access
- Verify all functionality works
- Remove old code

### Step 5: Remove Old Structure
- Delete `models/` folder
- Update all documentation
- Final cleanup

---

## Benefits

### 1. **Type Safety** ✅
```python
# IDE knows the type!
step: PlanStep = ...
step.description  # ← Autocomplete works!
step.descripton   # ← IDE shows error before runtime
```

### 2. **No More String Keys** ✅
```python
# Old: Easy to typo
data.get("descripion")  # Silent bug!

# New: Caught immediately
node.descripion  # AttributeError!
```

### 3. **Validation** ✅
```python
class RouterStep(GraphNode):
    routes: list[str]

    @field_validator('routes')
    @classmethod
    def validate_routes(cls, v):
        if len(v) < 2:
            raise ValueError("Need at least 2 routes")
        return v

# Automatically validated!
router = RouterStep(routes=["only_one"])  # ← Raises ValidationError
```

### 4. **Better Documentation** ✅
```python
class PlanStep(GraphNode):
    """A single step in a plan."""

    description: str  # IDE shows this in tooltips
    index: Optional[str]  # Clear that it's optional
    status: StepStatus  # Shows enum options
```

### 5. **JSON Schema** ✅
```python
# Get schema for free
schema = PlanStep.model_json_schema()

# Use for:
# - API documentation
# - Validation
# - Code generation
# - OpenAPI specs
```

---

## Example: Before & After

### Before (Dictionary Goop)

```python
# Creating a router
router = RouterStep(
    router_type="expression",
    condition="${score} > 0.7",
    routes=["high", "low"],
    description="Quality check"
)

# Accessing fields - lots of .get()
rtype = router.data.get("router_type")  # String, could be None
cond = router.data.get("condition", "")  # Default handling
routes = router.data.get("routes", [])   # More defaults

# Using it
if rtype == "expression":  # String comparison, typo-prone
    # ...
```

### After (Pure Pydantic)

```python
# Creating a router
router = RouterStep(
    router_type=RouterType.EXPRESSION,
    condition="${score} > 0.7",
    routes=["high", "low"],
    description="Quality check"
)

# Accessing fields - direct, typed
rtype: RouterType = router.router_type  # Enum, never None
cond: Optional[str] = router.condition   # Type-safe
routes: list[str] = router.routes        # Always a list

# Using it
if router.router_type == RouterType.EXPRESSION:  # Type-safe!
    # IDE autocompletes RouterType options
```

---

## Implementation Checklist

- [ ] Create `graph/types.py` with all enums
- [ ] Create `graph/nodes/base.py` with GraphNode
- [ ] Create typed node classes:
  - [ ] `graph/nodes/plan.py` (PlanNode, PlanStep, RouterStep)
  - [ ] `graph/nodes/execution.py` (ToolCall, TaskRun)
  - [ ] `graph/nodes/messages.py` (UserMessage, AssistantMessage)
  - [ ] `graph/nodes/session.py` (SessionNode)
- [ ] Create typed edge classes:
  - [ ] `graph/edges/base.py` (GraphEdge)
  - [ ] `graph/edges/routing.py` (RouteEdge)
  - [ ] `graph/edges/hierarchy.py` (ParentChildEdge)
  - [ ] `graph/edges/planning.py` (PlanEdge, StepEdge)
- [ ] Move `store/` into `graph/store/`
- [ ] Update all imports
- [ ] Update UniversalExecutor to use typed nodes
- [ ] Update RoutingExecutor to use typed nodes
- [ ] Update examples
- [ ] Run tests
- [ ] Remove old `models/` folder
- [ ] Update documentation

---

## Timeline

**Phase 1 (Day 1):** Create new structure in parallel
- Define all enums and types
- Create base classes
- Implement core node types

**Phase 2 (Day 2):** Implement all node/edge types
- All typed nodes
- All typed edges
- Validation logic

**Phase 3 (Day 3):** Update codebase
- Update executors
- Update examples
- Fix imports

**Phase 4 (Day 4):** Test & cleanup
- Run all tests
- Fix issues
- Remove old code
- Update docs

---

## Risks & Mitigation

**Risk:** Breaking existing code
**Mitigation:** Keep both structures during migration, gradual cutover

**Risk:** Lots of code to update
**Mitigation:** Use IDE refactoring tools, update incrementally

**Risk:** Forgetting edge cases
**Mitigation:** Comprehensive tests, careful review

---

## Decision

**Proceed with refactoring?**

This will make the codebase:
- ✅ More maintainable
- ✅ Type-safe
- ✅ Easier to understand
- ✅ Better documented
- ✅ Production-ready

It's the right move before building more features on top!
