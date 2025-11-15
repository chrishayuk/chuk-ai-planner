# Chuk-AI-Planner: New Codebase Structure Analysis

## Executive Summary

The codebase has been refactored into a **modular, extensible architecture** with clear separation between:
- **Core Framework** (`src/chuk_ai_planner/core/`) - Domain-agnostic planning and execution infrastructure
- **Extensions** (`src/chuk_ai_planner/extensions/`) - Domain-specific features (currently LLM-focused)

All code follows a **pure Pydantic model** approach with **no dictionary goop** - everything is explicitly typed and validated.

---

## 1. Core Framework Structure (`src/chuk_ai_planner/core/`)

### Directory Organization

```
src/chuk_ai_planner/core/
├── __init__.py                    # Core module entry point
├── graph/                         # Pure Pydantic graph system
│   ├── __init__.py               # Re-exports all graph types
│   ├── types.py                  # Enums: NodeType, EdgeType, RouterType, StepStatus, etc.
│   ├── node_manager.py           # Node management utilities
│   ├── nodes/                    # Node type definitions
│   │   ├── __init__.py
│   │   ├── base.py              # GraphNode (base class for all nodes)
│   │   ├── plan.py              # PlanNode, PlanStep, RouterStep
│   │   ├── execution.py         # ToolCall, TaskRun
│   │   ├── session.py           # SessionNode, SummaryNode
│   │   ├── workflow.py          # ApprovalNode (human-in-the-loop)
│   │   ├── artifact.py          # ArtifactNode (lineage tracking)
│   │   └── job.py               # JobNode, JobRunNode
│   └── edges/                    # Edge type definitions
│       ├── __init__.py
│       ├── base.py              # GraphEdge (base class for all edges)
│       ├── hierarchy.py         # ParentChildEdge
│       ├── planning.py          # PlanLinkEdge, StepEdge
│       ├── routing.py           # RouteEdge
│       ├── ordering.py          # NextEdge, CustomEdge
│       └── workflow.py          # ApprovalEdge, FallbackEdge, ArtifactDependencyEdge
├── planner/                      # Planning engine and DSL
│   ├── __init__.py              # Re-exports Plan, PlanExecutor
│   ├── plan.py                  # Plan class (author-facing DSL)
│   ├── plan_executor.py         # PlanExecutor (internal helper)
│   ├── plan_registry.py         # Plan storage and retrieval
│   ├── universal_plan.py        # Universal plan representation
│   ├── universal_plan_executor.py  # Universal plan executor
│   ├── _ids.py                  # UUID generation utilities
│   ├── _persist.py              # Graph persistence helpers
│   └── _step_tree.py            # In-memory step tree structure
├── routing/                      # Conditional execution logic
│   ├── __init__.py              # Re-exports RoutingExecutor, RoutingDecision
│   └── executor.py              # Routing evaluation (expression, LLM, function)
└── store/                        # Graph storage abstraction
    ├── __init__.py              # Re-exports GraphStore, InMemoryGraphStore
    ├── base.py                  # GraphStore (abstract base)
    └── memory.py                # InMemoryGraphStore implementation
```

---

## 2. Pydantic Model Architecture

### 2.1 Base Node Model (`GraphNode`)

**Location:** `/src/chuk_ai_planner/core/graph/nodes/base.py`

```python
class GraphNode(BaseModel):
    """Base class for all graph nodes."""
    
    # All nodes are immutable, type-safe, self-documenting
    model_config = ConfigDict(
        frozen=True,  # Immutable
        arbitrary_types_allowed=True,
        use_enum_values=True,  # Serialize enums as values
    )
    
    # Core fields (every node has these)
    id: str                                      # UUID
    kind: str                                    # Subclasses override with Literal types
    ts: datetime                                 # Timestamp
    metadata: dict[str, Any] = Field(default_factory=dict)  # Extensibility
```

**Key Design Principles:**
- **Immutable** (frozen=True) for data integrity
- **Type-safe** with explicit Literal types in subclasses
- **Hashable** by ID for use in sets/dicts
- **Equality** based on ID, not content

### 2.2 Base Edge Model (`GraphEdge`)

**Location:** `/src/chuk_ai_planner/core/graph/edges/base.py`

```python
class GraphEdge(BaseModel):
    """Base class for all graph edges."""
    
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )
    
    # Core fields (every edge has these)
    id: str                                      # UUID
    kind: EdgeType                               # Enum enforced by subclasses
    src: str                                     # Source node ID
    dst: str                                     # Destination node ID
    metadata: dict[str, Any] = Field(default_factory=dict)
```

---

## 3. Core Node Types

### 3.1 Planning Nodes (`nodes/plan.py`)

#### PlanNode
- Top-level container for a workflow
- Fields: title, description, variables, tags, concurrency_level, reliability_profile, version

#### PlanStep
- Single executable step in a plan
- Fields: description, index (hierarchical: "1.2.3"), status (StepStatus enum)
- Error handling: max_retries, retry_delay_seconds, fallback_step_id, timeout_seconds
- Artifacts: input_artifacts, output_artifacts (lineage tracking)
- Cost tracking: max_cost, estimated_cost, actual_cost, estimated_duration, actual_duration

#### RouterStep
- Conditional routing decision point
- router_type: RouterType (EXPRESSION, LLM, FUNCTION)
- Routes: list of available paths
- Type-specific fields:
  - **Expression routing**: condition (e.g., "${score} > 0.7"), route_mapping
  - **LLM routing**: llm_prompt (question for LLM)
  - **Function routing**: router_function (reference to custom function)

### 3.2 Execution Nodes (`nodes/execution.py`)

#### ToolCall
- Invocation request to an MCP tool or external function
- Fields: name, args (dict), result_variable

#### TaskRun
- Result of executing a ToolCall
- Fields: tool_call_id, status (TaskStatus), result, error, timing info
- Tracks attempts, cost, tokens_used, model_used
- Property: duration_seconds (calculated from start/end times)

### 3.3 Session & Summary Nodes (`nodes/session.py`)

#### SessionNode
- Top-level execution context container
- Fields: name, description, user_id, context (dict), status, started_at, completed_at

#### SummaryNode
- Checkpoint/summary during execution
- Fields: title, content, summary_type (SummaryType enum), execution_state

### 3.4 Workflow Nodes (`nodes/workflow.py`)

#### ApprovalNode
- Human-in-the-loop approval gate
- Fields: approval_type, prompt, status (ApprovalStatus)
- Approval tracking: approved_by, approved_at, rejection_reason
- Behavior: timeout_seconds, auto_approve_after, escalate_to

### 3.5 Artifact Nodes (`nodes/artifact.py`)

#### ArtifactNode
- References and lineage tracking for artifacts (video, script, image, etc.)
- Fields: artifact_id, artifact_type, storage_path, presigned_url
- Lineage: produced_by_step, consumed_by_steps
- Metadata: size_bytes, mime_type, checksum

### 3.6 Job Nodes (`nodes/job.py`)

#### JobNode
- High-level task representation
- Fields: description, status, metadata, tags, current_run_id
- Statistics: run_count, successful_runs, failed_runs
- Timestamps: created_at, updated_at

#### JobRunNode
- Single execution attempt of a job
- References: job_id, plan_id, session_id
- Status tracking: status, error, error_details
- Execution stats: steps_completed, steps_total, steps_failed, steps_skipped

---

## 4. Core Edge Types

### 4.1 Hierarchy Edges (`edges/hierarchy.py`)

#### ParentChildEdge
- Parent-child/containment relationship
- Example: Plan contains Steps, Session contains Plans
- Pure structural relationship (src is parent, dst is child)

### 4.2 Planning Edges (`edges/planning.py`)

#### PlanLinkEdge
- Links plan to its components (steps, routers, etc.)
- Direct association edge

#### StepEdge
- Ordering/dependency between steps
- Fields: dependency (bool), condition (optional)
- SemanticS: dst depends on src completing

### 4.3 Routing Edges (`edges/routing.py`)

#### RouteEdge
- One path from a router to a target step
- Fields: route_key (matches RouterStep.routes), is_default (bool)
- Router evaluates condition and chooses which RouteEdge to follow

### 4.4 Ordering Edges (`edges/ordering.py`)

#### NextEdge
- Temporal/sequential ordering between nodes
- Fields: weight (optional, for priority)
- Used for chat message ordering (user -> assistant -> user)

#### CustomEdge
- Extensibility for domain-specific edge types
- Fields: custom_type, properties (dict)

### 4.5 Workflow Edges (`edges/workflow.py`)

#### ApprovalEdge
- Connects ApprovalNode to conditional flow
- Fields: approval_node_id, on_approved, on_rejected, on_timeout
- Routes differently based on approval outcome

#### FallbackEdge
- Error recovery path
- Fields: trigger_on (list of conditions: "error", "timeout"), priority
- max_cost_exceeded (trigger if budget exceeded)

#### ArtifactDependencyEdge
- Artifact flow between steps
- Fields: artifact_id, artifact_type, required (bool)
- Tracks which steps produce/consume artifacts

---

## 5. Type System & Enums (`graph/types.py`)

All core enums are in a single, centralized file for easy reference:

```python
class NodeType(str, Enum):
    """Core domain-agnostic node types"""
    SESSION = "session"
    PLAN = "plan"
    PLAN_STEP = "plan_step"
    ROUTER_STEP = "router_step"
    TOOL_CALL = "tool_call"
    TASK_RUN = "task_run"
    SUMMARY = "summary"
    APPROVAL = "approval"
    ARTIFACT = "artifact"

class EdgeType(str, Enum):
    """All edge relationship types"""
    PARENT_CHILD = "parent_child"
    NEXT = "next"
    PLAN_LINK = "plan_link"
    STEP_ORDER = "step_order"
    ROUTE = "route"
    CUSTOM = "custom"
    APPROVAL = "approval"
    FALLBACK = "fallback"
    ARTIFACT_DEPENDENCY = "artifact_dependency"

class RouterType(str, Enum):
    """Router evaluation strategies"""
    EXPRESSION = "expression"  # Evaluate boolean expression
    LLM = "llm"                # Ask LLM to choose
    FUNCTION = "function"      # Execute custom function

class StepStatus(str, Enum):
    """Plan step lifecycle"""
    PENDING, RUNNING, COMPLETED, FAILED, SKIPPED, BLOCKED,
    PAUSED, WAITING_APPROVAL, CANCELLED, TIMEOUT, RETRYING

class ApprovalStatus(str, Enum):
    """Human approval gate states"""
    PENDING, APPROVED, REJECTED, TIMEOUT, ESCALATED

class ReliabilityProfile(str, Enum):
    """Plan safety/resilience levels"""
    AGGRESSIVE, BALANCED, ULTRA_SAFE

class TaskStatus(str, Enum):
    """Tool execution result"""
    SUCCESS, FAILURE, RUNNING

class SummaryType(str, Enum):
    """Checkpoint/summary reasons"""
    CHECKPOINT, COMPLETION, ERROR, MILESTONE
```

---

## 6. Storage Abstraction (`store/`)

### 6.1 GraphStore Interface (`store/base.py`)

**Abstract base class** - fully async-native:

```python
class GraphStore(ABC):
    """Abstract base for graph storage implementations"""
    
    # Core operations
    async def add_node(self, node: GraphNode) -> None
    async def get_node(self, node_id: str) -> Optional[GraphNode]
    async def update_node(self, node: GraphNode) -> None
    
    async def add_edge(self, edge: GraphEdge) -> None
    async def get_edges(self, src: Optional[str] = None,
                       dst: Optional[str] = None,
                       kind: Optional[EdgeType] = None) -> List[GraphEdge]
    
    # Helper methods
    async def get_nodes_by_kind(self, kind: NodeType) -> List[GraphNode]
    async def list_nodes(self, kind: Optional[str] = None) -> List[GraphNode]
    async def get_edges_by_src(self, src: str, kind: Optional[EdgeType] = None)
```

### 6.2 InMemoryGraphStore (`store/memory.py`)

Simple in-memory implementation for testing/prototyping:
- Stores nodes in `Dict[str, GraphNode]`
- Stores edges in `List[GraphEdge]`
- Fully async-native
- Supports querying by src, dst, kind

---

## 7. Planning Engine (`planner/`)

### 7.1 Plan (Author-Facing DSL) (`planner/plan.py`)

```python
class Plan:
    """Mutable hierarchy of plan steps - author-facing"""
    
    # Builder pattern
    def step(self, title: str, *, after: Sequence[str] = ()) -> "Plan"
    def up(self) -> "Plan"
    
    # Runtime addition
    async def add_step(self, title: str, *, parent: str | None = None,
                      after: Sequence[str] = ()) -> str
```

**Key concepts:**
- Fluent builder pattern for ergonomic authoring
- Hierarchical step structure (nested children)
- Lazy indexing: steps numbered as "1", "1.1", "1.2.3" on demand
- Supports adding steps at runtime after plan is saved

### 7.2 Internal Components

#### _step_tree.py
- `_Step` class: in-memory tree node for plan authoring
- `assign_indices()`: lazy hierarchical numbering
- `iter_steps()`: DFS iteration

#### _ids.py
- `new_plan_id()`: UUID generation for plans

#### _persist.py
- `persist_full_plan()`: save entire plan to GraphStore
- `persist_single_step()`: save single step (for runtime additions)
- Converts _Step tree to PlanNode/PlanStep/Edge graphs

### 7.3 PlanExecutor (`planner/plan_executor.py`)

**Internal helper** - NOT author-facing:

```python
class PlanExecutor:
    # Get all steps under a plan (DFS over PARENT_CHILD edges)
    async def get_plan_steps(plan_id: str) -> List[PlanStep]
    
    # Topological sort for parallel execution
    async def determine_execution_order(steps: List[PlanStep]) 
        -> List[List[PlanStep]]  # Batches
    
    # Execute a single step
    async def execute_step(step: PlanStep, context: Dict) 
        -> List[ToolResult]
```

### 7.4 Plan Registry (`planner/plan_registry.py`)

Storage and retrieval of plans by name/ID/tags

### 7.5 Universal Plan (`planner/universal_plan.py`)

Generic plan representation with:
- Flat structure (easier for some use cases)
- JSON serialization
- Conversion utilities

### 7.6 Universal Plan Executor (`planner/universal_plan_executor.py`)

Executor for universal plans with:
- Step execution
- Tool invocation
- Result tracking
- Error handling

---

## 8. Routing System (`routing/`)

### RoutingExecutor (`routing/executor.py`)

**Handles conditional routing during plan execution**

```python
class RoutingExecutor:
    """Three types of routing evaluation"""
    
    async def evaluate_route(self, router_step: RouterStep,
                            context: Dict[str, Any]) 
        -> RoutingDecision
```

Supports:
1. **Expression-based**: Evaluate `${variable}` expressions
2. **LLM-based**: Ask LLM to choose route
3. **Function-based**: Execute custom function

Result is a `RoutingDecision` dataclass:
```python
@dataclass
class RoutingDecision:
    route_key: str              # Chosen route
    router_step_id: str
    target_step_id: str
    skipped_routes: List[str]
    evaluation_method: str      # "expression", "llm", "function"
    evaluation_details: Optional[Dict]
```

---

## 9. Extensions Framework (`src/chuk_ai_planner/extensions/`)

### 9.1 LLM Extension (`extensions/llm/`)

**Domain-specific extension for chat/LLM workflows**

```
extensions/llm/
├── __init__.py              # Re-exports types and nodes
├── types.py                 # LLMNodeType enum
└── nodes.py                 # UserMessage, AssistantMessage, SystemMessage
```

### 9.2 LLM Types (`extensions/llm/types.py`)

```python
class LLMNodeType(str, Enum):
    """Extension types for chat workflows (separate from core)"""
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    SYSTEM_MESSAGE = "system_message"
```

**Design Philosophy**: Keep extension types separate from core to maintain modularity.

### 9.3 LLM Node Models (`extensions/llm/nodes.py`)

#### UserMessage
```python
class UserMessage(GraphNode):
    kind: Literal[LLMNodeType.USER_MESSAGE] = LLMNodeType.USER_MESSAGE
    content: str
    role: str = "user"
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
```

#### AssistantMessage
```python
class AssistantMessage(GraphNode):
    kind: Literal[LLMNodeType.ASSISTANT_MESSAGE] = LLMNodeType.ASSISTANT_MESSAGE
    content: str
    role: str = "assistant"
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    model: Optional[str] = None
    finish_reason: Optional[str] = None
```

#### SystemMessage
```python
class SystemMessage(GraphNode):
    kind: Literal[LLMNodeType.SYSTEM_MESSAGE] = LLMNodeType.SYSTEM_MESSAGE
    content: str
    role: str = "system"
```

### 9.4 Extension Architecture

**Key principles:**
- **Modular**: Extensions are imported on-demand, not re-exported by core
- **Lightweight**: Core doesn't depend on extensions
- **Extensible**: Projects can create their own extensions (audio, video, etc.)
- **Composable**: Can use core + LLM + custom extensions together

---

## 10. Type Safety & Pydantic Features

### 10.1 Literal Types for Discriminated Unions

Every node and edge subclass uses Literal types for kind:

```python
class PlanNode(GraphNode):
    kind: Literal[NodeType.PLAN] = NodeType.PLAN  # Not a string, typed enum!
    # ... rest of fields

class UserMessage(GraphNode):
    kind: Literal[LLMNodeType.USER_MESSAGE] = LLMNodeType.USER_MESSAGE
    # ... rest of fields
```

**Benefits:**
- Type checker knows exact kind at compile time
- Discriminated union behavior for isinstance checks
- Serialization/deserialization is type-safe

### 10.2 Field Validators

```python
class RouterStep(GraphNode):
    routes: list[str]
    
    @field_validator("routes")
    @classmethod
    def validate_routes(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("Router must have at least 2 routes")
        return v
```

### 10.3 Computed Properties

```python
class TaskRun(GraphNode):
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds()
        return None
```

### 10.4 Immutability & Hashing

```python
class GraphNode(BaseModel):
    model_config = ConfigDict(frozen=True)  # Immutable!
    
    def __hash__(self) -> int:
        return hash(self.id)  # Hashable for sets/dicts
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphNode):
            return NotImplemented
        return self.id == other.id  # Equality by ID
```

---

## 11. Key Design Patterns

### 11.1 No Dictionary Goop

**Before (bad):**
```python
step = {
    "id": "123",
    "title": "Do something",
    "metadata": {...}  # String keys, lose type safety
}
```

**After (good):**
```python
step = PlanStep(
    description="Do something",
    index="1.1"
)  # All fields typed, validated, self-documenting
```

### 11.2 Discriminated Unions

All node/edge hierarchies use Pydantic's discriminated union pattern:
- Base class with generic `kind: str`
- Subclasses with `kind: Literal[SpecificType]`
- Type checker understands the hierarchy
- Serialization handles polymorphism automatically

### 11.3 Lazy Initialization

```python
class GraphNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))  # Generate on creation
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

### 11.4 Fluent Builder Pattern

```python
plan = (Plan("My Plan")
    .step("Research")
        .step("Find sources")
        .step("Read papers")
    .up()
    .step("Write")
        .step("Draft outline")
    .up())
```

### 11.5 Separation of Concerns

- **Core** (`core/`): Infrastructure, no dependencies on domain
- **Extensions** (`extensions/`): Domain-specific, depends on core
- **User code**: Depends on core + extensions as needed

---

## 12. File Statistics

### Core Module Line Counts

```
Graph System:
  nodes/base.py              72 lines
  nodes/plan.py             167 lines
  nodes/execution.py         90 lines
  nodes/session.py           71 lines
  nodes/workflow.py          54 lines
  nodes/artifact.py          58 lines
  nodes/job.py               68 lines
  
Edges:
  edges/base.py              71 lines
  edges/planning.py          58 lines
  edges/routing.py           47 lines
  edges/workflow.py          98 lines
  edges/ordering.py          63 lines
  edges/hierarchy.py         39 lines

Planner:
  planner/plan.py           139 lines
  planner/plan_executor.py  205 lines
  planner/plan_registry.py  278 lines
  planner/universal_plan.py 466 lines
  planner/universal_plan_executor.py 769 lines

Storage:
  store/base.py             152 lines
  store/memory.py            78 lines

Routing:
  routing/executor.py       ~150+ lines (partially read)

Extensions:
  llm/types.py               28 lines
  llm/nodes.py               94 lines
```

---

## 13. Extension Pattern Example

To create a new extension (e.g., for video/audio):

```python
# chuk_ai_planner/extensions/multimedia/types.py
class MultimediaNodeType(str, Enum):
    VIDEO_NODE = "video_node"
    AUDIO_NODE = "audio_node"

# chuk_ai_planner/extensions/multimedia/nodes.py
class VideoNode(GraphNode):
    kind: Literal[MultimediaNodeType.VIDEO_NODE] = MultimediaNodeType.VIDEO_NODE
    duration_seconds: float
    resolution: str
    codec: str
    # ... domain-specific fields

# chuk_ai_planner/extensions/multimedia/__init__.py
from .types import MultimediaNodeType
from .nodes import VideoNode, AudioNode

__all__ = ["MultimediaNodeType", "VideoNode", "AudioNode"]

# Usage in user code:
from chuk_ai_planner.core.graph import GraphNode
from chuk_ai_planner.extensions.multimedia import VideoNode
```

---

## 14. Import Organization

### Core Graph
```python
from chuk_ai_planner.core.graph import (
    # Types
    NodeType, EdgeType, RouterType, StepStatus,
    # Nodes
    GraphNode, PlanNode, PlanStep, RouterStep,
    ToolCall, TaskRun, SessionNode, ApprovalNode,
    # Edges
    GraphEdge, RouteEdge, StepEdge, ParentChildEdge,
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

### LLM Extension (optional)
```python
from chuk_ai_planner.extensions.llm import (
    UserMessage, AssistantMessage, SystemMessage
)
```

---

## 15. Summary: Key Takeaways

### Strengths of New Architecture

1. **Pure Pydantic**: No dictionary goop - all fields typed and validated
2. **Type Safety**: Literal types, field validators, computed properties
3. **Immutability**: Frozen models prevent accidental mutations
4. **Extensibility**: Domain-specific extensions don't pollute core
5. **Clear Separation**: Core is domain-agnostic; extensions are optional
6. **Async-Native**: GraphStore interface is fully async
7. **Flexible Routing**: Supports expression, LLM, and function-based routing
8. **Artifact Tracking**: Built-in lineage tracking for multi-artifact workflows
9. **Approval Gates**: Human-in-the-loop workflow support
10. **Cost/Performance Tracking**: Built-in fields for monitoring

### Architectural Highlights

- **Graph-based**: Everything is nodes + edges, DAG-structured
- **Hierarchical**: Steps nested under plans, with dotted indices
- **Stateful**: Status tracking throughout execution lifecycle
- **Fault-tolerant**: Retries, fallbacks, approval gates, timeouts
- **Modular**: Can use core alone or mix with extensions
- **Pattern-rich**: Fluent builders, lazy initialization, discriminated unions

### For Future Extensions

The pattern is clear:
1. Create `extensions/{domain}/types.py` with domain-specific enums
2. Create `extensions/{domain}/nodes.py` extending GraphNode
3. Create `extensions/{domain}/edges.py` extending GraphEdge (if needed)
4. Create `extensions/{domain}/__init__.py` re-exporting public API
5. Keep it separate from core - users import what they need

