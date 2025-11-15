# Implementation Plan: 4 Critical Features

This document details the implementation of the 4 features needed to achieve "LangGraph-level" capabilities:

1. **Conditional Steps & Router Support**
2. **Database-backed GraphStore with Checkpointing**
3. **Human-in-Loop Approvals**
4. **Per-Step Error Policies**

---

## Feature 1: Conditional Steps & Router Support

### Goal
Support branching logic where execution path depends on runtime values or LLM decisions.

### Current State
- Plans are strict DAGs with explicit dependencies
- No conditional execution paths
- All steps with dependencies met are executed

### Target State
```python
# Expression-based routing
plan.step("Analyze data")
    .router(
        condition="${quality_score} > 0.7",
        routes={
            True: "high_quality_path",
            False: "low_quality_path"
        }
    )

# LLM-based routing
plan.step("Review content")
    .llm_router(
        prompt="Should we publish (publish) or revise (revise)?",
        routes=["publish", "revise"]
    )
```

### Implementation Details

#### 1.1 Data Model Changes

**New Edge Type: `ROUTE_EDGE`**

Location: `src/chuk_ai_planner/models/edges.py`

```python
class RouteEdge(GraphEdge):
    """Edge representing a conditional route from a router step."""

    def __init__(
        self,
        src: str,
        dst: str,
        route_key: str,  # "true", "false", "publish", "revise", etc.
        condition: Optional[str] = None,  # "${quality_score} > 0.7"
        **kwargs
    ):
        super().__init__(
            src=src,
            dst=dst,
            kind=EdgeKind.ROUTE,
            data={
                "route_key": route_key,
                "condition": condition,
                **kwargs
            }
        )
```

**New Node Type: `ROUTER_STEP`**

Location: `src/chuk_ai_planner/models/nodes.py`

```python
class RouterStep(GraphNode):
    """A step that routes to different paths based on conditions."""

    def __init__(
        self,
        router_type: Literal["expression", "llm", "function"],
        routes: List[str],  # Possible route keys
        **kwargs
    ):
        super().__init__(
            kind=NodeKind.ROUTER_STEP,
            data={
                "router_type": router_type,
                "routes": routes,
                **kwargs
            }
        )
```

#### 1.2 Plan DSL Extension

Location: `src/chuk_ai_planner/planner/plan.py`

```python
class Plan:
    # ... existing methods ...

    def router(
        self,
        condition: Optional[str] = None,
        routes: Optional[Dict[Any, str]] = None,
        llm_prompt: Optional[str] = None,
        router_function: Optional[Callable] = None
    ) -> "Plan":
        """
        Add a router step that conditionally routes to different paths.

        Examples:
            # Expression-based
            plan.router(
                condition="${score} > 0.7",
                routes={True: "high", False: "low"}
            )

            # LLM-based
            plan.router(
                llm_prompt="Should we publish or revise?",
                routes={"publish": "publish_flow", "revise": "revise_flow"}
            )

            # Function-based
            plan.router(
                router_function=lambda ctx: "path_a" if ctx["x"] > 10 else "path_b",
                routes={"path_a": "flow_a", "path_b": "flow_b"}
            )
        """
        # Create router step
        router_step = _RouterStep(
            description=f"Route based on: {condition or llm_prompt or 'function'}",
            condition=condition,
            llm_prompt=llm_prompt,
            router_function=router_function,
            routes=routes
        )

        # Add to step tree
        self._current_step._children.append(router_step)

        # For each route, create a target step marker
        for route_key, route_name in routes.items():
            # Mark where this route leads
            route_step = _RouteTarget(name=route_name, route_key=route_key)
            router_step._route_targets[route_key] = route_step

        return self
```

#### 1.3 Execution Logic

Location: `src/chuk_ai_planner/executor/routing_executor.py` (new file)

```python
class RoutingExecutor:
    """Handles conditional routing during plan execution."""

    async def evaluate_route(
        self,
        router_step: GraphNode,
        context: Dict[str, Any],
        graph: GraphStore
    ) -> str:
        """
        Evaluate a router step and return the chosen route key.

        Returns:
            route_key: The key of the selected route
        """
        router_type = router_step.data.get("router_type")

        if router_type == "expression":
            return await self._evaluate_expression(router_step, context)
        elif router_type == "llm":
            return await self._evaluate_llm(router_step, context)
        elif router_type == "function":
            return await self._evaluate_function(router_step, context)
        else:
            raise ValueError(f"Unknown router type: {router_type}")

    async def _evaluate_expression(
        self,
        router_step: GraphNode,
        context: Dict[str, Any]
    ) -> str:
        """Evaluate a boolean expression against context."""
        from chuk_ai_planner.executor.variable_resolver import resolve_variables

        condition = router_step.data["condition"]

        # Resolve variables in condition
        resolved = resolve_variables(condition, context)

        # Safely evaluate (use ast.literal_eval or safe eval library)
        # For MVP, support simple comparisons
        result = self._safe_eval(resolved, context)

        # Map True/False to route keys
        routes = router_step.data["routes"]
        return routes.get(result, routes.get("default"))

    async def _evaluate_llm(
        self,
        router_step: GraphNode,
        context: Dict[str, Any]
    ) -> str:
        """Use LLM to decide the route."""
        from chuk_ai_planner.agents.base_agent import BaseAgent

        prompt = router_step.data["llm_prompt"]
        routes = router_step.data["routes"]

        # Create structured prompt
        system_prompt = f"""You are a routing decision maker.
Given the context, choose one of these routes: {', '.join(routes)}

Respond with ONLY the route key, nothing else.
"""

        user_prompt = f"""Context:
{json.dumps(context, indent=2)}

Question: {prompt}

Available routes: {', '.join(routes)}

Your choice:"""

        # Call LLM
        agent = BaseAgent()  # Use configured LLM
        response = await agent.call_llm(system_prompt, user_prompt)

        # Extract route key from response
        chosen_route = response.strip().lower()

        # Validate it's a valid route
        if chosen_route not in routes:
            # Try fuzzy matching or default
            chosen_route = routes[0] if routes else "default"

        return chosen_route

    async def _evaluate_function(
        self,
        router_step: GraphNode,
        context: Dict[str, Any]
    ) -> str:
        """Execute a Python function to determine route."""
        router_function = router_step.data["router_function"]

        # Execute function with context
        result = router_function(context)

        return result

    def _safe_eval(self, expression: str, context: Dict[str, Any]) -> Any:
        """Safely evaluate expressions."""
        # For MVP, support basic comparisons
        # Use a library like simpleeval for safety
        # Or implement a simple parser for >, <, ==, etc.

        import ast
        import operator

        ops = {
            ast.Gt: operator.gt,
            ast.Lt: operator.lt,
            ast.GtE: operator.ge,
            ast.LtE: operator.le,
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
        }

        # Parse and evaluate safely
        # This is simplified - use simpleeval in production
        try:
            tree = ast.parse(expression, mode='eval')
            # ... safe evaluation logic ...
            return eval(compile(tree, '', 'eval'), {"__builtins__": {}}, context)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}': {e}")
```

#### 1.4 Integration with UniversalExecutor

Location: `src/chuk_ai_planner/executor/universal_executor.py`

```python
class UniversalExecutor:
    # ... existing code ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.routing_executor = RoutingExecutor()

    async def _execute_step(
        self,
        step_id: str,
        context: Dict[str, Any]
    ) -> Any:
        """Execute a single step, handling routing if needed."""
        step_node = self.graph.get_node(step_id)

        # Check if this is a router step
        if step_node.kind == NodeKind.ROUTER_STEP:
            # Evaluate the router
            chosen_route = await self.routing_executor.evaluate_route(
                step_node,
                context,
                self.graph
            )

            # Find the outgoing ROUTE_EDGE for this route key
            edges = self.graph.get_edges_by_src(step_id)
            route_edges = [e for e in edges if e.kind == EdgeKind.ROUTE]

            chosen_edge = next(
                (e for e in route_edges if e.data["route_key"] == chosen_route),
                None
            )

            if not chosen_edge:
                raise ValueError(f"No route edge found for key: {chosen_route}")

            # Mark other routes as skipped
            for edge in route_edges:
                if edge != chosen_edge:
                    self._mark_route_skipped(edge.dst, context)

            # Store routing decision
            context["routing_decisions"] = context.get("routing_decisions", {})
            context["routing_decisions"][step_id] = chosen_route

            # Return the next step to execute
            return {"route_chosen": chosen_route, "next_step": chosen_edge.dst}

        else:
            # Regular step execution
            return await super()._execute_step(step_id, context)

    def _mark_route_skipped(self, step_id: str, context: Dict[str, Any]):
        """Mark a route and all its descendants as skipped."""
        skipped = context.setdefault("skipped_steps", set())
        skipped.add(step_id)

        # Recursively skip descendants
        children = self._get_all_descendants(step_id)
        skipped.update(children)
```

#### 1.5 Tests

Location: `tests/test_routing.py` (new file)

```python
import pytest
from chuk_ai_planner import Plan
from chuk_ai_planner.executor.universal_executor import UniversalExecutor
from chuk_ai_planner.store.memory import InMemoryGraphStore


@pytest.mark.asyncio
async def test_expression_routing():
    """Test expression-based routing."""
    graph = InMemoryGraphStore()

    plan = (
        Plan("Quality Check", graph=graph)
            .step("Analyze data").up()
            .router(
                condition="${quality_score} > 0.7",
                routes={
                    True: "high_quality",
                    False: "low_quality"
                }
            )
            .step("High quality path", route_target="high_quality").up()
            .step("Low quality path", route_target="low_quality").up()
    )

    plan_id = plan.save()

    # Execute with high quality score
    executor = UniversalExecutor(graph=graph)
    context = {"quality_score": 0.85}
    results = await executor.execute(plan_id, context=context)

    # Verify high quality path was taken
    assert "high_quality" in context["routing_decisions"]
    assert "Low quality path" not in [r["step"] for r in results]


@pytest.mark.asyncio
async def test_llm_routing():
    """Test LLM-based routing."""
    graph = InMemoryGraphStore()

    plan = (
        Plan("Content Review", graph=graph)
            .step("Generate content").up()
            .router(
                llm_prompt="Is the content ready to publish or does it need revision?",
                routes=["publish", "revise"]
            )
            .step("Publish content", route_target="publish").up()
            .step("Revise content", route_target="revise").up()
    )

    plan_id = plan.save()

    # Execute (LLM will decide)
    executor = UniversalExecutor(graph=graph)
    results = await executor.execute(plan_id)

    # Verify one path was chosen
    assert len(context["routing_decisions"]) > 0
```

### Migration Path
1. Add new edge and node types to models
2. Extend Plan DSL with `.router()` method
3. Create RoutingExecutor
4. Integrate with UniversalExecutor
5. Add tests
6. Update examples

### Success Criteria
- ✅ Expression-based routing works
- ✅ LLM-based routing works
- ✅ Only one route is taken per router
- ✅ Skipped routes don't execute
- ✅ Routing decisions are tracked
- ✅ Works with existing DAG features

---

## Feature 2: Database-Backed GraphStore with Checkpointing

### Goal
Persist execution state to survive crashes and enable resume functionality.

### Current State
- Only `InMemoryGraphStore` implemented
- No persistence beyond process lifetime
- No resume capability

### Target State
```python
from chuk_ai_planner.store.postgres import PostgresGraphStore

# Connect to database
graph = PostgresGraphStore(
    connection_string="postgresql://localhost/planner"
)

# Execute plan with checkpointing
executor = UniversalExecutor(graph=graph)
results = await executor.execute(plan_id, session_id="sess-123")

# Later, resume after crash
results = await executor.resume(plan_id, session_id="sess-123")
```

### Implementation Details

#### 2.1 Database Schema

Location: `schema/postgres/001_initial.sql`

```sql
-- Nodes table
CREATE TABLE nodes (
    id UUID PRIMARY KEY,
    kind VARCHAR(50) NOT NULL,
    ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_nodes_kind ON nodes(kind);
CREATE INDEX idx_nodes_ts ON nodes(ts);
CREATE INDEX idx_nodes_data_gin ON nodes USING GIN(data);

-- Edges table
CREATE TABLE edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kind VARCHAR(50) NOT NULL,
    src UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    dst UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_edges_kind ON edges(kind);
CREATE INDEX idx_edges_src ON edges(src);
CREATE INDEX idx_edges_dst ON edges(dst);
CREATE INDEX idx_edges_src_kind ON edges(src, kind);

-- Execution checkpoints
CREATE TABLE execution_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    plan_id UUID NOT NULL REFERENCES nodes(id),
    step_id UUID REFERENCES nodes(id),
    status VARCHAR(50) NOT NULL, -- 'pending', 'running', 'completed', 'failed'
    context JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_checkpoints_session ON execution_checkpoints(session_id);
CREATE INDEX idx_checkpoints_plan ON execution_checkpoints(plan_id);
CREATE INDEX idx_checkpoints_status ON execution_checkpoints(status);
CREATE UNIQUE INDEX idx_checkpoints_session_step ON execution_checkpoints(session_id, step_id);

-- Step execution status
CREATE TABLE step_status (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    step_id UUID NOT NULL REFERENCES nodes(id),
    status VARCHAR(50) NOT NULL,
    result JSONB,
    error TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_step_status_session ON step_status(session_id);
CREATE INDEX idx_step_status_step ON step_status(step_id);
CREATE UNIQUE INDEX idx_step_status_session_step ON step_status(session_id, step_id);
```

#### 2.2 PostgresGraphStore Implementation

Location: `src/chuk_ai_planner/store/postgres.py`

```python
import asyncpg
import json
from typing import List, Optional, Dict, Any
from uuid import UUID

from chuk_ai_planner.store.base import GraphStore
from chuk_ai_planner.models import GraphNode, GraphEdge


class PostgresGraphStore(GraphStore):
    """PostgreSQL-backed graph store with persistence."""

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        max_pool_size: int = 20
    ):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.max_pool_size = max_pool_size
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Initialize connection pool."""
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=self.pool_size,
                max_size=self.max_pool_size
            )

    async def close(self):
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def add_node(self, node: GraphNode) -> str:
        """Add a node to the graph."""
        await self.connect()

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO nodes (id, kind, ts, data)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (id) DO UPDATE
                SET kind = $2, ts = $3, data = $4, updated_at = NOW()
                """,
                UUID(node.id),
                node.kind.value,
                node.ts,
                json.dumps(dict(node.data))
            )

        return node.id

    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by ID."""
        await self.connect()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, kind, ts, data FROM nodes WHERE id = $1",
                UUID(node_id)
            )

        if not row:
            return None

        return GraphNode(
            id=str(row['id']),
            kind=row['kind'],
            ts=row['ts'],
            data=row['data']
        )

    async def add_edge(self, edge: GraphEdge) -> str:
        """Add an edge to the graph."""
        await self.connect()

        async with self._pool.acquire() as conn:
            edge_id = await conn.fetchval(
                """
                INSERT INTO edges (kind, src, dst, data)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                edge.kind.value,
                UUID(edge.src),
                UUID(edge.dst),
                json.dumps(edge.data)
            )

        return str(edge_id)

    async def get_edges_by_src(
        self,
        src: str,
        kind: Optional[str] = None
    ) -> List[GraphEdge]:
        """Get all edges from a source node."""
        await self.connect()

        async with self._pool.acquire() as conn:
            if kind:
                rows = await conn.fetch(
                    """
                    SELECT id, kind, src, dst, data
                    FROM edges
                    WHERE src = $1 AND kind = $2
                    """,
                    UUID(src), kind
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, kind, src, dst, data
                    FROM edges
                    WHERE src = $1
                    """,
                    UUID(src)
                )

        return [
            GraphEdge(
                kind=row['kind'],
                src=str(row['src']),
                dst=str(row['dst']),
                data=row['data']
            )
            for row in rows
        ]

    # ... implement other GraphStore methods ...

    # Checkpointing methods

    async def save_checkpoint(
        self,
        session_id: str,
        plan_id: str,
        step_id: Optional[str],
        status: str,
        context: Dict[str, Any],
        error: Optional[str] = None
    ):
        """Save execution checkpoint."""
        await self.connect()

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO execution_checkpoints
                (session_id, plan_id, step_id, status, context, error)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (session_id, step_id)
                DO UPDATE SET
                    status = $4,
                    context = $5,
                    error = $6,
                    updated_at = NOW()
                """,
                session_id,
                UUID(plan_id),
                UUID(step_id) if step_id else None,
                status,
                json.dumps(context),
                error
            )

    async def load_checkpoint(
        self,
        session_id: str,
        plan_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint for a session."""
        await self.connect()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT step_id, status, context, error
                FROM execution_checkpoints
                WHERE session_id = $1 AND plan_id = $2
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                session_id,
                UUID(plan_id)
            )

        if not row:
            return None

        return {
            "step_id": str(row['step_id']) if row['step_id'] else None,
            "status": row['status'],
            "context": row['context'],
            "error": row['error']
        }

    async def save_step_status(
        self,
        session_id: str,
        step_id: str,
        status: str,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        started_at: Optional[Any] = None,
        completed_at: Optional[Any] = None
    ):
        """Save step execution status."""
        await self.connect()

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO step_status
                (session_id, step_id, status, result, error, started_at, completed_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (session_id, step_id)
                DO UPDATE SET
                    status = $3,
                    result = $4,
                    error = $5,
                    started_at = COALESCE($6, step_status.started_at),
                    completed_at = $7
                """,
                session_id,
                UUID(step_id),
                status,
                json.dumps(result) if result is not None else None,
                error,
                started_at,
                completed_at
            )

    async def get_completed_steps(
        self,
        session_id: str
    ) -> List[str]:
        """Get all completed step IDs for a session."""
        await self.connect()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT step_id FROM step_status
                WHERE session_id = $1 AND status = 'completed'
                """,
                session_id
            )

        return [str(row['step_id']) for row in rows]
```

#### 2.3 Resume Capability

Location: `src/chuk_ai_planner/executor/universal_executor.py`

```python
class UniversalExecutor:
    # ... existing code ...

    async def resume(
        self,
        plan_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Resume execution from last checkpoint."""

        # Load checkpoint
        checkpoint = await self.graph.load_checkpoint(session_id, plan_id)

        if not checkpoint:
            raise ValueError(f"No checkpoint found for session {session_id}")

        # Restore context
        context = checkpoint["context"]

        # Get completed steps
        completed_steps = await self.graph.get_completed_steps(session_id)
        context["executed_steps"] = set(completed_steps)

        # Resume from last step
        if checkpoint["status"] == "failed":
            # Retry failed step or skip it based on error policy
            last_step_id = checkpoint["step_id"]
            # ... handle retry logic ...

        # Continue execution
        return await self.execute(
            plan_id,
            session_id=session_id,
            context=context,
            resume=True
        )

    async def execute(
        self,
        plan_id: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        resume: bool = False
    ) -> Dict[str, Any]:
        """Execute plan with checkpointing."""

        # ... existing execution logic ...

        # Save checkpoint after each step
        if session_id and hasattr(self.graph, 'save_checkpoint'):
            await self.graph.save_checkpoint(
                session_id=session_id,
                plan_id=plan_id,
                step_id=current_step_id,
                status="running",
                context=context
            )

        # ... continue execution ...

        # Save step status
        if session_id and hasattr(self.graph, 'save_step_status'):
            await self.graph.save_step_status(
                session_id=session_id,
                step_id=step_id,
                status="completed",
                result=result,
                started_at=start_time,
                completed_at=end_time
            )
```

### Migration Path
1. Create database schema
2. Implement PostgresGraphStore
3. Add checkpointing to executor
4. Implement resume logic
5. Add migration tools
6. Update documentation

### Success Criteria
- ✅ All nodes/edges persist to database
- ✅ Execution survives process restart
- ✅ Resume continues from last checkpoint
- ✅ No data loss on crash
- ✅ Performance acceptable (<100ms per checkpoint)

---

## Feature 3: Human-in-Loop Approvals

### Goal
Allow plans to pause for human approval before critical steps.

### Current State
- No built-in approval mechanism
- Plans execute end-to-end without interruption

### Target State
```python
plan = Plan("Deploy Application")
    .step("Run tests").up()
    .step("Build artifacts").up()
    .approval("Manager approval required before deployment")
    .step("Deploy to production").up()

# Execute - will pause at approval
executor = UniversalExecutor(graph=graph)
result = await executor.execute(plan_id, session_id="deploy-001")

# Check status
status = await executor.get_status(session_id="deploy-001")
# Returns: {"status": "awaiting_approval", "approval_id": "..."}

# Approve
await executor.approve(approval_id, approved=True, comment="LGTM")

# Resume
result = await executor.resume(plan_id, session_id="deploy-001")
```

### Implementation Details

#### 3.1 Data Model

**New Node Type: `APPROVAL_STEP`**

Location: `src/chuk_ai_planner/models/nodes.py`

```python
class ApprovalStep(GraphNode):
    """A step that requires human approval."""

    def __init__(
        self,
        description: str,
        approvers: Optional[List[str]] = None,  # User IDs who can approve
        timeout_seconds: Optional[int] = None,
        auto_approve_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            kind=NodeKind.APPROVAL_STEP,
            data={
                "description": description,
                "approvers": approvers or [],
                "timeout_seconds": timeout_seconds,
                "auto_approve_after": auto_approve_after,
                **kwargs
            }
        )
```

**New Event Type:**

```python
class EventType(str, Enum):
    # ... existing events ...
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_TIMEOUT = "approval_timeout"
```

#### 3.2 Database Schema

Location: `schema/postgres/002_approvals.sql`

```sql
CREATE TABLE approvals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    step_id UUID NOT NULL REFERENCES nodes(id),
    status VARCHAR(50) NOT NULL, -- 'pending', 'approved', 'denied', 'timeout'
    description TEXT NOT NULL,
    approvers TEXT[], -- Array of user IDs
    approved_by VARCHAR(255),
    approval_comment TEXT,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    responded_at TIMESTAMPTZ,
    timeout_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_approvals_session ON approvals(session_id);
CREATE INDEX idx_approvals_status ON approvals(status);
CREATE INDEX idx_approvals_timeout ON approvals(timeout_at) WHERE status = 'pending';
```

#### 3.3 Plan DSL Extension

Location: `src/chuk_ai_planner/planner/plan.py`

```python
class Plan:
    # ... existing methods ...

    def approval(
        self,
        description: str,
        approvers: Optional[List[str]] = None,
        timeout_seconds: Optional[int] = None
    ) -> "Plan":
        """
        Add an approval step that pauses execution.

        Args:
            description: What needs approval
            approvers: List of user IDs who can approve (None = any user)
            timeout_seconds: Auto-deny after this many seconds
        """
        approval_step = _ApprovalStep(
            description=description,
            approvers=approvers,
            timeout_seconds=timeout_seconds
        )

        self._current_step._children.append(approval_step)
        return self
```

#### 3.4 Execution Logic

Location: `src/chuk_ai_planner/executor/approval_executor.py` (new file)

```python
class ApprovalExecutor:
    """Handles human-in-loop approvals."""

    def __init__(self, graph: GraphStore):
        self.graph = graph

    async def request_approval(
        self,
        session_id: str,
        step_id: str,
        step_node: GraphNode
    ) -> str:
        """
        Request approval and return approval ID.
        Raises an ApprovalRequired exception to pause execution.
        """
        description = step_node.data["description"]
        approvers = step_node.data.get("approvers", [])
        timeout_seconds = step_node.data.get("timeout_seconds")

        # Calculate timeout
        timeout_at = None
        if timeout_seconds:
            from datetime import datetime, timedelta
            timeout_at = datetime.utcnow() + timedelta(seconds=timeout_seconds)

        # Create approval record
        approval_id = await self._create_approval(
            session_id=session_id,
            step_id=step_id,
            description=description,
            approvers=approvers,
            timeout_at=timeout_at
        )

        # Emit event
        await self._emit_approval_event(
            session_id=session_id,
            event_type="approval_required",
            approval_id=approval_id,
            description=description
        )

        # Pause execution
        raise ApprovalRequired(
            approval_id=approval_id,
            description=description,
            approvers=approvers
        )

    async def check_approval(
        self,
        approval_id: str
    ) -> Dict[str, Any]:
        """Check status of an approval request."""
        async with self.graph._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, status, approved_by, approval_comment,
                       requested_at, responded_at
                FROM approvals
                WHERE id = $1
                """,
                UUID(approval_id)
            )

        if not row:
            raise ValueError(f"Approval {approval_id} not found")

        return {
            "approval_id": str(row['id']),
            "status": row['status'],
            "approved_by": row['approved_by'],
            "comment": row['approval_comment'],
            "requested_at": row['requested_at'],
            "responded_at": row['responded_at']
        }

    async def approve(
        self,
        approval_id: str,
        approved: bool,
        user_id: Optional[str] = None,
        comment: Optional[str] = None
    ):
        """Approve or deny an approval request."""
        # Verify approval exists and is pending
        approval = await self.check_approval(approval_id)

        if approval["status"] != "pending":
            raise ValueError(f"Approval {approval_id} is not pending")

        # Update approval
        status = "approved" if approved else "denied"

        async with self.graph._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE approvals
                SET status = $1,
                    approved_by = $2,
                    approval_comment = $3,
                    responded_at = NOW()
                WHERE id = $4
                """,
                status,
                user_id,
                comment,
                UUID(approval_id)
            )

        # Emit event
        event_type = "approval_granted" if approved else "approval_denied"
        # ... emit event ...

    async def _create_approval(self, **kwargs) -> str:
        """Create approval record in database."""
        async with self.graph._pool.acquire() as conn:
            approval_id = await conn.fetchval(
                """
                INSERT INTO approvals
                (session_id, step_id, status, description, approvers, timeout_at)
                VALUES ($1, $2, 'pending', $3, $4, $5)
                RETURNING id
                """,
                kwargs['session_id'],
                UUID(kwargs['step_id']),
                kwargs['description'],
                kwargs.get('approvers', []),
                kwargs.get('timeout_at')
            )

        return str(approval_id)


class ApprovalRequired(Exception):
    """Exception raised when approval is needed."""

    def __init__(self, approval_id: str, description: str, approvers: List[str]):
        self.approval_id = approval_id
        self.description = description
        self.approvers = approvers
        super().__init__(f"Approval required: {description}")
```

#### 3.5 Integration with UniversalExecutor

Location: `src/chuk_ai_planner/executor/universal_executor.py`

```python
class UniversalExecutor:
    # ... existing code ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.approval_executor = ApprovalExecutor(self.graph)

    async def _execute_step(
        self,
        step_id: str,
        context: Dict[str, Any]
    ) -> Any:
        """Execute a single step, handling approvals if needed."""
        step_node = self.graph.get_node(step_id)

        # Check if this is an approval step
        if step_node.kind == NodeKind.APPROVAL_STEP:
            # Request approval (this will raise ApprovalRequired)
            await self.approval_executor.request_approval(
                session_id=context.get("session_id"),
                step_id=step_id,
                step_node=step_node
            )

        # Regular step execution
        return await super()._execute_step(step_id, context)

    async def execute(self, *args, **kwargs):
        """Execute with approval handling."""
        try:
            return await super().execute(*args, **kwargs)
        except ApprovalRequired as e:
            # Execution paused for approval
            return {
                "status": "awaiting_approval",
                "approval_id": e.approval_id,
                "description": e.description,
                "approvers": e.approvers
            }

    async def approve(
        self,
        approval_id: str,
        approved: bool,
        user_id: Optional[str] = None,
        comment: Optional[str] = None
    ):
        """Approve or deny a pending approval."""
        await self.approval_executor.approve(
            approval_id=approval_id,
            approved=approved,
            user_id=user_id,
            comment=comment
        )
```

### Migration Path
1. Add ApprovalStep node type
2. Create approvals table
3. Implement ApprovalExecutor
4. Add `.approval()` to Plan DSL
5. Integrate with UniversalExecutor
6. Add approval UI/CLI commands
7. Add tests

### Success Criteria
- ✅ Plans pause at approval steps
- ✅ Approvals can be granted/denied
- ✅ Execution resumes after approval
- ✅ Timeouts work correctly
- ✅ Multiple approvers supported
- ✅ Events emitted for all approval actions

---

## Feature 4: Per-Step Error Policies

### Goal
Configure retry, fallback, and escalation behavior per step.

### Current State
- Basic error handling
- No automatic retries
- No fallback strategies
- All errors fatal

### Target State
```python
plan = Plan("Resilient Workflow")
    .step("Fetch data", error_policy={
        "max_retries": 3,
        "retry_delay": 5,
        "retry_backoff": "exponential",
        "on_final_failure": "continue"  # or "fail", "fallback"
    })
    .step("Fallback data source", is_fallback_for="1")
    .step("Process data", error_policy={
        "max_retries": 1,
        "on_final_failure": "fail"
    })
```

### Implementation Details

#### 4.1 Error Policy Schema

Location: `src/chuk_ai_planner/executor/error_policy.py` (new file)

```python
from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel


class BackoffStrategy(str, Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    CONSTANT = "constant"


class OnFailureAction(str, Enum):
    FAIL = "fail"  # Stop execution, mark plan as failed
    CONTINUE = "continue"  # Log error, continue with next steps
    FALLBACK = "fallback"  # Execute fallback step
    ESCALATE = "escalate"  # Notify humans, await decision


class ErrorPolicy(BaseModel):
    """Error handling policy for a step."""

    max_retries: int = 0
    retry_delay: float = 1.0  # seconds
    retry_backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    retry_on: Optional[List[str]] = None  # Exception class names to retry
    timeout: Optional[float] = None  # Step timeout in seconds
    on_final_failure: OnFailureAction = OnFailureAction.FAIL
    fallback_step_id: Optional[str] = None
    escalation_contacts: Optional[List[str]] = None

    def calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay for given attempt number."""
        if self.retry_backoff == BackoffStrategy.CONSTANT:
            return self.retry_delay
        elif self.retry_backoff == BackoffStrategy.LINEAR:
            return self.retry_delay * attempt
        elif self.retry_backoff == BackoffStrategy.EXPONENTIAL:
            return self.retry_delay * (2 ** (attempt - 1))
        elif self.retry_backoff == BackoffStrategy.FIBONACCI:
            def fib(n):
                if n <= 1:
                    return 1
                return fib(n-1) + fib(n-2)
            return self.retry_delay * fib(attempt)
```

#### 4.2 Plan DSL Extension

Location: `src/chuk_ai_planner/planner/plan.py`

```python
class Plan:
    # ... existing methods ...

    def step(
        self,
        description: str,
        error_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "Plan":
        """
        Add a step with optional error policy.

        Args:
            description: Step description
            error_policy: Error handling configuration
                {
                    "max_retries": 3,
                    "retry_delay": 5,
                    "retry_backoff": "exponential",
                    "on_final_failure": "continue"
                }
        """
        step = _Step(
            description=description,
            error_policy=error_policy,
            **kwargs
        )

        # ... existing step creation logic ...

        return self
```

#### 4.3 Error Executor

Location: `src/chuk_ai_planner/executor/error_executor.py` (new file)

```python
import asyncio
import logging
from typing import Any, Dict, Optional, Callable

from chuk_ai_planner.executor.error_policy import ErrorPolicy, OnFailureAction


class ErrorExecutor:
    """Handles error policies during execution."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def execute_with_policy(
        self,
        step_id: str,
        step_func: Callable,
        error_policy: Optional[ErrorPolicy],
        context: Dict[str, Any]
    ) -> Any:
        """
        Execute a function with error policy.

        Args:
            step_id: Step identifier
            step_func: Async function to execute
            error_policy: Error handling policy
            context: Execution context

        Returns:
            Result of step_func

        Raises:
            Exception if all retries exhausted and policy is FAIL
        """
        if not error_policy:
            # No policy, execute directly
            return await step_func()

        attempt = 0
        last_error = None

        while attempt <= error_policy.max_retries:
            attempt += 1

            try:
                # Execute with timeout if specified
                if error_policy.timeout:
                    result = await asyncio.wait_for(
                        step_func(),
                        timeout=error_policy.timeout
                    )
                else:
                    result = await step_func()

                # Success!
                if attempt > 1:
                    self.logger.info(
                        f"Step {step_id} succeeded on attempt {attempt}"
                    )

                return result

            except Exception as e:
                last_error = e

                # Check if we should retry this exception
                if not self._should_retry(e, error_policy):
                    self.logger.error(
                        f"Step {step_id} failed with non-retryable error: {e}"
                    )
                    break

                # Check if we have retries left
                if attempt > error_policy.max_retries:
                    self.logger.error(
                        f"Step {step_id} failed after {attempt} attempts: {e}"
                    )
                    break

                # Calculate backoff delay
                delay = error_policy.calculate_delay(attempt)

                self.logger.warning(
                    f"Step {step_id} failed (attempt {attempt}), "
                    f"retrying in {delay}s: {e}"
                )

                # Wait before retry
                await asyncio.sleep(delay)

        # All retries exhausted, handle final failure
        return await self._handle_final_failure(
            step_id=step_id,
            error=last_error,
            policy=error_policy,
            context=context
        )

    def _should_retry(
        self,
        error: Exception,
        policy: ErrorPolicy
    ) -> bool:
        """Check if this error should be retried."""
        if not policy.retry_on:
            # Retry all errors
            return True

        # Check if error type is in retry list
        error_type = type(error).__name__
        return error_type in policy.retry_on

    async def _handle_final_failure(
        self,
        step_id: str,
        error: Exception,
        policy: ErrorPolicy,
        context: Dict[str, Any]
    ) -> Any:
        """Handle failure after all retries exhausted."""

        if policy.on_final_failure == OnFailureAction.FAIL:
            # Re-raise the error to stop execution
            raise error

        elif policy.on_final_failure == OnFailureAction.CONTINUE:
            # Log and continue
            self.logger.error(
                f"Step {step_id} failed, continuing execution: {error}"
            )
            return {
                "status": "failed",
                "error": str(error),
                "continued": True
            }

        elif policy.on_final_failure == OnFailureAction.FALLBACK:
            # Execute fallback step
            if not policy.fallback_step_id:
                raise ValueError(
                    f"Step {step_id} has FALLBACK policy but no fallback_step_id"
                )

            self.logger.warning(
                f"Step {step_id} failed, executing fallback: "
                f"{policy.fallback_step_id}"
            )

            # Signal to executor to run fallback
            return {
                "status": "failed",
                "error": str(error),
                "fallback_to": policy.fallback_step_id
            }

        elif policy.on_final_failure == OnFailureAction.ESCALATE:
            # Notify humans and wait for decision
            self.logger.critical(
                f"Step {step_id} failed, escalating to: "
                f"{policy.escalation_contacts}"
            )

            # Create escalation (similar to approval)
            raise EscalationRequired(
                step_id=step_id,
                error=error,
                contacts=policy.escalation_contacts or []
            )


class EscalationRequired(Exception):
    """Exception raised when human escalation is needed."""

    def __init__(self, step_id: str, error: Exception, contacts: List[str]):
        self.step_id = step_id
        self.error = error
        self.contacts = contacts
        super().__init__(f"Escalation required for step {step_id}: {error}")
```

#### 4.4 Integration with UniversalExecutor

Location: `src/chuk_ai_planner/executor/universal_executor.py`

```python
class UniversalExecutor:
    # ... existing code ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_executor = ErrorExecutor()

    async def _execute_step(
        self,
        step_id: str,
        context: Dict[str, Any]
    ) -> Any:
        """Execute a single step with error policy."""
        step_node = self.graph.get_node(step_id)

        # Get error policy from step metadata
        policy_data = step_node.data.get("error_policy")
        error_policy = ErrorPolicy(**policy_data) if policy_data else None

        # Create step execution function
        async def execute():
            # ... existing step execution logic ...
            return result

        # Execute with error policy
        try:
            result = await self.error_executor.execute_with_policy(
                step_id=step_id,
                step_func=execute,
                error_policy=error_policy,
                context=context
            )

            # Handle fallback if needed
            if isinstance(result, dict) and result.get("fallback_to"):
                fallback_id = result["fallback_to"]
                return await self._execute_step(fallback_id, context)

            return result

        except EscalationRequired as e:
            # Handle escalation (similar to approval flow)
            # ... create escalation record, pause execution ...
            raise
```

#### 4.5 Integration with chuk-tool-processor

Location: `src/chuk_ai_planner/executor/tool_executor_adapter.py` (new file)

```python
from chuk_tool_processor import ToolProcessor, RetryPolicy

class ToolExecutorAdapter:
    """Adapter to use chuk-tool-processor with error policies."""

    @staticmethod
    def convert_to_tool_processor_policy(
        error_policy: ErrorPolicy
    ) -> RetryPolicy:
        """Convert ErrorPolicy to ToolProcessor's RetryPolicy."""
        from chuk_tool_processor import RetryPolicy, BackoffStrategy

        # Map backoff strategies
        backoff_map = {
            "linear": BackoffStrategy.LINEAR,
            "exponential": BackoffStrategy.EXPONENTIAL,
            "constant": BackoffStrategy.CONSTANT
        }

        return RetryPolicy(
            max_attempts=error_policy.max_retries + 1,
            initial_delay=error_policy.retry_delay,
            backoff_strategy=backoff_map.get(
                error_policy.retry_backoff.value,
                BackoffStrategy.EXPONENTIAL
            ),
            timeout=error_policy.timeout
        )
```

### Migration Path
1. Create ErrorPolicy model
2. Implement ErrorExecutor
3. Add error_policy parameter to Plan.step()
4. Integrate with UniversalExecutor
5. Add integration with chuk-tool-processor
6. Add tests for all failure scenarios
7. Add examples

### Success Criteria
- ✅ Retries work with all backoff strategies
- ✅ Fallback steps execute on failure
- ✅ Continue mode logs errors but proceeds
- ✅ Escalation pauses execution
- ✅ Integration with chuk-tool-processor
- ✅ Comprehensive error tracking

---

## Implementation Timeline

### Week 1: Conditional Steps & Routing
- Days 1-2: Data model + DSL extension
- Days 3-4: RoutingExecutor implementation
- Day 5: Integration + testing

### Week 2: Database Store & Checkpointing
- Days 1-2: Database schema + PostgresGraphStore
- Days 3-4: Checkpointing + resume logic
- Day 5: Migration tools + testing

### Week 3: Human-in-Loop Approvals
- Days 1-2: Data model + database schema
- Days 3-4: ApprovalExecutor + integration
- Day 5: Testing + examples

### Week 4: Error Policies
- Days 1-2: ErrorPolicy model + ErrorExecutor
- Days 3-4: Integration with executor + tool-processor
- Day 5: Comprehensive testing

### Week 5: Polish & Documentation
- Examples showing all 4 features together
- LangGraph comparison document
- Update README
- Create migration guide

---

## Success Metrics

After implementing these 4 features:

### Technical Parity with LangGraph
- ✅ Conditional routing (expressions + LLM)
- ✅ Persistence & checkpointing
- ✅ Human-in-loop interrupts
- ✅ Sophisticated error handling

### Beyond LangGraph
- ✅ LLM-generated plans (unique!)
- ✅ Automatic dependency detection (coming)
- ✅ Plan optimization (coming)
- ✅ MCP-native integration
- ✅ Production tool execution via chuk-tool-processor

### Quality Bars
- ✅ 90%+ test coverage
- ✅ <100ms checkpoint overhead
- ✅ Zero data loss on crash/restart
- ✅ Comprehensive examples
- ✅ Production-ready documentation

---

## Next Actions

1. **Get approval on approach** - Review this plan
2. **Set up project** - Create feature branches
3. **Start with routing** - Highest value, foundational
4. **Parallel workstreams** - Can work on DB + approvals concurrently
5. **Integration week** - Week 5 to tie everything together
6. **Launch** - Ship v0.3 with full LangGraph parity + planning superpowers

Ready to start implementation? Let's begin with Feature 1: Conditional Steps & Router Support!
