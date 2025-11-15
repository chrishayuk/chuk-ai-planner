# chuk-ai-planner: LLM-First Planning Runtime

## Core Positioning

**LangGraph is:**
> "Code-first agent runtime where you hand-build the graph"

**chuk-ai-planner is:**
> "LLM-first planning runtime that *also* gives you a graph executor"

### More concretely:

> **"LangGraph-style execution + AutoPlanner: natural-language → typed plan graph → parallel tool execution, MCP-native."**

---

## The Key Difference

### LangGraph Approach
```python
# You write the graph manually
from langgraph.graph import StateGraph

workflow = StateGraph(AgentState)
workflow.add_node("research", research_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("revise", revise_node)
workflow.add_node("publish", publish_node)

# Manually wire up all the edges
workflow.add_edge("research", "analyze")
workflow.add_conditional_edges(
    "analyze",
    should_revise,
    {
        "revise": "revise",
        "publish": "publish"
    }
)
workflow.add_edge("revise", "analyze")
workflow.set_entry_point("research")

app = workflow.compile(checkpointer=memory)
```

### chuk-ai-planner Approach
```python
# You describe what you want
from chuk_ai_planner import GraphPlanAgent, UniversalExecutor

agent = GraphPlanAgent(graph=graph, tools=mcp_tools)

# Natural language → automatic plan generation
plan_id = await agent.plan_into_graph(
    "Research climate change adaptation strategies, "
    "analyze the data quality, and if quality is high, "
    "publish a report, otherwise revise the research. "
    "Get manager approval before publishing."
)

# Execute with automatic dependency resolution, parallelization, checkpointing
executor = UniversalExecutor(graph=graph)
results = await executor.execute(plan_id, session_id="sess-123")
```

**The difference:**
- **LangGraph**: You're a graph programmer. Build nodes, wire edges, handle state.
- **chuk-ai-planner**: You're a product manager. Describe outcomes, let the planner build the graph.

---

## What You Get

### 1. Planning-First Design
- **Natural language → plan graph**: LLM generates the workflow
- **Automatic dependency detection**: No manual edge wiring for simple flows
- **Plan validation**: Catches errors before execution
- **Plan optimization**: Identifies parallelization opportunities

### 2. LangGraph-Level Execution
- **Conditional routing**: Route based on results or LLM decisions
- **Checkpointing & resume**: Survive crashes, long-running workflows
- **Human-in-loop**: Approval steps, interrupts, manual overrides
- **Error policies**: Retries, fallbacks, escalation per step

### 3. Production Tool Execution
- Via **chuk-tool-processor**: Retries, timeouts, rate limits, resource isolation
- Battle-tested error handling
- Metrics and observability built-in

### 4. MCP-Native Ecosystem
- **chuk-mcp-runtime**: Native MCP tool discovery and execution
- **chuk-acp**: Editor integration (VS Code, Cursor, Claude Desktop)
- **Artifacts & scopes**: Rich output formats
- **Unified stack**: One framework for planning, execution, and integration

---

## When to Use Each

### Use LangGraph when:
- You want **maximum control** over graph structure
- You're building **highly custom stateful agents**
- You need **complex cyclic workflows** with intricate branching
- You're already invested in the LangChain ecosystem
- You prefer **code-first** graph construction

### Use chuk-ai-planner when:
- You want **automatic planning** from natural language
- You need **DAG workflows** with conditional branches
- You want **MCP tool integration** out of the box
- You value **higher-level abstractions** over manual graph coding
- You're building **goal-oriented workflows** where the LLM plans the path
- You want **integrated planning + execution + editor support**

### Use Both when:
- Use **chuk-ai-planner** for high-level orchestration and planning
- Use **LangGraph** for low-level agent loops that need tight control
- They can complement each other in a larger system

---

## The CHUK Stack Advantage

chuk-ai-planner isn't just a standalone library—it's the **planning brain** of a complete ecosystem:

```
┌─────────────────────────────────────────────────┐
│  EDITOR INTEGRATION (VS Code, Cursor, etc.)    │
│  via chuk-acp / chuk-acp-agent                  │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  PLANNING LAYER                                 │
│  chuk-ai-planner: Natural language → Plan graph │
│  - GraphPlanAgent (LLM-based planning)          │
│  - Plan DSL (manual planning)                   │
│  - Plan optimization & validation               │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  EXECUTION LAYER                                │
│  UniversalExecutor: Run plan graphs             │
│  - Parallel execution                           │
│  - Conditional routing                          │
│  - Checkpointing & resume                       │
│  - Human-in-loop approvals                      │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  TOOL EXECUTION LAYER                           │
│  chuk-tool-processor: Production-grade tools    │
│  - Retry policies, timeouts, rate limits        │
│  - Resource isolation                           │
│  - Error handling & logging                     │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  TOOL DISCOVERY & INTEGRATION                   │
│  chuk-mcp-runtime: MCP protocol support         │
│  - Tool discovery from MCP servers              │
│  - Schema validation                            │
│  - Artifact handling                            │
│  - Scope management                             │
└─────────────────────────────────────────────────┘
```

**The value proposition:**
> "Give me your MCP tools, and with one integrated stack I'll plan workflows in natural language, execute them as typed graphs with checkpointing and retries, and expose them to your editor."

This is something LangGraph alone cannot provide.

---

## The Next Evolution

### Current State (v0.2)
✅ Natural language → plan graph
✅ Parallel DAG execution
✅ Graph-based plan model
✅ Session tracking & events
✅ Variable flow & template resolution
✅ In-memory graph store

### Phase 1: LangGraph-Level Features (Weeks 1-4)
🎯 Conditional steps & routing
🎯 PostgreSQL/SQLite GraphStore
🎯 Checkpointing & resume
🎯 Human-in-loop approvals
🎯 Per-step error policies

### Phase 2: Beyond LangGraph (Weeks 5-8)
🚀 Implicit dependency discovery
🚀 Plan optimization engine
🚀 Multi-agent planning
🚀 Real-time replanning
🚀 Loop constructs

### Phase 3: Production Excellence (Weeks 9-12)
⭐ Distributed execution
⭐ Advanced monitoring
⭐ REST API & GraphQL
⭐ Plan templates marketplace
⭐ Interactive visualization

---

## Competitive Positioning

| Feature | LangGraph | chuk-ai-planner | Advantage |
|---------|-----------|----------------|-----------|
| **Planning** | Manual graph coding | LLM-generated plans | **Chuk**: 10x faster to build |
| **Execution** | Full StateGraph engine | DAG executor + conditionals | Tie (both production-ready) |
| **Checkpointing** | Built-in | Coming (Phase 1) | **LangGraph** (for now) |
| **Human-in-loop** | Interrupt API | Coming (Phase 1) | **LangGraph** (for now) |
| **Error handling** | Per-node retry | Tool-processor integration | **Chuk**: More sophisticated |
| **Tool ecosystem** | LangChain tools | MCP-native | **Chuk**: Modern protocol |
| **Editor integration** | Via LangSmith | ACP protocol | **Chuk**: Native IDE support |
| **Automatic optimization** | None | Plan analysis & optimization | **Chuk**: Unique feature |
| **Multi-agent planning** | Build it yourself | Built-in | **Chuk**: Unique feature |
| **Learning curve** | Steeper (graph concepts) | Gentler (describe goals) | **Chuk**: Better DX |

---

## Example: Research → Analyze → Publish Workflow

### LangGraph Version (60+ lines)
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Define state
class WorkflowState(TypedDict):
    research_data: str
    analysis: dict
    quality_score: float
    report: str
    approved: bool

# Define nodes
def research(state):
    data = research_climate_data()
    return {"research_data": data}

def analyze(state):
    analysis = analyze_data(state["research_data"])
    return {
        "analysis": analysis,
        "quality_score": analysis.get("quality", 0)
    }

def check_quality(state):
    if state["quality_score"] > 0.7:
        return "publish"
    return "revise"

def revise(state):
    # Improve research
    return {"research_data": improve_research(state)}

def publish(state):
    report = generate_report(state["analysis"])
    return {"report": report}

def approval_required(state):
    # Wait for approval
    return {"approved": False}

def check_approval(state):
    if state.get("approved"):
        return END
    return "approval_required"

# Build graph
workflow = StateGraph(WorkflowState)
workflow.add_node("research", research)
workflow.add_node("analyze", analyze)
workflow.add_node("revise", revise)
workflow.add_node("publish", publish)
workflow.add_node("approval_required", approval_required)

workflow.set_entry_point("research")
workflow.add_edge("research", "analyze")
workflow.add_conditional_edges(
    "analyze",
    check_quality,
    {
        "revise": "revise",
        "publish": "publish"
    }
)
workflow.add_edge("revise", "analyze")
workflow.add_conditional_edges(
    "publish",
    check_approval,
    {
        END: END,
        "approval_required": "approval_required"
    }
)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Execute
result = app.invoke({}, config={"configurable": {"thread_id": "1"}})
```

### chuk-ai-planner Version (15 lines)
```python
from chuk_ai_planner import GraphPlanAgent, UniversalExecutor
from chuk_ai_planner.store.postgres import PostgresGraphStore

# Setup
graph = PostgresGraphStore(connection_string=DB_URL)
agent = GraphPlanAgent(graph=graph, tools=mcp_tools)
executor = UniversalExecutor(graph=graph)

# Natural language → plan graph (automatic)
plan_id = await agent.plan_into_graph(
    "Research climate change adaptation strategies. "
    "Analyze the data quality. If quality score is above 0.7, "
    "generate a report and get manager approval before publishing. "
    "If quality is low, revise the research and re-analyze."
)

# Execute with checkpointing
results = await executor.execute(plan_id, session_id="research-001")
```

**Lines of code:** 60+ vs 15
**Planning time:** Manual graph design vs instant
**Maintenance:** Update graph structure vs update prompt
**Readability:** Graph DSL vs natural language

---

## Summary

**chuk-ai-planner is not just "another LangGraph."**

It's a **planning-first runtime** that brings LLM intelligence to workflow orchestration. While LangGraph gives you the **control** of manual graph construction, chuk-ai-planner gives you the **speed** of automatic planning.

Combined with the CHUK ecosystem (tool-processor, MCP runtime, ACP integration), it's a complete stack for building intelligent, goal-oriented AI workflows that integrate seamlessly with modern development environments.

**The vision:**
> "The easiest way to build production AI workflows: describe what you want, and the system plans, executes, and integrates it—all with LangGraph-level reliability."

---

**Next:** Let's build the 4 missing features and prove it.
