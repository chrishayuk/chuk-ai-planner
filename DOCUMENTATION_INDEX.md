# Chuk-AI-Planner Documentation Index

This directory contains comprehensive documentation of the refactored codebase structure (v2).

## Documents

### 1. QUICK_REFERENCE.md (START HERE)
**What**: Quick overview and module breakdown
**Best for**: Getting oriented quickly, understanding the architecture at 10,000 feet
**Time**: 5-10 minutes
**Contents**:
- What changed (before/after structure)
- Module breakdown with key classes
- Node and edge types at a glance
- Pydantic features used
- Import paths for common tasks
- Extension template

### 2. CODEBASE_STRUCTURE.md (DEEP DIVE)
**What**: Comprehensive analysis of every module
**Best for**: Understanding implementation details, seeing code examples
**Time**: 20-30 minutes
**Contents**:
- Complete directory tree structure
- Pydantic model architecture (base classes)
- Detailed node type descriptions (all 11 types)
- Detailed edge type descriptions (all 9 types)
- Type system & enums reference
- Storage abstraction documentation
- Planning engine architecture
- Routing system details
- LLM extension walkthrough
- Type safety patterns
- Design patterns explanation
- File statistics

### 3. ARCHITECTURE_OVERVIEW.txt (VISUAL REFERENCE)
**What**: ASCII diagrams and flowcharts
**Best for**: Visual learners, understanding relationships
**Time**: 10-15 minutes
**Contents**:
- High-level architecture diagram
- Planning system hierarchy
- Pydantic model hierarchy
- Execution flow example
- File organization checklist

---

## Learning Path

### Path 1: "Just Show Me The Code"
1. Read: QUICK_REFERENCE.md (5 min)
2. Look at: `/src/chuk_ai_planner/core/graph/nodes/base.py`
3. Look at: `/src/chuk_ai_planner/core/graph/nodes/plan.py`
4. Look at: `/src/chuk_ai_planner/core/planner/plan.py`

### Path 2: "Full Understanding"
1. Read: QUICK_REFERENCE.md (5 min)
2. Read: ARCHITECTURE_OVERVIEW.txt (15 min)
3. Read: CODEBASE_STRUCTURE.md (30 min)
4. Read actual source files for areas of interest

### Path 3: "I Need to Extend This"
1. Read: QUICK_REFERENCE.md section "Creating Custom Extensions"
2. Read: CODEBASE_STRUCTURE.md section "9. Extensions Framework"
3. Look at: `/src/chuk_ai_planner/extensions/llm/` as a template
4. Create your own extension following the pattern

---

## Key Concepts Quick Links

### Graph System
- **Base Classes**: See CODEBASE_STRUCTURE.md section 2.1-2.2
- **All Node Types**: See CODEBASE_STRUCTURE.md section 3
- **All Edge Types**: See CODEBASE_STRUCTURE.md section 4
- **Type Enums**: See CODEBASE_STRUCTURE.md section 5

### Planning
- **Author DSL**: See CODEBASE_STRUCTURE.md section 7.1
- **Internal Execution**: See CODEBASE_STRUCTURE.md section 7.3
- **Hierarchical Structure**: See ARCHITECTURE_OVERVIEW.txt "Planning System Hierarchy"

### Extensibility
- **Extension Pattern**: See QUICK_REFERENCE.md section "Creating Custom Extensions"
- **LLM Example**: See CODEBASE_STRUCTURE.md section 9
- **Core Principle**: See QUICK_REFERENCE.md principle #1

---

## Module Structure (Quick Ref)

```
src/chuk_ai_planner/
├── core/                          # Domain-agnostic infrastructure
│   ├── graph/                     # Pure Pydantic nodes + edges
│   │   ├── types.py              # All enums
│   │   ├── nodes/                # 11 node types
│   │   ├── edges/                # 9 edge types
│   │   └── node_manager.py       # Utilities
│   ├── planner/                  # Planning DSL + execution
│   │   ├── plan.py              # Author API
│   │   ├── plan_executor.py     # Internal helper
│   │   ├── plan_registry.py     # Storage
│   │   ├── universal_plan*.py   # Generic variants
│   │   └── _*.py                # Internal helpers
│   ├── routing/                  # Conditional routing
│   │   └── executor.py          # Route evaluation
│   └── store/                    # Storage abstraction
│       ├── base.py              # GraphStore interface
│       └── memory.py            # In-memory implementation
└── extensions/                   # Domain-specific (optional)
    └── llm/                      # Chat/LLM nodes
        ├── types.py             # LLMNodeType enum
        └── nodes.py             # 3 message types
```

---

## Node Type Reference

| Type | Location | Purpose |
|------|----------|---------|
| PlanNode | nodes/plan.py | Workflow container |
| PlanStep | nodes/plan.py | Executable step |
| RouterStep | nodes/plan.py | Conditional routing |
| ToolCall | nodes/execution.py | Tool invocation |
| TaskRun | nodes/execution.py | Tool result |
| SessionNode | nodes/session.py | Execution context |
| SummaryNode | nodes/session.py | Checkpoint |
| ApprovalNode | nodes/workflow.py | Human approval gate |
| ArtifactNode | nodes/artifact.py | Artifact lineage |
| JobNode | nodes/job.py | High-level task |
| JobRunNode | nodes/job.py | Task execution |
| UserMessage | extensions/llm/nodes.py | Chat message |
| AssistantMessage | extensions/llm/nodes.py | LLM response |
| SystemMessage | extensions/llm/nodes.py | System prompt |

---

## Edge Type Reference

| Type | Location | Purpose |
|------|----------|---------|
| ParentChildEdge | edges/hierarchy.py | Containment |
| PlanLinkEdge | edges/planning.py | Plan component link |
| StepEdge | edges/planning.py | Step dependency |
| RouteEdge | edges/routing.py | Routing path |
| NextEdge | edges/ordering.py | Sequential order |
| CustomEdge | edges/ordering.py | Custom relationship |
| ApprovalEdge | edges/workflow.py | Approval routing |
| FallbackEdge | edges/workflow.py | Error recovery |
| ArtifactDependencyEdge | edges/workflow.py | Artifact flow |

---

## File Statistics

- **Total Core Code**: ~3,500 lines
- **Pure Pydantic Models**: ~700 lines
- **Planning Engine**: ~2,100 lines
- **Extensions**: ~120 lines

---

## Key Principles

1. **Pure Pydantic**: No string keys, everything typed
2. **Domain-Agnostic**: Core knows nothing about LLM
3. **Modular**: Extensions in separate namespace
4. **Type-Safe**: Literal types, validators, properties
5. **Immutable**: Frozen models prevent mutations
6. **Async-Native**: GraphStore fully async
7. **Hashable**: Nodes/edges usable in sets/dicts
8. **Extensible**: Clear pattern for new extensions

---

## Questions & Answers

**Q: Where do I find node definitions?**
A: `/src/chuk_ai_planner/core/graph/nodes/` - organized by purpose

**Q: How do I create a plan?**
A: Use `Plan("title").step(...).up()...` fluent API in QUICK_REFERENCE.md

**Q: What's the difference between core and extensions?**
A: Core is domain-agnostic infrastructure; extensions are optional domain-specific features

**Q: How do I add my own node type?**
A: Create extension following template in QUICK_REFERENCE.md section "Creating Custom Extensions"

**Q: What's a GraphStore?**
A: Abstract storage interface for graph persistence - see CODEBASE_STRUCTURE.md section 6

**Q: How does routing work?**
A: Three strategies (expression, LLM, function) - see CODEBASE_STRUCTURE.md section 8

**Q: Are nodes mutable?**
A: No, they're frozen. Can't change after creation.

**Q: Can I use just the core without extensions?**
A: Yes! Extensions are completely optional.

---

## Getting Started Checklist

- [ ] Read QUICK_REFERENCE.md
- [ ] Skim ARCHITECTURE_OVERVIEW.txt diagrams
- [ ] Review `/src/chuk_ai_planner/core/graph/nodes/base.py`
- [ ] Review `/src/chuk_ai_planner/core/graph/types.py`
- [ ] Look at one node type (e.g., PlanNode)
- [ ] Look at one edge type (e.g., ParentChildEdge)
- [ ] Try creating a simple Plan with the DSL
- [ ] (Optional) Read deep dive in CODEBASE_STRUCTURE.md

---

## Navigation Tips

- **Searching for a type**: Check "Node Type Reference" or "Edge Type Reference" tables
- **Understanding Pydantic usage**: See CODEBASE_STRUCTURE.md section 10
- **See code examples**: Look at docstrings in actual Python files
- **Architecture overview**: Start with ARCHITECTURE_OVERVIEW.txt
- **Extending the system**: See QUICK_REFERENCE.md or CODEBASE_STRUCTURE.md section 13

---

Generated: 2025-11-15
Last Updated: 2025-11-15
Thoroughness Level: Very Thorough
