# Import Path Mapping: Old → New

After the refactor, import paths changed. Use this guide to update examples.

## Core Modules

### Planner
```python
# Old
from chuk_ai_planner.planner import Plan, PlanExecutor

# New  
from chuk_ai_planner.core.planner import Plan, PlanExecutor
```

### Graph Store
```python
# Old
from chuk_ai_planner.store.memory import InMemoryGraphStore

# New
from chuk_ai_planner.core.store.memory import InMemoryGraphStore
```

### Graph Nodes & Edges
```python
# Old
from chuk_ai_planner.graph import PlanNode, PlanStep, ...

# New
from chuk_ai_planner.core.graph import PlanNode, PlanStep, ...
```

### Graph Types
```python
# Old
from chuk_ai_planner.graph.types import NodeType, EdgeType, ...

# New
from chuk_ai_planner.core.graph.types import NodeType, EdgeType, ...
```

## Higher-Level APIs

### Universal Plan
```python
# Old (if existed)
from chuk_ai_planner.planner.universal_plan import UniversalPlan

# New
from chuk_ai_planner.core.planner.universal_plan import UniversalPlan
```

### Universal Executor
```python
# Old (if existed)
from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor

# New
from chuk_ai_planner.core.planner.universal_plan_executor import UniversalExecutor
```

### Plan Registry  
```python
# Old (if existed)
from chuk_ai_planner.planner.plan_registry import PlanRegistry

# New
from chuk_ai_planner.core.planner.plan_registry import PlanRegistry
```

## Extensions (Not Yet Implemented)

### LLM Nodes ⚠️
```python
# Old (worked before)
from chuk_ai_planner.graph.nodes.llm import UserMessage, AssistantMessage, SystemMessage

# New (doesn't exist yet!)
# TODO: Implement LLM extension
```

## Complete Mapping Table

| Old Path | New Path | Status |
|----------|----------|--------|
| `chuk_ai_planner.planner` | `chuk_ai_planner.core.planner` | ✅ |
| `chuk_ai_planner.store` | `chuk_ai_planner.core.store` | ✅ |
| `chuk_ai_planner.graph` | `chuk_ai_planner.core.graph` | ✅ |
| `chuk_ai_planner.graph.types` | `chuk_ai_planner.core.graph.types` | ✅ |
| `chuk_ai_planner.graph.nodes.llm` | N/A | ⚠️ Not implemented |

## Quick Fix Script

```bash
# Find all examples with old imports
grep -r "from chuk_ai_planner.planner import" examples/
grep -r "from chuk_ai_planner.store" examples/
grep -r "from chuk_ai_planner.graph import" examples/

# Replace (do manually to be safe)
# Old: from chuk_ai_planner.planner import
# New: from chuk_ai_planner.core.planner import

# Old: from chuk_ai_planner.store
# New: from chuk_ai_planner.core.store

# Old: from chuk_ai_planner.graph
# New: from chuk_ai_planner.core.graph
```
