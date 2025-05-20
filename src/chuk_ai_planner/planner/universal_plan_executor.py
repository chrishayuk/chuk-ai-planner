# chuk_ai_planner/planner/universal_plan_executor.py
"""
Enhanced Universal Plan Executor
================================

Drop-in replacement for ``src/chuk_ai_planner/planner/universal_plan_executor.py``
that fixes:

1. **Variable resolution** - Properly resolves variable references in tool/function arguments
2. **Sequential execution** - Respects step dependencies and ensures correct execution order
3. **Result unwrapping** - Correctly unwraps tool results for downstream steps

This implementation can be used without changing the existing API.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from chuk_session_manager.models.session import Session
from chuk_session_manager.storage import InMemorySessionStore, SessionStoreProvider

from chuk_ai_planner.models.edges import EdgeKind
from chuk_ai_planner.processor import GraphAwareToolProcessor
from chuk_ai_planner.store.base import GraphStore
from chuk_ai_planner.store.memory import InMemoryGraphStore

from .plan_executor import PlanExecutor
from .universal_plan import UniversalPlan

__all__ = ["UniversalExecutor"]


class UniversalExecutor:
    """Execute :class:`~chuk_ai_planner.planner.universal_plan.UniversalPlan` with robust variable handling."""

    # ------------------------------------------------------------------ init
    def __init__(self, graph_store: GraphStore | None = None):
        # Ensure there is a session store
        try:
            SessionStoreProvider.get_store()
        except Exception:
            SessionStoreProvider.set_store(InMemorySessionStore())

        self.session: Session = Session()
        SessionStoreProvider.get_store().save(self.session)

        # Allow caller‑provided graph store (avoids step‑not‑found issue)
        self.graph_store: GraphStore = graph_store or InMemoryGraphStore()

        self.processor = GraphAwareToolProcessor(
            self.session.id,
            self.graph_store,
            enable_caching=True,
            enable_retries=True,
        )
        self.plan_executor = PlanExecutor(self.graph_store)
        self.assistant_node_id: str = str(uuid.uuid4())

        self.tool_registry: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self.function_registry: Dict[str, Callable[..., Any]] = {}

    # ----------------------------------------------------------- registry
    def register_tool(self, name: str, fn: Callable[..., Awaitable[Any]]) -> None:
        """Register a tool function."""
        self.tool_registry[name] = fn
        self.processor.register_tool(name, fn)

    def register_function(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a function that can be called from plan steps."""
        self.function_registry[name] = fn

        async def wrapper(args: Dict[str, Any]):
            fn_name = args.get("function")
            fn_args = args.get("args", {})
            target = self.function_registry.get(fn_name)
            if target is None:
                raise ValueError(f"Unknown function {fn_name!r}")
            if asyncio.iscoroutinefunction(target):
                return await target(**fn_args)
            return target(**fn_args)

        self.processor.register_tool("function", wrapper)

    # ----------------------------------------------------------- variable helpers
    def _resolve_vars(self, value: Any, variables: Dict[str, Any]) -> Any:
        """Recursively resolve variable references in a value."""
        # Handle string variable references like "${varname}"
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]  # Remove ${ and }
            if var_name in variables:
                return variables[var_name]
            return value  # Keep as is if variable not found
        
        # Handle dictionaries
        if isinstance(value, dict):
            return {k: self._resolve_vars(v, variables) for k, v in value.items()}
        
        # Handle lists
        if isinstance(value, list):
            return [self._resolve_vars(item, variables) for item in value]
        
        # Any other type
        return value
    
    def _extract_value(self, obj: Any) -> Any:
        """Return a plain payload regardless of how deeply it's wrapped."""
        # --- 0. None ------------------------------------------------------
        if obj is None:
            return None

        # --- 1. single‑element list --------------------------------------
        if isinstance(obj, list):
            if len(obj) == 1:
                return self._extract_value(obj[0])
            return [self._extract_value(x) for x in obj]

        # --- 2. dicts -----------------------------------------------------
        if isinstance(obj, dict):
            val = obj
            # peel layers of {"result": …}, {"data": …}, {"payload": …}
            while (
                isinstance(val, dict)
                and len(val) == 1
                and next(iter(val)) in ("result", "payload", "data")
            ):
                val = next(iter(val.values()))
            return val

        # --- 3. objects with common attributes ---------------------------
        for attr in ("result", "payload", "data"):
            if hasattr(obj, attr):
                inner = getattr(obj, attr)
                if inner is not None:
                    return self._extract_value(inner)

        # --- 4. dataclass -------------------------------------------------
        if is_dataclass(obj):
            return asdict(obj)

        # --- 5. fallback --------------------------------------------------
        return getattr(obj, "__dict__", obj)

    # ----------------------------------------------------------- topological sort
    def _topological_sort(self, steps: List[Any], dependencies: Dict[str, Set[str]]) -> List[Any]:
        """Sort steps based on dependencies using topological sort."""
        # Create a mapping from step ID to step object
        id_to_step = {step.id: step for step in steps}
        
        # Track visited and temp markers for cycle detection
        visited = set()
        temp_mark = set()
        
        # Result list
        sorted_steps = []
        
        def visit(step_id):
            if step_id in temp_mark:
                raise ValueError(f"Dependency cycle detected involving step {step_id}")
            
            if step_id not in visited:
                temp_mark.add(step_id)
                
                # Visit dependencies
                for dep_id in dependencies.get(step_id, set()):
                    visit(dep_id)
                
                temp_mark.remove(step_id)
                visited.add(step_id)
                
                # Add to result
                if step_id in id_to_step:
                    sorted_steps.append(id_to_step[step_id])
        
        # Visit all steps
        for step in steps:
            if step.id not in visited:
                visit(step.id)
        
        return sorted_steps

    # ----------------------------------------------------------- execute single step
    async def _execute_step(self, step: Any, context: Dict[str, Any]) -> List[Any]:
        """Execute a single step and store its results in the context."""
        step_id = step.id
        
        # Find tool calls for this step
        results = []
        for edge in self.graph_store.get_edges(src=step_id, kind=EdgeKind.PLAN_LINK):
            tool_node = self.graph_store.get_node(edge.dst)
            if tool_node and tool_node.kind.value == "tool_call":
                # Get tool info
                tool_name = tool_node.data.get("name")
                args = tool_node.data.get("args", {})
                
                # Resolve variables in args
                resolved_args = self._resolve_vars(args, context["variables"])
                
                # Execute the appropriate function
                if tool_name == "function":
                    # Handle function calls
                    fn_name = resolved_args.get("function")
                    fn_args = resolved_args.get("args", {})
                    
                    # Further resolve function args
                    fn_args = self._resolve_vars(fn_args, context["variables"])
                    
                    fn = self.function_registry.get(fn_name)
                    if fn is None:
                        raise ValueError(f"Unknown function: {fn_name}")
                    
                    # Call function with args
                    if asyncio.iscoroutinefunction(fn):
                        result = await fn(**fn_args)
                    else:
                        result = fn(**fn_args)
                else:
                    # Format for tool_call
                    tool_call = {
                        "id": tool_node.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(resolved_args),
                        },
                    }
                    
                    # Direct execution of the function without using processor's API
                    fn = self.tool_registry.get(tool_name)
                    if fn is None:
                        raise ValueError(f"Unknown tool: {tool_name}")
                    
                    # Execute the tool function directly
                    if asyncio.iscoroutinefunction(fn):
                        result = await fn(resolved_args)
                    else:
                        result = fn(resolved_args)
                
                results.append(result)
        
        # Update context with results
        context["results"][step_id] = results
        
        # Process step results - store in variables
        self._process_step_results(step_id, results, context)
        
        return results

    # ----------------------------------------------------------- execution
    async def execute_plan(self, plan: UniversalPlan, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a UniversalPlan with proper variable resolution.
        
        Parameters
        ----------
        plan : UniversalPlan
            The plan to execute
        variables : Dict[str, Any], optional
            Initial variables for the plan
            
        Returns
        -------
        Dict[str, Any]
            The execution result, with 'success', 'variables', and 'results' keys
        """
        # Copy plan graph into our store if necessary
        if plan.graph is not self.graph_store:
            for node in plan.graph.nodes.values():
                self.graph_store.add_node(node)
            for edge in plan.graph.edges:
                self.graph_store.add_edge(edge)

        if not plan._indexed:
            plan.save()

        ctx: Dict[str, Any] = {
            "variables": {**plan.variables, **(variables or {})},
            "results": {},
        }

        try:
            # Get all steps for the plan
            steps = self.plan_executor.get_plan_steps(plan.id)
            
            # Build dependency map
            step_dependencies: Dict[str, Set[str]] = {}
            for step in steps:
                deps = set()
                # Get explicit dependencies from STEP_ORDER edges
                for edge in self.graph_store.get_edges(dst=step.id, kind=EdgeKind.STEP_ORDER):
                    deps.add(edge.src)
                step_dependencies[step.id] = deps
            
            # Sort steps topologically
            sorted_steps = self._topological_sort(steps, step_dependencies)
            
            # Execute steps in order
            for step in sorted_steps:
                await self._execute_step(step, ctx)
            
            return {"success": True, **ctx}
        except Exception as exc:
            return {"success": False, "error": str(exc), **ctx}

    # ----------------------------------------------------------- helpers
    def _process_step_results(self, step_id: str, results: Any, context: Dict[str, Any]):
        """Process step results and store them in the variables dict."""
        if not results:
            return True
        
        first = results[0] if isinstance(results, list) else results
        value = self._extract_value(first)

        for edge in self.graph_store.get_edges(src=step_id, kind=EdgeKind.CUSTOM):
            if edge.data.get("type") == "result_variable":
                context["variables"][edge.data["variable"]] = value
        return True

    # ----------------------------------------------------------- convenience
    async def execute_plan_by_id(self, plan_id: str, variables: Optional[Dict[str, Any]] = None):
        """
        Execute a plan by its ID.
        
        Parameters
        ----------
        plan_id : str
            The ID of the plan to execute
        variables : Dict[str, Any], optional
            Initial variables for the plan
            
        Returns
        -------
        Dict[str, Any]
            The execution result
        """
        node = self.graph_store.get_node(plan_id)
        if node is None:
            raise ValueError(f"Plan {plan_id} not found")
        plan = UniversalPlan(title=node.data.get("title", "Plan"), id=plan_id, graph=self.graph_store)
        return await self.execute_plan(plan, variables)