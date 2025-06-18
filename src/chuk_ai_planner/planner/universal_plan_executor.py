# chuk_ai_planner/planner/universal_plan_executor.py
"""
Universal Plan Executor - ENHANCED VERSION WITH FIXED VARIABLE RESOLUTION
================================
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from types import MappingProxyType

from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.session_storage import setup_chuk_sessions_storage

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
        # Don't create session immediately - defer to async method
        self.session = None
        self._session_initialized = False

        # Allow caller‑provided graph store (avoids step‑not‑found issue)
        self.graph_store: GraphStore = graph_store or InMemoryGraphStore()

        self.processor = None  # Will be initialized when session is ready
        self.plan_executor = PlanExecutor(self.graph_store)
        self.assistant_node_id: str = str(uuid.uuid4())

        self.tool_registry: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self.function_registry: Dict[str, Callable[..., Any]] = {}

    # ----------------------------------------------------------- async setup
    async def _ensure_session(self):
        """Ensure session is initialized (async)"""
        if not self._session_initialized:
            # Set up session storage using the correct API
            setup_chuk_sessions_storage(sandbox_id="universal-executor", default_ttl_hours=2)
            
            # Create session
            self.session = await Session.create()
            
            # Now initialize the processor
            self.processor = GraphAwareToolProcessor(
                self.session.id,
                self.graph_store,
                enable_caching=True,
                enable_retries=True,
            )
            
            self._session_initialized = True

    # ----------------------------------------------------------- registry
    def register_tool(self, name: str, fn: Callable[..., Awaitable[Any]]) -> None:
        """Register a tool function."""
        self.tool_registry[name] = fn

    def register_function(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a function that can be called from plan steps."""
        self.function_registry[name] = fn

    async def _register_tools_with_processor(self):
        """Register tools with processor after it's initialized"""
        if self.processor is None:
            await self._ensure_session()
        
        # Register all tools
        for name, fn in self.tool_registry.items():
            self.processor.register_tool(name, fn)
        
        # Register function wrapper
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

    # ----------------------------------------------------------- JSON serialization helpers
    def _get_json_serializable_data(self, data: Any) -> Any:
        """
        Convert potentially frozen data structures to JSON-serializable format.
        Be extra careful about type preservation - preserve dictionaries as dictionaries.
        """
        try:
            # Try to import _ReadOnlyList if it exists
            from ..models.base import _ReadOnlyList
        except ImportError:
            # If not available, create a dummy class that will never match
            class _ReadOnlyList:
                pass
        
        if isinstance(data, MappingProxyType):
            # Convert MappingProxyType to regular dict
            return {k: self._get_json_serializable_data(v) for k, v in data.items()}
        elif isinstance(data, _ReadOnlyList):
            # Convert _ReadOnlyList to regular list
            return [self._get_json_serializable_data(item) for item in data]
        elif isinstance(data, dict):
            # Handle nested dicts that might contain frozen structures
            # CRITICAL: Preserve dict structure - don't convert to list
            return {k: self._get_json_serializable_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            # Handle regular lists and tuples
            return [self._get_json_serializable_data(item) for item in data]
        elif isinstance(data, frozenset):
            # Convert frozensets to lists for JSON compatibility
            return [self._get_json_serializable_data(item) for item in data]
        elif hasattr(data, '__iter__') and hasattr(data, '__getitem__') and hasattr(data, '__len__'):
            # This catches other list-like objects, but we need to be careful not to catch strings or dicts
            if isinstance(data, (str, bytes, dict)):
                # These are iterable but should not be converted to lists
                return data
            else:
                # It's a list-like object, convert to list
                try:
                    return [self._get_json_serializable_data(item) for item in data]
                except (TypeError, AttributeError):
                    # If iteration fails, return as-is
                    return data
        else:
            # Primitive types are already JSON serializable
            return data

    # ----------------------------------------------------------- ENHANCED variable helpers
    def _resolve_vars(self, value: Any, variables: Dict[str, Any]) -> Any:
        """
        ENHANCED: Recursively resolve variable references with support for nested field access.
        Supports both ${variable} and ${variable.field.subfield} syntax, including template strings.
        """
        # Handle string variable references and template strings
        if isinstance(value, str):
            # Check if the entire string is a single variable reference
            if value.startswith("${") and value.endswith("}") and value.count("${") == 1:
                var_path = value[2:-1]  # Remove ${ and }
                return self._resolve_nested_variable(var_path, variables)
            
            # Check if string contains variable references (template string)
            elif "${" in value:
                return self._resolve_template_string(value, variables)
            
            # Regular string with no variables
            return value
        
        # Handle dictionaries (both regular and MappingProxyType)
        elif isinstance(value, (dict, MappingProxyType)):
            # Convert to regular dict and recursively resolve
            return {k: self._resolve_vars(v, variables) for k, v in value.items()}
        
        # Handle lists and tuples (including _ReadOnlyList)
        elif isinstance(value, (list, tuple)):
            return [self._resolve_vars(item, variables) for item in value]
        
        # Handle other iterable types carefully
        elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
            # Check if it's a string-like object to avoid infinite recursion
            if hasattr(value, 'replace') or hasattr(value, 'split'):
                return value  # It's a string-like object, return as-is
            else:
                try:
                    # Try to iterate and resolve each item
                    return [self._resolve_vars(item, variables) for item in value]
                except (TypeError, AttributeError):
                    # If iteration fails, return as-is
                    return value
        
        # Any other type (int, float, bool, None, etc.)
        else:
            return value
    
    def _resolve_template_string(self, template: str, variables: Dict[str, Any]) -> str:
        """
        ENHANCED: Resolve template strings containing multiple variable references.
        Example: "https://${api.endpoint}:${api.port}/users/${user.id}"
        """
        import re
        
        def replace_var(match):
            var_path = match.group(1)  # Extract content between ${ and }
            resolved = self._resolve_nested_variable(var_path, variables)
            
            # If resolution failed (returns original ${...}), keep as-is
            if isinstance(resolved, str) and resolved.startswith("${") and resolved.endswith("}"):
                return resolved
            
            # Convert resolved value to string for template interpolation
            return str(resolved)
        
        # Find all ${...} patterns and replace them
        pattern = r'\$\{([^}]+)\}'
        result = re.sub(pattern, replace_var, template)
        return result
    
    def _resolve_nested_variable(self, var_path: str, variables: Dict[str, Any]) -> Any:
        """
        ENHANCED: Resolve nested variable access like 'variable.field.subfield'.
        """
        parts = var_path.split('.')
        current = variables
        
        for i, part in enumerate(parts):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                # Variable or field not found
                print(f"🔍 Variable resolution: '{part}' not found in {'.'.join(parts[:i]) or 'variables'}")
                print(f"🔍 Available keys: {list(current.keys()) if isinstance(current, dict) else 'not a dict'}")
                return f"${{{var_path}}}"  # Return original variable string if not found
        
        return current
    
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

    # ----------------------------------------------------------- FIXED: result variable lookup
    def _find_result_variable(self, step_id: str, tool_id: str = None) -> Optional[str]:
        """
        Find the result variable for a step by checking custom edges.
        This is the key fix - properly retrieve result_variable from custom edges.
        """
        # Look for custom edges from step to tool with result_variable data
        for edge in self.graph_store.get_edges(src=step_id, kind=EdgeKind.CUSTOM):
            edge_data = edge.data or {}
            if (edge_data.get("type") == "result_variable" and 
                (tool_id is None or edge.dst == tool_id)):
                return edge_data.get("variable")
        
        # Fallback: check if tool_id is provided and has result_variable in its data
        if tool_id:
            tool_node = self.graph_store.get_node(tool_id)
            if tool_node:
                return tool_node.data.get("result_variable")
        
        return None

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

    # ----------------------------------------------------------- FIXED: execute single step with deduplication
    async def _execute_step(self, step: Any, context: Dict[str, Any]) -> List[Any]:
        """Execute a single step and return results as a list (matching test expectations)."""
        step_id = step.id
        
        # FIXED: Check if step has already been executed to prevent duplicates
        if step_id in context.get("executed_steps", set()):
            print(f"🔍 Step {step_id[:8]} already executed, skipping")
            return context["results"].get(step_id, [])
        
        # Mark step as executed
        if "executed_steps" not in context:
            context["executed_steps"] = set()
        context["executed_steps"].add(step_id)
        
        # Find tool calls for this step
        results = []
        
        # FIXED: Deduplicate tool calls by tracking executed tool call IDs
        executed_tool_calls = context.get("executed_tool_calls", set())
        
        for edge in self.graph_store.get_edges(src=step_id, kind=EdgeKind.PLAN_LINK):
            tool_node = self.graph_store.get_node(edge.dst)
            if tool_node and tool_node.kind.value == "tool_call":
                
                # FIXED: Skip if this tool call was already executed
                if tool_node.id in executed_tool_calls:
                    print(f"🔍 Tool call {tool_node.id[:8]} already executed, skipping")
                    continue
                
                executed_tool_calls.add(tool_node.id)
                context["executed_tool_calls"] = executed_tool_calls
                
                # Get tool info
                tool_name = tool_node.data.get("name")
                args = tool_node.data.get("args", {})
                
                # FIXED: Find result variable using the new method
                result_variable = self._find_result_variable(step_id, tool_node.id)
                
                # ENHANCED: Resolve variables in args with nested field support
                resolved_args = self._resolve_vars(args, context["variables"])
                
                # Convert to JSON-serializable format - PRESERVE DICT STRUCTURE
                json_safe_args = self._get_json_serializable_data(resolved_args)
                
                # Ensure we still have a dict for tool execution
                if not isinstance(json_safe_args, dict):
                    raise ValueError(f"Tool args must be a dictionary, got {type(json_safe_args)}: {json_safe_args}")
                
                try:
                    # Execute the appropriate function
                    if tool_name == "function":
                        # Handle function calls
                        fn_name = json_safe_args.get("function")
                        fn_args = json_safe_args.get("args", {})
                        
                        # ENHANCED: Resolve variables in function args again with nested support
                        fn_args = self._resolve_vars(fn_args, context["variables"])
                        fn_args = self._get_json_serializable_data(fn_args)
                        
                        # Ensure fn_args is a dict
                        if not isinstance(fn_args, dict):
                            raise ValueError(f"Function args must be a dictionary, got {type(fn_args)}: {fn_args}")
                        
                        fn = self.function_registry.get(fn_name)
                        if fn is None:
                            raise ValueError(f"Unknown function: {fn_name}")
                        
                        # Call function with args (ensure async-native handling)
                        if asyncio.iscoroutinefunction(fn):
                            result = await fn(**fn_args)
                        else:
                            result = fn(**fn_args)
                    else:
                        # Direct execution of the tool function
                        fn = self.tool_registry.get(tool_name)
                        if fn is None:
                            raise ValueError(f"Unknown tool: {tool_name}")
                        
                        # Execute the tool function directly with JSON-safe args (async-native)
                        if asyncio.iscoroutinefunction(fn):
                            result = await fn(json_safe_args)
                        else:
                            # For sync functions, we can still call them directly
                            result = fn(json_safe_args)
                    
                    # Store result for return
                    results.append(result)
                    
                    # FIXED: Store result immediately if we have a result_variable
                    if result_variable:
                        context["variables"][result_variable] = result
                    
                except Exception as e:
                    # For test compatibility, we need to raise the exception
                    # rather than return an error dict
                    raise e
        
        # Update context with results for other methods that might use it
        context["results"][step_id] = results
        
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
        # Ensure session is initialized
        await self._ensure_session()
        await self._register_tools_with_processor()
        
        # Copy plan graph into our store if necessary
        if plan.graph is not self.graph_store:
            for node in plan.graph.nodes.values():
                self.graph_store.add_node(node)
            for edge in plan.graph.edges:
                self.graph_store.add_edge(edge)

        # Ensure plan is saved/indexed
        if not plan._indexed:
            plan.save()

        ctx: Dict[str, Any] = {
            "variables": {**plan.variables, **(variables or {})},
            "results": {},
            "executed_steps": set(),  # FIXED: Track executed steps
            "executed_tool_calls": set(),  # FIXED: Track executed tool calls
        }

        try:
            # Get all steps for the plan - try multiple approaches
            steps = self.plan_executor.get_plan_steps(plan.id)
            
            # If no steps found via plan_executor, search directly
            if not steps:
                # Method 1: Find all plan_step nodes in the graph
                steps = [node for node in self.graph_store.nodes.values() 
                        if node.kind.value == "plan_step"]
                
                # Method 2: If still no steps, check if there are any tool_call nodes
                # that might be orphaned (this shouldn't happen but let's be safe)
                if not steps:
                    tool_calls = [node for node in self.graph_store.nodes.values() 
                                 if node.kind.value == "tool_call"]
                    
                    # For each tool call, try to execute it directly
                    for tool_node in tool_calls:
                        result = await self._execute_tool_directly(tool_node, ctx)
                    
                    return {"success": True, **ctx}
            
            if not steps:
                # Truly no steps found - return success with original variables
                return {"success": True, **ctx}
            
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
                step_results = await self._execute_step(step, ctx)
            
            # Clean up execution tracking from context before returning
            ctx.pop("executed_steps", None)
            ctx.pop("executed_tool_calls", None)
            
            return {"success": True, **ctx}
        except Exception as exc:
            return {"success": False, "error": str(exc), **ctx}

    # ----------------------------------------------------------- direct tool execution
    async def _execute_tool_directly(self, tool_node: Any, context: Dict[str, Any]):
        """Execute a tool node directly (fallback method)."""
        tool_name = tool_node.data.get("name")
        args = tool_node.data.get("args", {})
        result_variable = tool_node.data.get("result_variable")
        
        # Resolve variables in args
        resolved_args = self._resolve_vars(args, context["variables"])
        json_safe_args = self._get_json_serializable_data(resolved_args)
        
        if not isinstance(json_safe_args, dict):
            return None
        
        try:
            # Execute the tool
            if tool_name == "function":
                fn_name = json_safe_args.get("function")
                fn_args = json_safe_args.get("args", {})
                
                if not isinstance(fn_args, dict):
                    return None
                
                fn = self.function_registry.get(fn_name)
                if fn is None:
                    return None
                
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(**fn_args)
                else:
                    result = fn(**fn_args)
            else:
                fn = self.tool_registry.get(tool_name)
                if fn is None:
                    return None
                
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(json_safe_args)
                else:
                    result = fn(json_safe_args)
            
            # Store result if result_variable is specified
            if result_variable:
                context["variables"][result_variable] = result
            
            return result
            
        except Exception:
            return None

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