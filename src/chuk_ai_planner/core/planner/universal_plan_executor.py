# chuk_ai_planner/planner/universal_plan_executor.py
"""
Universal Plan Executor - ENHANCED VERSION WITH FIXED VARIABLE RESOLUTION
================================
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import asdict, is_dataclass
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from chuk_session_manager.models.session import Session  # type: ignore
from chuk_session_manager.storage import (  # type: ignore
    InMemorySessionStore,
    SessionStoreProvider,
)

from chuk_ai_planner.core.graph import EdgeType, PlanStep, RouteEdge, ToolCall
from chuk_ai_planner.core.graph.types import NodeType
from chuk_ai_planner.core.routing import RoutingExecutor
from chuk_ai_planner.core.store.base import GraphStore
from chuk_ai_planner.core.store.memory import InMemoryGraphStore
from chuk_ai_planner.execution.interfaces import ToolExecutionBackend
from chuk_ai_planner.execution.models import ToolExecutionRequest
from chuk_ai_planner.processor import GraphAwareToolProcessor

from .plan_executor import PlanExecutor
from .universal_plan import EDGE_TYPE_RESULT_VARIABLE, KEY_VARIABLE, UniversalPlan

_log = logging.getLogger(__name__)

# Context keys
CTX_VARIABLES = "variables"
CTX_RESULTS = "results"
CTX_EXECUTED_STEPS = "executed_steps"
CTX_EXECUTED_TOOL_CALLS = "executed_tool_calls"
CTX_ROUTING_DECISIONS = "routing_decisions"
CTX_SKIPPED_STEPS = "skipped_steps"
CTX_SUCCESS = "success"
CTX_ERROR = "error"

__all__ = ["UniversalExecutor"]


class UniversalExecutor:
    """Execute :class:`~chuk_ai_planner.planner.universal_plan.UniversalPlan` with robust variable handling."""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        graph_store: GraphStore | None = None,
        tool_backend: ToolExecutionBackend | None = None,
        *,
        processor: Any = None,
        namespace: Optional[str] = None,
    ):
        """
        Initialize UniversalExecutor with chuk-tool-processor.

        Args:
            graph_store: Optional graph store to use
            tool_backend: Optional custom backend (advanced usage)
            processor: Optional ToolProcessor instance. If None, creates a new one.
            namespace: Optional namespace prefix for tool names (e.g., "github")

        Note:
            The default execution backend is ToolProcessorBackend, providing:
            - Python functions (via register_fn_tool)
            - MCP tools (Notion, GitHub, etc.)
            - ACP agents
            - Container execution
            - Built-in retries, caching, rate limiting
        """
        # Ensure there is a session store
        try:
            SessionStoreProvider.get_store()
        except Exception:
            SessionStoreProvider.set_store(InMemorySessionStore())

        # Don't create session immediately - defer to async method
        self.session = None
        self._session_initialized = False

        # Allow caller‑provided graph store (avoids step‑not‑found issue)
        self.graph_store: GraphStore = graph_store or InMemoryGraphStore()

        self.processor = None  # Will be initialized when session is ready
        self.plan_executor = PlanExecutor(self.graph_store)
        self.routing_executor = RoutingExecutor(self.graph_store)
        self.assistant_node_id: str = str(uuid.uuid4())

        # Store processor config for lazy initialization
        self._processor_instance = processor
        self._namespace = namespace

        # Tool execution backend (defaults to CTP)
        if tool_backend is None:
            # Will be created in _ensure_session
            self.tool_backend = None
            self._needs_backend_init = True
        else:
            self.tool_backend = tool_backend
            self._needs_backend_init = False

    # ----------------------------------------------------------- factory methods
    @classmethod
    async def with_tool_processor(
        cls,
        graph_store: GraphStore | None = None,
        *,
        processor: Any = None,
        namespace: Optional[str] = None,
    ) -> "UniversalExecutor":
        """
        Create a UniversalExecutor using chuk-tool-processor for tool execution.

        This makes the planner the universal agent runtime for the CHUK ecosystem,
        supporting:
        - Local tools (@tool decorator)
        - MCP tools (setup_mcp_stdio, setup_mcp_sse, etc.)
        - ACP agents
        - Container-based execution
        - Built-in reliability (retries, circuit breakers, rate limits)

        Args:
            graph_store: Optional graph store to use
            processor: Optional ToolProcessor instance. If None, creates a new one.
            namespace: Optional namespace prefix for tool names (e.g., "mcp")

        Returns:
            UniversalExecutor configured to use chuk-tool-processor

        Example:
            ```python
            from chuk_tool_processor import ToolProcessor, tool
            from chuk_ai_planner.core.planner import UniversalPlan
            from chuk_ai_planner.core.planner.universal_plan_executor import UniversalExecutor

            @tool(name="fetch_data")
            class FetchData:
                async def execute(self, url: str) -> dict:
                    return {"url": url, "data": [1, 2, 3]}

            # Create executor with tool processor
            async with ToolProcessor() as tp:
                executor = await UniversalExecutor.with_tool_processor(
                    processor=tp,
                    namespace=None,
                )

                # Create and execute plan
                plan = UniversalPlan(title="Data Pipeline", graph=executor.graph_store)
                await plan.add_tool_step(
                    title="Fetch",
                    tool_name="fetch_data",
                    args={"url": "/api/data"},
                    result_variable="data",
                )
                plan_id = await plan.save()
                results = await executor.execute(plan_id)
            ```
        """
        from chuk_ai_planner.execution.ctp_backend import ToolProcessorBackend

        # Import ToolProcessor here to avoid hard dependency
        if processor is None:
            from chuk_tool_processor import ToolProcessor

            processor = ToolProcessor()

        # Create backend
        backend = ToolProcessorBackend(processor=processor, namespace=namespace)

        # Create executor with backend
        return cls(graph_store=graph_store, tool_backend=backend)

    # ----------------------------------------------------------- async setup
    async def _ensure_session(self):
        """Ensure session is initialized (async)"""
        if not self._session_initialized:
            self.session = Session()
            store = SessionStoreProvider.get_store()
            await store.save(self.session)

            # Initialize ToolProcessor backend if needed
            if self._needs_backend_init:
                from chuk_ai_planner.execution.ctp_backend import ToolProcessorBackend

                if self._processor_instance is None:
                    try:
                        from chuk_tool_processor import ToolProcessor

                        self._processor_instance = ToolProcessor()
                    except ImportError as e:
                        raise ImportError(
                            "chuk-tool-processor is required but could not be imported. "
                            "Install it with: uv pip install chuk-tool-processor"
                        ) from e

                self.tool_backend = ToolProcessorBackend(
                    processor=self._processor_instance, namespace=self._namespace
                )

            # Now initialize the processor
            self.processor = GraphAwareToolProcessor(
                self.session.id,
                self.graph_store,
                enable_caching=True,
                enable_retries=True,
            )

            self._session_initialized = True

    # ----------------------------------------------------------- registry
    async def register_tool(self, name: str, fn: Callable[..., Awaitable[Any]]) -> None:
        """
        Register a tool function with chuk-tool-processor.

        Args:
            name: Name of the tool
            fn: Async or sync callable to execute

        Note:
            Tools registered here get automatic retries, caching, and rate limiting
            from chuk-tool-processor. Use the same registry for all tool types
            (Python functions, MCP tools, ACP agents, containers).

        Example:
            ```python
            async def add_numbers(a: int, b: int) -> dict:
                return {"sum": a + b}

            executor = UniversalExecutor()
            await executor.register_tool("add", add_numbers)
            ```
        """
        # Register with CTP's global registry (production)
        try:
            from chuk_tool_processor.registry.auto_register import register_fn_tool

            await register_fn_tool(
                fn, name=name, namespace=self._namespace or "default"
            )
        except ImportError:
            # In tests, CTP might not be available - that's OK
            pass

        # Ensure backend is initialized
        await self._ensure_session()

        # Register with the backend's processor (for both production and tests)
        if self.tool_backend and hasattr(self.tool_backend, "_processor"):
            processor = self.tool_backend._processor
            if hasattr(processor, "register_fn_tool"):
                await processor.register_fn_tool(
                    fn, name=name, namespace=self._namespace or "default"
                )

        # Also register with GraphAwareToolProcessor if it exists
        if self.processor is not None:
            self.processor.register_tool(name, fn)

    async def register_function(self, name: str, fn: Callable[..., Any]) -> None:
        """
        Register a function that can be called from plan steps.

        Args:
            name: Name of the function
            fn: Async or sync callable to execute

        Note:
            This is an alias for register_tool. All tools are registered the same way.
        """
        await self.register_tool(name, fn)

    async def _register_tools_with_processor(self):
        """
        Ensure session and processor are initialized.

        Note: Tools are now registered directly with CTP via register_tool(),
        not through this method. This method just ensures the processor exists.
        """
        if self.processor is None:
            await self._ensure_session()

    # ----------------------------------------------------------- JSON serialization helpers
    def _get_json_serializable_data(self, data: Any) -> Any:
        """
        Convert potentially frozen data structures to JSON-serializable format.
        Be extra careful about type preservation - preserve dictionaries as dictionaries.
        """
        try:  # pragma: no cover - future-proofing for when models.base exists
            # Try to import _ReadOnlyList if it exists
            from ..models.base import _ReadOnlyList  # type: ignore

            _ReadOnlyListType = _ReadOnlyList
        except (
            ImportError
        ):  # pragma: no cover - fallback for when models.base doesn't exist
            # If not available, create a dummy class that will never match
            _ReadOnlyListType = type(None)  # Use a type that will never match

        if isinstance(data, MappingProxyType):
            # Convert MappingProxyType to regular dict
            return {k: self._get_json_serializable_data(v) for k, v in data.items()}
        elif isinstance(
            data, _ReadOnlyListType
        ):  # pragma: no cover - _ReadOnlyList rarely used
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
        elif (
            hasattr(data, "__iter__")
            and hasattr(data, "__getitem__")
            and hasattr(data, "__len__")
        ):
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
            if (
                value.startswith("${")
                and value.endswith("}")
                and value.count("${") == 1
            ):
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
        elif hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            # Check if it's a string-like object to avoid infinite recursion
            if hasattr(value, "replace") or hasattr(value, "split"):
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

        def replace_var(match):
            var_path = match.group(1)  # Extract content between ${ and }
            resolved = self._resolve_nested_variable(var_path, variables)

            # If resolution failed (returns original ${...}), keep as-is
            if (
                isinstance(resolved, str)
                and resolved.startswith("${")
                and resolved.endswith("}")
            ):
                return resolved

            # Convert resolved value to string for template interpolation
            return str(resolved)

        # Find all ${...} patterns and replace them
        pattern = r"\$\{([^}]+)\}"
        result = re.sub(pattern, replace_var, template)
        return result

    def _resolve_nested_variable(self, var_path: str, variables: Dict[str, Any]) -> Any:
        """
        ENHANCED: Resolve nested variable access like 'variable.field.subfield'.
        """
        parts = var_path.split(".")
        current = variables

        for i, part in enumerate(parts):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                # Variable or field not found
                print(
                    f"🔍 Variable resolution: '{part}' not found in {'.'.join(parts[:i]) or 'variables'}"
                )
                print(
                    f"🔍 Available keys: {list(current.keys()) if isinstance(current, dict) else 'not a dict'}"
                )
                return (
                    f"${{{var_path}}}"  # Return original variable string if not found
                )

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
            return asdict(obj)  # type: ignore[arg-type]

        # --- 5. fallback --------------------------------------------------
        return getattr(obj, "__dict__", obj)

    # ----------------------------------------------------------- FIXED: result variable lookup
    async def _find_result_variable(
        self, step_id: str, tool_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Find the result variable for a step by checking custom edges.
        This is the key fix - properly retrieve result_variable from custom edges.
        """
        # Look for custom edges from step to tool with result_variable type
        for edge in await self.graph_store.get_edges(src=step_id, kind=EdgeType.CUSTOM):
            # Check if edge has custom_type field (typed CustomEdge)
            if (
                hasattr(edge, "custom_type")
                and edge.custom_type == EDGE_TYPE_RESULT_VARIABLE
            ):
                if tool_id is None or edge.dst == tool_id:
                    # Variable name is in metadata
                    metadata = edge.metadata or {}
                    return metadata.get(KEY_VARIABLE)

        return None

    # ----------------------------------------------------------- topological sort
    def _topological_sort(
        self, steps: List[Any], dependencies: Dict[str, Set[str]]
    ) -> List[Any]:
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
        if step_id in context.get(CTX_EXECUTED_STEPS, set()):
            print(f"🔍 Step {step_id[:8]} already executed, skipping")
            return context[CTX_RESULTS].get(step_id, [])

        # Mark step as executed
        if CTX_EXECUTED_STEPS not in context:
            context[CTX_EXECUTED_STEPS] = set()
        context[CTX_EXECUTED_STEPS].add(step_id)

        # ROUTING: Check if this is a router step
        if (
            step.kind == NodeType.ROUTER_STEP
        ):  # pragma: no cover - routing not yet implemented
            print(f"🔀 Router step detected: {step_id[:8]}")
            return await self._handle_router_step(step, context)

        # Find tool calls for this step
        results = []

        # FIXED: Deduplicate tool calls by tracking executed tool call IDs
        executed_tool_calls = context.get(CTX_EXECUTED_TOOL_CALLS, set())

        for edge in await self.graph_store.get_edges(
            src=step_id, kind=EdgeType.PLAN_LINK
        ):
            tool_node = await self.graph_store.get_node(edge.dst)
            if not tool_node or not isinstance(tool_node, ToolCall):
                continue

            # FIXED: Skip if this tool call was already executed
            if tool_node.id in executed_tool_calls:
                print(f"🔍 Tool call {tool_node.id[:8]} already executed, skipping")
                continue

            executed_tool_calls.add(tool_node.id)
            context[CTX_EXECUTED_TOOL_CALLS] = executed_tool_calls

            # Get tool info (type-safe now!)
            tool_name = tool_node.name
            args = tool_node.args

            # Validate tool/function name is not empty
            if not tool_name:
                raise ValueError("function name is required")

            # FIXED: Find result variable using the new method
            result_variable = await self._find_result_variable(step_id, tool_node.id)

            # ENHANCED: Resolve variables in args with nested field support
            resolved_args = self._resolve_vars(args, context[CTX_VARIABLES])

            # Convert to JSON-serializable format - PRESERVE DICT STRUCTURE
            json_safe_args = self._get_json_serializable_data(resolved_args)

            # Ensure we still have a dict for tool execution
            if not isinstance(
                json_safe_args, dict
            ):  # pragma: no cover - defensive check
                raise ValueError(
                    f"Tool args must be a dictionary, got {type(json_safe_args)}: {json_safe_args}"
                )

            # Validate function wrapper calls (after resolving variables)
            if tool_name == "function":
                # This is a function wrapper call - validate it has required fields
                if not json_safe_args.get("function"):
                    raise ValueError("function name is required")
                # Validate the inner args are a dict
                inner_args = json_safe_args.get("args", {})
                if not isinstance(inner_args, dict):
                    raise ValueError(
                        f"Function args must be a dictionary, got {type(inner_args)}"
                    )

            try:
                # Execute via backend (pluggable!) - Pydantic-native
                request = ToolExecutionRequest(
                    tool_name=tool_name,
                    args=json_safe_args,
                    step_id=step_id,
                    session_id=self.session.id if self.session else None,
                )
                if self.tool_backend is None:
                    raise RuntimeError("Tool backend not initialized")
                execution_result = await self.tool_backend.execute_tool(request)

                # Check for errors
                if execution_result.error:
                    raise RuntimeError(
                        f"Tool '{tool_name}' failed: {execution_result.error}"
                    )

                result = execution_result.result

                # Store result for return
                results.append(result)

                # FIXED: Store result immediately if we have a result_variable
                if result_variable:
                    context[CTX_VARIABLES][result_variable] = result

            except Exception as e:
                # For test compatibility, we need to raise the exception
                # rather than return an error dict
                raise e

        # Update context with results for other methods that might use it
        context[CTX_RESULTS][step_id] = results

        return results

    # ----------------------------------------------------------- ROUTING: handle router steps
    async def _handle_router_step(  # pragma: no cover - routing feature not yet implemented
        self, router_step: Any, context: Dict[str, Any]
    ) -> List[Any]:
        """
        Handle a router step by evaluating the routing condition and marking skipped routes.

        Args:
            router_step: The router step node
            context: Execution context

        Returns:
            List with routing decision information
        """
        # Evaluate the routing decision
        decision = await self.routing_executor.evaluate_route(
            router_step=router_step, context=context
        )

        print(f"🔀 Route chosen: {decision.route_key} → {decision.target_step_id[:8]}")
        print(f"🔀 Skipped routes: {', '.join(decision.skipped_routes)}")

        # Store routing decision in context
        if CTX_ROUTING_DECISIONS not in context:
            context[CTX_ROUTING_DECISIONS] = {}
        context[CTX_ROUTING_DECISIONS][router_step.id] = {
            "route_key": decision.route_key,
            "target_step_id": decision.target_step_id,
            "skipped_routes": decision.skipped_routes,
            "evaluation_method": decision.evaluation_method,
            "evaluation_details": decision.evaluation_details,
        }

        # Mark skipped routes and their descendants
        if CTX_SKIPPED_STEPS not in context:
            context[CTX_SKIPPED_STEPS] = set()

        for skipped_route in decision.skipped_routes:
            # Find the target step for this route
            route_edges = await self.routing_executor._get_route_edges(router_step.id)
            for edge in route_edges:
                # Type-safe access to RouteEdge
                if isinstance(edge, RouteEdge) and edge.route_key == skipped_route:
                    # Mark this step and all its descendants as skipped
                    await self._mark_route_skipped(edge.dst, context)
                    break

        # Return routing decision as result
        return [
            {
                "routing_decision": True,
                "route_chosen": decision.route_key,
                "target_step": decision.target_step_id,
                "router_step": router_step.id,
            }
        ]

    async def _mark_route_skipped(  # pragma: no cover - routing feature not yet implemented
        self, step_id: str, context: Dict[str, Any]
    ):
        """
        Mark a route and all its descendants as skipped.

        Args:
            step_id: ID of the step to skip
            context: Execution context
        """
        skipped = context.setdefault(CTX_SKIPPED_STEPS, set())

        # Mark this step as skipped
        skipped.add(step_id)

        # Recursively skip all descendants
        descendants = await self._get_all_descendants(step_id)
        skipped.update(descendants)

        print(
            f"🔀 Marking {len(descendants) + 1} steps as skipped starting from {step_id[:8]}"
        )

    async def _get_all_descendants(  # pragma: no cover - routing feature not yet implemented
        self, step_id: str
    ) -> Set[str]:
        """
        Get all descendant steps from a given step.

        Args:
            step_id: ID of the step

        Returns:
            Set of descendant step IDs
        """
        descendants = set()

        # Get all outgoing edges (PARENT_CHILD, NEXT, STEP_ORDER)
        for edge_kind in [EdgeType.PARENT_CHILD, EdgeType.NEXT, EdgeType.STEP_ORDER]:
            edges = await self.graph_store.get_edges(src=step_id, kind=edge_kind)
            for edge in edges:
                if edge.dst not in descendants:
                    descendants.add(edge.dst)
                    # Recursively get descendants
                    descendants.update(await self._get_all_descendants(edge.dst))

        return descendants

    # ----------------------------------------------------------- execution
    async def execute_plan(
        self, plan: UniversalPlan, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
            # Type-safe access - only InMemoryGraphStore has .nodes and .edges
            if hasattr(plan.graph, "nodes") and hasattr(plan.graph, "edges"):
                for node in plan.graph.nodes.values():  # type: ignore
                    await self.graph_store.add_node(node)
                for edge in plan.graph.edges:  # type: ignore
                    await self.graph_store.add_edge(edge)

        # Ensure plan is saved/indexed
        if not plan._indexed:
            await plan.save()

        ctx: Dict[str, Any] = {
            "variables": {**plan.variables, **(variables or {})},
            "results": {},
            CTX_EXECUTED_STEPS: set(),  # FIXED: Track executed steps
            CTX_EXECUTED_TOOL_CALLS: set(),  # FIXED: Track executed tool calls
        }

        try:
            # Get all steps for the plan - try multiple approaches
            steps = await self.plan_executor.get_plan_steps(plan.id)

            # If no steps found via plan_executor, search directly
            if not steps:
                # Method 1: Find all plan_step nodes in the graph
                # Type-safe access - only InMemoryGraphStore has .nodes
                if hasattr(self.graph_store, "nodes"):
                    steps = [
                        node
                        for node in self.graph_store.nodes.values()  # type: ignore
                        if isinstance(node, PlanStep)
                    ]

                    # Method 2: If still no steps, check if there are any tool_call nodes
                    # that might be orphaned (this shouldn't happen but let's be safe)
                    if not steps:
                        tool_calls = [
                            node
                            for node in self.graph_store.nodes.values()  # type: ignore
                            if isinstance(node, ToolCall)
                        ]

                        # For each tool call, try to execute it directly
                        for (
                            tool_node
                        ) in tool_calls:  # pragma: no cover - fallback execution path
                            await self._execute_tool_directly(tool_node, ctx)

                    return {
                        "success": True,
                        **ctx,
                    }  # pragma: no cover - fallback execution path

            if (
                not steps
            ):  # pragma: no cover - defensive check, plan should always have steps
                # Truly no steps found - return success with original variables
                return {CTX_SUCCESS: True, **ctx}

            # Build dependency map
            step_dependencies: Dict[str, Set[str]] = {}
            for step in steps:
                deps = set()
                # Get explicit dependencies from STEP_ORDER edges
                for edge in await self.graph_store.get_edges(
                    dst=step.id, kind=EdgeType.STEP_ORDER
                ):
                    deps.add(edge.src)
                step_dependencies[step.id] = deps

            # Sort steps topologically
            sorted_steps = self._topological_sort(steps, step_dependencies)

            # Execute steps in order
            for step in sorted_steps:
                # ROUTING: Skip steps that were marked as skipped by routing
                if step.id in ctx.get(
                    CTX_SKIPPED_STEPS, set()
                ):  # pragma: no cover - routing not yet implemented
                    print(f"⏭️  Skipping step {step.id[:8]} (excluded by routing)")
                    continue

                await self._execute_step(step, ctx)

            # Clean up execution tracking from context before returning
            ctx.pop(CTX_EXECUTED_STEPS, None)
            ctx.pop(CTX_EXECUTED_TOOL_CALLS, None)

            return {CTX_SUCCESS: True, **ctx}
        except Exception as exc:
            return {CTX_SUCCESS: False, CTX_ERROR: str(exc), **ctx}

    # ----------------------------------------------------------- direct tool execution
    async def _execute_tool_directly(self, tool_node: Any, context: Dict[str, Any]):
        """
        Execute a tool node using the execution backend.

        Note: This is a fallback path that should rarely be used.
        Normal execution goes through _execute_step which uses the ToolExecutionBackend.
        """
        tool_name = tool_node.name
        args = tool_node.args
        result_variable = tool_node.result_variable

        # Resolve variables in args
        resolved_args = self._resolve_vars(args, context[CTX_VARIABLES])
        json_safe_args = self._get_json_serializable_data(resolved_args)

        if not isinstance(json_safe_args, dict):  # pragma: no cover - defensive check
            return None

        try:
            # Use the execution backend (CTP)
            request = ToolExecutionRequest(
                tool_name=tool_name,
                args=json_safe_args,
                step_id=tool_node.id if hasattr(tool_node, "id") else str(uuid.uuid4()),
                session_id=self.session.id if self.session else None,
            )

            if self.tool_backend is None:
                raise RuntimeError("Tool backend not initialized")
            execution_result = await self.tool_backend.execute_tool(request)

            if not execution_result.success:
                _log.warning(f"Tool '{tool_name}' failed: {execution_result.error}")
                return None

            result = execution_result.result

            # Store result if result_variable is specified
            if result_variable:
                context[CTX_VARIABLES][result_variable] = result

            return result

        except Exception as e:
            _log.error(f"Error executing tool '{tool_name}': {e}")
            return None

    # ----------------------------------------------------------- convenience
    async def execute_plan_by_id(
        self, plan_id: str, variables: Optional[Dict[str, Any]] = None
    ):
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
        node = await self.graph_store.get_node(plan_id)
        if node is None:
            raise ValueError(f"Plan {plan_id} not found")
        # Type-safe access to PlanNode fields
        from chuk_ai_planner.core.graph import PlanNode

        if not isinstance(node, PlanNode):
            raise ValueError(f"Node {plan_id} is not a PlanNode")
        plan = UniversalPlan(title=node.title, id=plan_id, graph=self.graph_store)
        return await self.execute_plan(plan, variables)
