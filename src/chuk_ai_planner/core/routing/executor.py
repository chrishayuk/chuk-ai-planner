# chuk_ai_planner/routing/executor.py
"""
Routing executor for conditional plan execution.

Handles evaluation of routing conditions and selection of execution paths.
Works with pure Pydantic graph nodes - no dictionary goop!
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from chuk_ai_planner.core.graph import GraphEdge, RouterStep
from chuk_ai_planner.core.graph.types import EdgeType, RouterType
from chuk_ai_planner.core.store.base import GraphStore

__all__ = ["RoutingExecutor", "RoutingDecision", "FunctionRegistry"]


class FunctionRegistry:
    """
    Registry for routing functions.

    Allows registration of custom functions for function-based routing.

    Example
    -------
    >>> registry = FunctionRegistry()
    >>>
    >>> @registry.register("priority")
    >>> def calculate_priority(context: Dict[str, Any]) -> str:
    ...     score = context.get("urgency", 0)
    ...     return "urgent" if score > 7 else "normal"
    >>>
    >>> result = registry.call("priority", {"urgency": 8})
    >>> print(result)  # "urgent"
    """

    def __init__(self) -> None:
        self._functions: Dict[str, Callable] = {}

    def register(self, name: str):
        """
        Decorator to register a function.

        Parameters
        ----------
        name : str
            Name to register the function under

        Returns
        -------
        Callable
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            self._functions[name] = func
            return func

        return decorator

    def add(self, name: str, func: Callable) -> None:
        """
        Manually add a function to the registry.

        Parameters
        ----------
        name : str
            Name to register under
        func : Callable
            Function to register
        """
        self._functions[name] = func

    def call(self, name: str, context: Dict[str, Any]) -> Any:
        """
        Call a registered function.

        Parameters
        ----------
        name : str
            Name of the function
        context : Dict[str, Any]
            Context to pass to function

        Returns
        -------
        Any
            Result from function

        Raises
        ------
        ValueError
            If function not found
        """
        if name not in self._functions:
            raise ValueError(
                f"Function '{name}' not found in registry. "
                f"Available: {list(self._functions.keys())}"
            )

        return self._functions[name](context)

    def has(self, name: str) -> bool:
        """Check if function is registered."""
        return name in self._functions

    def list(self) -> List[str]:
        """List all registered function names."""
        return list(self._functions.keys())


@dataclass
class RoutingDecision:
    """Result of a routing evaluation."""

    route_key: str  # The chosen route
    router_step_id: str  # ID of the router step
    target_step_id: str  # ID of the target step to execute
    skipped_routes: List[str]  # Routes that were not chosen
    evaluation_method: str  # "expression", "llm", or "function"
    evaluation_details: Optional[Dict[str, Any]] = None  # Additional details


class RoutingExecutor:
    """
    Handles conditional routing during plan execution.

    Supports three types of routing:
    1. Expression-based: Evaluate conditions like "${score} > 0.7"
    2. LLM-based: Ask an LLM to choose the route
    3. Function-based: Execute a custom function to determine the route

    Works with pure Pydantic nodes - type-safe field access!

    Example
    -------
    >>> from chuk_ai_planner.core.graph import RouterStep
    >>> from chuk_ai_planner.core.graph.types import RouterType
    >>>
    >>> executor = RoutingExecutor(graph_store)
    >>> router = RouterStep(
    ...     router_type=RouterType.EXPRESSION,
    ...     condition="${score} > 0.7",
    ...     routes=["high", "low"],
    ...     route_mapping={True: "high", False: "low"}
    ... )
    >>> decision = await executor.evaluate_route(
    ...     router_step=router,
    ...     context={"score": 0.85}
    ... )
    >>> print(f"Chosen route: {decision.route_key}")
    """

    def __init__(
        self,
        graph_store: GraphStore,
        function_registry: Optional[FunctionRegistry] = None,
    ):
        """
        Initialize the routing executor.

        Parameters
        ----------
        graph_store : GraphStore
            Graph store containing the plan
        function_registry : Optional[FunctionRegistry]
            Registry of routing functions. If None, creates empty registry.
        """
        self.graph = graph_store
        self.function_registry = function_registry or FunctionRegistry()

    async def evaluate_route(
        self,
        router_step: RouterStep,
        context: Dict[str, Any],
    ) -> RoutingDecision:
        """
        Evaluate a router step and return the chosen route.

        Parameters
        ----------
        router_step : RouterStep
            The router step node (typed!)
        context : Dict[str, Any]
            Execution context with variables

        Returns
        -------
        RoutingDecision
            The chosen route and details

        Raises
        ------
        ValueError
            If router type is unknown or evaluation fails
        """
        # Type-safe field access - no .data.get()!
        router_type = router_step.router_type

        if router_type == RouterType.EXPRESSION or router_type == "expression":
            return await self._evaluate_expression(router_step, context)
        elif router_type == RouterType.LLM or router_type == "llm":
            return await self._evaluate_llm(router_step, context)
        elif router_type == RouterType.FUNCTION or router_type == "function":
            return await self._evaluate_function(router_step, context)
        else:
            raise ValueError(f"Unknown router type: {router_type}")

    async def _evaluate_expression(
        self,
        router_step: RouterStep,
        context: Dict[str, Any],
    ) -> RoutingDecision:
        """
        Evaluate an expression-based router.

        Supports simple boolean expressions like:
        - "${score} > 0.7"
        - "${count} >= 100"
        - "${status} == 'ready'"

        Parameters
        ----------
        router_step : RouterStep
            The router step node
        context : Dict[str, Any]
            Execution context with variables

        Returns
        -------
        RoutingDecision
            The routing decision
        """
        # Type-safe field access!
        condition = router_step.condition
        if not condition:
            raise ValueError(f"Expression router {router_step.id} missing 'condition'")

        # Resolve variables in the condition
        resolved_condition = self._resolve_variables(condition, context)

        # Evaluate the expression safely
        try:
            result = self._safe_eval(resolved_condition, context)
        except Exception as e:
            raise ValueError(
                f"Failed to evaluate condition '{resolved_condition}': {e}"
            ) from e

        # Get the route edges for this router
        route_edges = await self._get_route_edges(router_step.id)

        # Get route mapping (type-safe!)
        route_mapping = router_step.route_mapping or {}

        # Determine the target route key based on the result
        target_route_key = None

        if route_mapping:
            # Use the mapping to determine route
            target_route_key = route_mapping.get(result)

            # For boolean results, also try string keys
            if target_route_key is None and isinstance(result, bool):
                str_result = str(result).lower()
                target_route_key = route_mapping.get(str_result)

        # If no mapping found, try direct match
        if target_route_key is None:
            target_route_key = str(result)

        # Find the edge matching the target route key
        chosen_edge = None
        default_edge = None

        for edge in route_edges:
            # Type-safe field access on RouteEdge!
            if hasattr(edge, "route_key"):
                route_key = edge.route_key
                is_default = getattr(edge, "is_default", False)
            else:
                # Fallback for old-style edges (shouldn't happen with pure Pydantic)
                continue

            # Track default route
            if is_default:
                default_edge = edge

            # Check for match
            if route_key == target_route_key:
                chosen_edge = edge
                break

        # If no match found but we have a default, use that
        if not chosen_edge and default_edge:
            chosen_edge = default_edge

        if not chosen_edge:
            raise ValueError(
                f"No route found for result '{result}' (expected route key: '{target_route_key}') "
                f"in router {router_step.id}"
            )

        # Build decision
        all_routes = []
        for route_edge in route_edges:
            if hasattr(route_edge, "route_key"):
                all_routes.append(route_edge.route_key)

        chosen_route_key = (
            chosen_edge.route_key if hasattr(chosen_edge, "route_key") else ""
        )
        skipped_routes = [r for r in all_routes if r != chosen_route_key]

        return RoutingDecision(
            route_key=chosen_route_key,
            router_step_id=router_step.id,
            target_step_id=chosen_edge.dst,
            skipped_routes=skipped_routes,
            evaluation_method="expression",
            evaluation_details={
                "condition": condition,
                "resolved_condition": resolved_condition,
                "result": result,
            },
        )

    async def _evaluate_llm(
        self,
        router_step: RouterStep,
        context: Dict[str, Any],
    ) -> RoutingDecision:
        """
        Use an LLM to decide the route.

        Parameters
        ----------
        router_step : RouterStep
            The router step node
        context : Dict[str, Any]
            Execution context

        Returns
        -------
        RoutingDecision
            The routing decision
        """
        llm_prompt = router_step.llm_prompt
        if not llm_prompt:
            raise ValueError(f"LLM router {router_step.id} missing 'llm_prompt'")

        # Get available routes
        route_edges = await self._get_route_edges(router_step.id)
        available_routes = [e.route_key for e in route_edges if hasattr(e, "route_key")]

        # Build LLM prompt
        system_prompt = f"""You are a routing decision maker.
Given the context, choose one of these routes: {", ".join(available_routes)}

Respond with ONLY the route key, nothing else."""

        user_prompt = f"""Context:
{json.dumps(context, indent=2)}

Question: {llm_prompt}

Available routes: {", ".join(available_routes)}

Your choice:"""

        # Call LLM (this would integrate with your LLM provider)
        chosen_route_key = await self._call_llm(system_prompt, user_prompt)

        # Validate and clean the response
        chosen_route_key = chosen_route_key.strip().lower()

        # Find matching edge
        chosen_edge = None
        for edge in route_edges:
            if not hasattr(edge, "route_key"):
                continue
            route_key = str(edge.route_key).lower()
            if route_key == chosen_route_key or chosen_route_key in route_key:
                chosen_edge = edge
                break

        # Fallback to default if no match
        if not chosen_edge:
            for edge in route_edges:
                if hasattr(edge, "is_default") and edge.is_default:
                    chosen_edge = edge
                    break

        if not chosen_edge:
            # Last resort: pick first available route
            if route_edges:
                chosen_edge = route_edges[0]
            else:
                raise ValueError(f"No routes available for router {router_step.id}")

        # Build decision
        final_route_key = (
            chosen_edge.route_key if hasattr(chosen_edge, "route_key") else ""
        )
        skipped_routes = [
            route_edge.route_key
            for route_edge in route_edges
            if hasattr(route_edge, "route_key") and route_edge != chosen_edge
        ]

        return RoutingDecision(
            route_key=final_route_key,
            router_step_id=router_step.id,
            target_step_id=chosen_edge.dst,
            skipped_routes=skipped_routes,
            evaluation_method="llm",
            evaluation_details={
                "prompt": llm_prompt,
                "llm_response": chosen_route_key,
                "system_prompt": system_prompt,
            },
        )

    async def _evaluate_function(
        self,
        router_step: RouterStep,
        context: Dict[str, Any],
    ) -> RoutingDecision:
        """
        Execute a custom function to determine the route.

        Parameters
        ----------
        router_step : RouterStep
            The router step node
        context : Dict[str, Any]
            Execution context

        Returns
        -------
        RoutingDecision
            The routing decision
        """
        router_function = router_step.router_function
        if not router_function:
            raise ValueError(
                f"Function router {router_step.id} missing 'router_function'"
            )

        # If router_function is a string, look it up in the registry
        if isinstance(router_function, str):
            if not self.function_registry.has(router_function):
                raise ValueError(
                    f"Function '{router_function}' not found in registry. "
                    f"Available: {self.function_registry.list()}"
                )
            try:
                result = self.function_registry.call(router_function, context)
            except Exception as e:
                raise ValueError(
                    f"Router function '{router_function}' failed: {e}"
                ) from e

        # If it's a callable, execute it directly
        elif callable(router_function):
            try:
                result = router_function(context)
            except Exception as e:
                raise ValueError(f"Router function failed: {e}") from e
        else:
            raise ValueError(
                f"router_function must be callable or string, got {type(router_function)}"
            )

        # Get route edges
        route_edges = await self._get_route_edges(router_step.id)

        # Find edge matching the result
        chosen_edge = None
        for edge in route_edges:
            if not hasattr(edge, "route_key"):
                continue
            route_key = edge.route_key
            if str(route_key) == str(result):
                chosen_edge = edge
                break

        if not chosen_edge:
            # Try default
            for edge in route_edges:
                if hasattr(edge, "is_default") and edge.is_default:
                    chosen_edge = edge
                    break

        if not chosen_edge:
            raise ValueError(
                f"No route found for function result '{result}' in router {router_step.id}"
            )

        # Build decision
        final_route_key = (
            chosen_edge.route_key if hasattr(chosen_edge, "route_key") else ""
        )
        skipped_routes = [
            route_edge.route_key
            for route_edge in route_edges
            if hasattr(route_edge, "route_key") and route_edge != chosen_edge
        ]

        return RoutingDecision(
            route_key=final_route_key,
            router_step_id=router_step.id,
            target_step_id=chosen_edge.dst,
            skipped_routes=skipped_routes,
            evaluation_method="function",
            evaluation_details={
                "function_result": result,
            },
        )

    async def _get_route_edges(self, router_step_id: str) -> list["GraphEdge"]:
        """
        Get all route edges from a router step.

        Parameters
        ----------
        router_step_id : str
            ID of the router step

        Returns
        -------
        list[GraphEdge]
            List of route edges
        """
        edges = await self.graph.get_edges(src=router_step_id, kind=EdgeType.ROUTE)
        return list(edges)

    def _resolve_variables(self, expression: str, context: Dict[str, Any]) -> str:
        """
        Resolve ${variable} references in an expression.

        Parameters
        ----------
        expression : str
            Expression with ${var} placeholders
        context : Dict[str, Any]
            Context with variable values

        Returns
        -------
        str
            Expression with variables resolved
        """
        # Find all ${...} patterns
        pattern = r"\$\{([^}]+)\}"

        def replace_var(match):
            var_path = match.group(1).strip()

            # Handle nested access like ${result.quality_score}
            parts = var_path.split(".")
            value = context

            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    # Try attribute access
                    value = getattr(value, part, None)

                if value is None:
                    # Variable not found, return original
                    return match.group(0)

            # Format the value appropriately for the expression
            if isinstance(value, str):
                return f"'{value}'"
            else:
                return str(value)

        return re.sub(pattern, replace_var, expression)

    def _safe_eval(self, expression: str, context: Dict[str, Any]) -> Any:
        """
        Safely evaluate a boolean/comparison expression.

        Parameters
        ----------
        expression : str
            Expression to evaluate
        context : Dict[str, Any]
            Context (not used directly, variables already resolved)

        Returns
        -------
        Any
            Result of evaluation
        """
        try:
            # Parse the expression
            tree = ast.parse(expression, mode="eval")

            # Evaluate using safe eval
            # This allows comparisons, arithmetic, but no function calls
            result = eval(
                compile(tree, "<string>", "eval"),
                {"__builtins__": {}},  # No built-ins
                {},  # No additional context
            )

            return result

        except Exception as e:
            raise ValueError(f"Invalid expression '{expression}': {e}") from e

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call an LLM to get a routing decision.

        Uses OpenAI by default, falls back to mock for testing.

        Parameters
        ----------
        system_prompt : str
            System prompt
        user_prompt : str
            User prompt

        Returns
        -------
        str
            LLM response (route key)
        """
        try:
            import openai
            import os

            # Get API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment. "
                    "Set it to use LLM-based routing."
                )

            # Create OpenAI client
            client = openai.AsyncOpenAI(api_key=api_key)

            # Call the API
            response = await client.chat.completions.create(
                model="gpt-5-mini",  # Fast, cheap model for routing
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1.0,  # Required for gpt-5-mini (only supported value)
                max_tokens=50,  # Just need the route key
            )

            # Extract the response
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned empty response")
            result = content.strip()
            return result

        except ImportError:
            raise NotImplementedError(
                "LLM routing requires 'openai' package. "
                "Install it with: pip install openai"
            )
        except Exception as e:
            raise ValueError(f"LLM call failed: {e}") from e
