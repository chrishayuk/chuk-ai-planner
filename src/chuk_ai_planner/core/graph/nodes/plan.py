# chuk_ai_planner/graph/nodes/plan.py
"""
Planning node types.

Contains all node types related to plan structure and routing.
Pure Pydantic - every field is typed and validated.
"""

from typing import Any, Literal, Optional

from pydantic import Field, field_validator

from chuk_ai_planner.core.graph.nodes.base import GraphNode
from chuk_ai_planner.core.graph.types import (
    NodeType,
    RouterType,
    StepStatus,
    ReliabilityProfile,
)

__all__ = ["PlanNode", "PlanStep", "RouterStep"]


class PlanNode(GraphNode):
    """
    A plan - top-level container for a workflow.

    Represents a complete plan with steps, routing, and execution logic.
    Plans are DAGs where nodes are steps and edges are dependencies/routes.
    """

    kind: Literal[NodeType.PLAN] = NodeType.PLAN

    # Core plan fields
    title: str
    description: Optional[str] = None

    # Runtime variables available during execution
    variables: dict[str, Any] = Field(default_factory=dict)

    # Organizational metadata
    tags: list[str] = Field(default_factory=list)

    # Plan tokens (design system for workflows)
    concurrency_level: int = 1  # Max parallel steps
    max_total_cost: Optional[float] = None  # Total cost budget
    target_latency: Optional[float] = None  # Target completion time (seconds)
    reliability_profile: ReliabilityProfile = ReliabilityProfile.BALANCED

    # Versioning & A/B testing
    version: str = "1.0.0"
    parent_version: Optional[str] = None  # For tracking plan evolution

    # Simulation mode
    is_simulation: bool = False  # Dry-run mode with fake tools


class PlanStep(GraphNode):
    """
    A single executable step in a plan.

    Represents an atomic unit of work that can be executed.
    Steps are connected via edges to form the execution graph.
    """

    kind: Literal[NodeType.PLAN_STEP] = NodeType.PLAN_STEP

    # What this step does
    description: str

    # Hierarchical index like "1", "1.1", "1.2.3"
    index: Optional[str] = None

    # Execution status
    status: StepStatus = StepStatus.PENDING

    # Where to store the result (variable name)
    result_variable: Optional[str] = None

    # Error handling & resilience
    max_retries: int = 0  # Number of retries on failure
    retry_delay_seconds: Optional[float] = None  # Delay between retries
    fallback_step_id: Optional[str] = None  # Step to run on failure
    timeout_seconds: Optional[int] = None  # Max execution time

    # Artifact dependencies
    input_artifacts: list[str] = Field(default_factory=list)  # Artifact IDs needed
    output_artifacts: list[str] = Field(default_factory=list)  # Artifact IDs produced

    # Cost & performance tracking
    max_cost: Optional[float] = None  # Max allowed cost
    estimated_cost: Optional[float] = None  # Estimated cost
    estimated_duration: Optional[float] = None  # Estimated seconds
    actual_cost: Optional[float] = None  # Actual cost (after execution)
    actual_duration: Optional[float] = None  # Actual seconds (after execution)


class RouterStep(GraphNode):
    """
    A routing decision point in the execution graph.

    Evaluates a condition and chooses which path to take.
    Supports three routing strategies: expression, LLM, or function.

    Examples
    --------
    Expression-based routing:

    >>> router = RouterStep(
    ...     router_type=RouterType.EXPRESSION,
    ...     routes=["high", "low"],
    ...     description="Route based on quality",
    ...     condition="${quality_score} > 0.7",
    ...     route_mapping={True: "high", False: "low"}
    ... )

    LLM-based routing:

    >>> router = RouterStep(
    ...     router_type=RouterType.LLM,
    ...     routes=["technical", "creative", "balanced"],
    ...     description="Choose writing style",
    ...     llm_prompt="What writing style best fits the content?"
    ... )
    """

    kind: Literal[NodeType.ROUTER_STEP] = NodeType.ROUTER_STEP

    # How to evaluate the route
    router_type: RouterType

    # Available route keys
    routes: list[str]

    # Human-readable description
    description: str

    # Type-specific fields (set based on router_type)

    # For EXPRESSION routing
    condition: Optional[str] = None  # e.g., "${score} > 0.7"
    route_mapping: Optional[dict[Any, str]] = None  # e.g., {True: "high", False: "low"}

    # For LLM routing
    llm_prompt: Optional[str] = None  # Question to ask the LLM

    # For FUNCTION routing
    router_function: Optional[str] = None  # Function name/reference

    @field_validator("routes")
    @classmethod
    def validate_routes(cls, v: list[str]) -> list[str]:
        """Routers must have at least 2 routes."""
        if len(v) < 2:
            raise ValueError("Router must have at least 2 routes")
        return v

    @field_validator("route_mapping")
    @classmethod
    def validate_route_mapping(
        cls, v: Optional[dict[Any, str]]
    ) -> Optional[dict[Any, str]]:
        """Route mapping values must be in routes list."""
        # Note: routes haven't been validated yet in this order
        # We'll do a runtime check in the router executor instead
        return v
