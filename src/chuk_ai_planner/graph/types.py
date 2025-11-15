# chuk_ai_planner/graph/types.py
"""
Type definitions for the graph system.

All enums and constants used throughout the graph implementation.
No hardcoded strings - everything is typed and validated.
"""

from enum import Enum

__all__ = [
    "NodeType",
    "EdgeType",
    "RouterType",
    "StepStatus",
    "ApprovalStatus",
    "ReliabilityProfile",
]


class NodeType(str, Enum):
    """
    All node types in the graph.

    Domain-agnostic types for planning and execution.
    Each node type represents a different concept in the planning/execution model.
    """
    SESSION = "session"
    PLAN = "plan"
    PLAN_STEP = "plan_step"
    ROUTER_STEP = "router_step"
    TOOL_CALL = "tool_call"
    TASK_RUN = "task_run"
    SUMMARY = "summary"
    APPROVAL = "approval"  # Human-in-the-loop approval gate
    ARTIFACT = "artifact"  # Artifact reference/lineage


class EdgeType(str, Enum):
    """
    All edge types in the graph.

    Edges define relationships and execution flow between nodes.
    """
    PARENT_CHILD = "parent_child"  # Hierarchical relationship
    NEXT = "next"  # Temporal/sequential ordering
    PLAN_LINK = "plan_link"  # Links plan to its steps
    STEP_ORDER = "step_order"  # Ordering between steps
    ROUTE = "route"  # Conditional routing path
    CUSTOM = "custom"  # Extensibility for custom edge types
    APPROVAL = "approval"  # Approval gate flow
    FALLBACK = "fallback"  # Error fallback path
    ARTIFACT_DEPENDENCY = "artifact_dependency"  # Artifact flow between steps


class RouterType(str, Enum):
    """
    Router evaluation strategies.

    Determines how a router step decides which path to take.
    """
    EXPRESSION = "expression"  # Evaluate boolean expression
    LLM = "llm"  # Ask LLM to choose route
    FUNCTION = "function"  # Execute custom function


class StepStatus(str, Enum):
    """
    Execution status for plan steps.

    Tracks the lifecycle of a step during plan execution.
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"  # Blocked by dependencies
    PAUSED = "paused"  # Manually paused
    WAITING_APPROVAL = "waiting_approval"  # Waiting for human approval
    CANCELLED = "cancelled"  # Manually cancelled
    TIMEOUT = "timeout"  # Exceeded time limit
    RETRYING = "retrying"  # In retry backoff


class ApprovalStatus(str, Enum):
    """
    Status of an approval gate.

    Tracks the state of human-in-the-loop approval nodes.
    """
    PENDING = "pending"  # Waiting for approval
    APPROVED = "approved"  # Approved by human
    REJECTED = "rejected"  # Rejected by human
    TIMEOUT = "timeout"  # Approval timeout exceeded
    ESCALATED = "escalated"  # Escalated to different approver


class ReliabilityProfile(str, Enum):
    """
    Plan reliability/safety profiles.

    Determines default retry, timeout, and error handling behavior.
    """
    AGGRESSIVE = "aggressive"  # Fast, minimal retries
    BALANCED = "balanced"  # Moderate retries and timeouts
    ULTRA_SAFE = "ultra_safe"  # Maximum retries, conservative timeouts
