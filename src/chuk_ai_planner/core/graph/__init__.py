# chuk_ai_planner/graph/__init__.py
"""
Pure Pydantic graph system.

Type-safe, immutable graph nodes and edges with no dictionary goop.
All fields are explicitly typed and validated.

Quick Start
-----------
>>> from chuk_ai_planner.core.graph import PlanNode, PlanStep, RouteEdge
>>> from chuk_ai_planner.core.graph.types import RouterType, StepStatus
>>>
>>> # Create a plan
>>> plan = PlanNode(title="My Plan", description="A clean, typed plan")
>>>
>>> # Create a step
>>> step = PlanStep(description="First step", index="1")
>>>
>>> # All fields are typed and validated!
>>> assert isinstance(step.status, StepStatus)
>>> assert step.status == StepStatus.PENDING

Domain Extensions
-----------------
The core graph is domain-agnostic. For domain-specific nodes:

>>> # LLM-specific nodes (UserMessage, AssistantMessage, SystemMessage)
>>> from chuk_ai_planner.core.graph.nodes.llm import UserMessage, AssistantMessage
>>>
>>> # Projects can create their own extensions (e.g., chuk-motion)
>>> # from chuk_motion.graph.nodes import VideoNode, AudioNode
"""

# Re-export types
from chuk_ai_planner.core.graph.types import (
    ApprovalStatus,
    EdgeType,
    NodeType,
    ReliabilityProfile,
    RouterType,
    StepStatus,
    TaskStatus,
)

# Re-export all nodes
from chuk_ai_planner.core.graph.nodes import (
    ApprovalNode,
    ArtifactNode,
    GraphNode,
    JobNode,
    JobRunNode,
    PlanNode,
    PlanStep,
    RouterStep,
    SessionNode,
    SummaryNode,
    TaskRun,
    ToolCall,
)

# Re-export all edges
from chuk_ai_planner.core.graph.edges import (
    ApprovalEdge,
    ArtifactDependencyEdge,
    CustomEdge,
    FallbackEdge,
    GraphEdge,
    NextEdge,
    ParentChildEdge,
    PlanLinkEdge,
    RouteEdge,
    StepEdge,
)

__all__ = [
    # Types
    "NodeType",
    "EdgeType",
    "RouterType",
    "StepStatus",
    "TaskStatus",
    "ApprovalStatus",
    "ReliabilityProfile",
    # Nodes
    "GraphNode",
    "PlanNode",
    "PlanStep",
    "RouterStep",
    "ToolCall",
    "TaskRun",
    "SessionNode",
    "SummaryNode",
    "ApprovalNode",
    "ArtifactNode",
    "JobNode",
    "JobRunNode",
    # Edges
    "GraphEdge",
    "ParentChildEdge",
    "PlanLinkEdge",
    "StepEdge",
    "RouteEdge",
    "NextEdge",
    "CustomEdge",
    "ApprovalEdge",
    "FallbackEdge",
    "ArtifactDependencyEdge",
]
