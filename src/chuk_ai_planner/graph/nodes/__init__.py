# chuk_ai_planner/graph/nodes/__init__.py
"""
Graph node types.

All node types in the graph system, organized by category.
Domain-agnostic - suitable for any planning/execution workflow.

For domain-specific extensions:
- LLM nodes: from chuk_ai_planner.graph.nodes.llm import UserMessage, AssistantMessage
- Projects can add their own: e.g., chuk-motion, chuk-video, etc.
"""

from chuk_ai_planner.graph.nodes.artifact import ArtifactNode
from chuk_ai_planner.graph.nodes.base import GraphNode
from chuk_ai_planner.graph.nodes.execution import TaskRun, ToolCall
from chuk_ai_planner.graph.nodes.plan import PlanNode, PlanStep, RouterStep
from chuk_ai_planner.graph.nodes.session import SessionNode, SummaryNode
from chuk_ai_planner.graph.nodes.workflow import ApprovalNode

__all__ = [
    # Base
    "GraphNode",
    # Planning
    "PlanNode",
    "PlanStep",
    "RouterStep",
    # Execution
    "ToolCall",
    "TaskRun",
    # Session
    "SessionNode",
    "SummaryNode",
    # Workflow
    "ApprovalNode",
    # Artifacts
    "ArtifactNode",
]
