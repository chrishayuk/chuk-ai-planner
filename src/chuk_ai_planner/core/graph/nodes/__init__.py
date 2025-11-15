# chuk_ai_planner/core/graph/nodes/__init__.py
"""
Core graph node types.

All node types in the core graph system, organized by category.
Domain-agnostic - suitable for any planning/execution workflow.

For domain-specific extensions:
- LLM nodes: from chuk_ai_planner.extensions.llm import UserMessage, AssistantMessage
- Future extensions: audio, video, etc.
"""

from chuk_ai_planner.core.graph.nodes.artifact import ArtifactNode
from chuk_ai_planner.core.graph.nodes.base import GraphNode
from chuk_ai_planner.core.graph.nodes.execution import TaskRun, ToolCall
from chuk_ai_planner.core.graph.nodes.job import JobNode, JobRunNode
from chuk_ai_planner.core.graph.nodes.plan import PlanNode, PlanStep, RouterStep
from chuk_ai_planner.core.graph.nodes.session import SessionNode, SummaryNode
from chuk_ai_planner.core.graph.nodes.workflow import ApprovalNode

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
    # Jobs
    "JobNode",
    "JobRunNode",
]
