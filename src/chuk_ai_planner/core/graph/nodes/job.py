# chuk_ai_planner/core/graph/nodes/job.py
"""
Job and JobRun nodes - Pydantic native, no dictionary goop.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from .base import GraphNode


class JobNode(GraphNode):
    """A Job node representing a high-level task."""

    kind: Literal["job"] = "job"

    # Core job fields
    description: str
    status: str  # JobStatus enum value

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

    # Current run
    current_run_id: Optional[str] = None

    # Statistics
    run_count: int = 0
    successful_runs: int = 0
    failed_runs: int = 0

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobRunNode(GraphNode):
    """A JobRun node representing a single execution attempt."""

    kind: Literal["job_run"] = "job_run"

    # References
    job_id: str
    plan_id: str
    session_id: str

    # Status
    status: str  # JobRunStatus enum value
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Results
    result_summary: Optional[Dict[str, Any]] = None

    # Execution stats
    steps_completed: int = 0
    steps_total: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
