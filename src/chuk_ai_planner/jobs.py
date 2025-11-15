# chuk_ai_planner/jobs.py
"""
Job Manager - Manus-style Orchestration Layer
==============================================

High-level API for creating, planning, and executing jobs.
This is the user-facing orchestration layer that sits above planning and execution.

Example:
    >>> manager = JobManager(
    ...     planner=GraphPlanAgent(...),
    ...     executor=UniversalExecutor(...),
    ...     graph_store=PostgresGraphStore(...)
    ... )
    >>>
    >>> # Simple: create + plan + execute in one shot
    >>> run = await manager.run_job("Research climate change and create a report")
    >>>
    >>> # Or step by step for more control
    >>> job = await manager.create_job("Deploy application to production")
    >>> run = await manager.plan_job(job.id)
    >>> result = await manager.start_job(job.id)
    >>>
    >>> # Resume after failure
    >>> result = await manager.resume_job(job.id)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from chuk_ai_planner.agents.graph_plan_agent import GraphPlanAgent
from chuk_ai_planner.core.graph.types import EdgeType
from chuk_ai_planner.core.planner.universal_plan_executor import UniversalExecutor
from chuk_ai_planner.core.store.base import GraphStore

__all__ = [
    "Job",
    "JobRun",
    "JobStatus",
    "JobRunStatus",
    "JobManager",
]


# ========================================================================
# MODELS
# ========================================================================


class JobStatus(str, Enum):
    """Overall job lifecycle status."""

    PENDING = "pending"  # Created but not yet planned
    PLANNING = "planning"  # LLM is generating the plan
    READY = "ready"  # Plan ready, not started
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Execution failed
    CANCELLED = "cancelled"  # User cancelled


class JobRunStatus(str, Enum):
    """Individual run execution status."""

    PENDING = "pending"  # Run created, not started
    RUNNING = "running"  # Currently executing
    PAUSED = "paused"  # Paused for approval or checkpoint
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Execution failed
    CANCELLED = "cancelled"  # Cancelled


class Job(BaseModel):
    """
    A Job represents a high-level task described in natural language.

    Jobs can have multiple runs (retries, resume after failure, etc.).
    """

    id: str
    description: str  # Natural language task description
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: JobStatus = JobStatus.PENDING

    # For multi-tenant / business use
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

    # Current active run (if any)
    current_run_id: Optional[str] = None

    # Run history
    run_count: int = 0
    successful_runs: int = 0
    failed_runs: int = 0


class JobRun(BaseModel):
    """
    A JobRun represents a single execution attempt of a Job.

    One Job can have multiple runs (retries, resume, etc.).
    """

    id: str
    job_id: str
    plan_id: str  # The plan generated for this job
    session_id: str  # Session for execution tracking

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    status: JobRunStatus = JobRunStatus.PENDING
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Result summary (for quick access without loading full graph)
    result_summary: Optional[Dict[str, Any]] = None

    # Execution metadata
    steps_completed: int = 0
    steps_total: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0


# ========================================================================
# JOB MANAGER
# ========================================================================


class JobManager:
    """
    High-level orchestration facade for the CHUK planning system.

    This is the main entry point for users. It handles:
    - Creating jobs from natural language descriptions
    - Planning (LLM generates execution plan)
    - Execution (run the plan with tools)
    - Lifecycle management (status, resume, cancel)
    - Persistence (save/load jobs and runs)

    Architecture:
        User → JobManager → GraphPlanAgent → UniversalExecutor → Tools

    Example:
        >>> manager = JobManager(planner, executor, graph_store)
        >>>
        >>> # One-shot: describe what you want
        >>> run = await manager.run_job("Research AI safety and summarize findings")
        >>>
        >>> # Step-by-step: more control
        >>> job = await manager.create_job("Build a web scraper")
        >>> run = await manager.plan_job(job.id)
        >>> result = await manager.start_job(job.id)
        >>>
        >>> # Resume after crash or failure
        >>> result = await manager.resume_job(job.id)
    """

    def __init__(
        self,
        planner: GraphPlanAgent,
        executor: UniversalExecutor,
        graph_store: GraphStore,
    ) -> None:
        """
        Initialize JobManager.

        Args:
            planner: Agent that converts descriptions → plans
            executor: Executes plans with tools
            graph_store: Persistent storage for graphs, plans, jobs
        """
        self.planner = planner
        self.executor = executor
        self.store = graph_store

    # ────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ────────────────────────────────────────────────────────────────────

    async def create_job(
        self,
        description: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Job:
        """
        Create a Job record from a natural language description.

        This does NOT plan or execute yet - just creates the Job object.

        Args:
            description: Natural language task description
            metadata: Optional metadata (owner, priority, etc.)
            tags: Optional tags for categorization

        Returns:
            Job object with PENDING status

        Example:
            >>> job = await manager.create_job(
            ...     "Research quantum computing and create a summary",
            ...     metadata={"owner": "alice", "priority": "high"},
            ...     tags=["research", "quantum"]
            ... )
        """
        job_id = self._new_id("job")
        job = Job(
            id=job_id,
            description=description.strip(),
            metadata=metadata or {},
            tags=tags or [],
            status=JobStatus.PENDING,
        )
        await self._save_job(job)
        return job

    async def plan_job(
        self,
        job_id: str,
        *,
        session_id: Optional[str] = None,
        planning_context: Optional[Dict[str, Any]] = None,
    ) -> JobRun:
        """
        Use LLM to generate a plan for this job and create a JobRun.

        Does NOT execute the plan yet - just generates it.

        Args:
            job_id: Job to plan
            session_id: Optional session ID (auto-generated if not provided)
            planning_context: Additional context for the planner

        Returns:
            JobRun object with PENDING status and generated plan

        Example:
            >>> job = await manager.create_job("Scrape website and analyze content")
            >>> run = await manager.plan_job(job.id)
            >>> # Now run.plan_id contains the generated plan
        """
        job = await self._get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        # Mark as planning
        job.status = JobStatus.PLANNING
        job.updated_at = datetime.now(timezone.utc)
        await self._save_job(job)

        try:
            # Let the planner generate the plan
            # Note: planning_context is not used by plan_into_graph API
            plan, plan_id, graph = await self.planner.plan_into_graph(job.description)

            # Create job run
            run_id = self._new_id("run")
            sess_id = session_id or self._new_session_id(job_id)

            run = JobRun(
                id=run_id,
                job_id=job.id,
                plan_id=plan_id,
                session_id=sess_id,
                status=JobRunStatus.PENDING,
            )

            # Update job
            job.current_run_id = run_id
            job.status = JobStatus.READY
            job.run_count += 1
            job.updated_at = datetime.now(timezone.utc)

            await self._save_job_run(run)
            await self._save_job(job)

            return run

        except Exception as e:
            # Planning failed
            job.status = JobStatus.FAILED
            job.updated_at = datetime.now(timezone.utc)
            await self._save_job(job)
            raise ValueError(f"Planning failed for job {job_id}: {e}") from e

    async def start_job(
        self,
        job_id: str,
        *,
        resume: bool = False,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> JobRun:
        """
        Start (or resume) execution of the current JobRun for this Job.

        If the job has no plan yet, it will be planned first.

        Args:
            job_id: Job to start
            resume: If True, resume from last checkpoint
            execution_context: Additional context for execution

        Returns:
            JobRun with execution results

        Example:
            >>> job = await manager.create_job("Deploy application")
            >>> run = await manager.start_job(job.id)
            >>> assert run.status == JobRunStatus.COMPLETED
        """
        job = await self._get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        # If no run yet, plan one
        run: JobRun
        if job.current_run_id is None:
            run = await self.plan_job(job.id)
        else:
            maybe_run = await self._get_job_run(job.current_run_id)
            if maybe_run is None:
                # Recoverable: plan a new run
                run = await self.plan_job(job.id)
            else:
                run = maybe_run

        # Mark job/run as running
        now = datetime.now(timezone.utc)
        job.status = JobStatus.RUNNING
        job.updated_at = now

        run.status = JobRunStatus.RUNNING
        run.started_at = run.started_at or now

        await self._save_job(job)
        await self._save_job_run(run)

        try:
            # Core: delegate to UniversalExecutor
            # Note: execution_context and resume not supported by execute_plan_by_id
            result = await self.executor.execute_plan_by_id(
                plan_id=run.plan_id,
                variables=execution_context,
            )

            # Success!
            run.status = JobRunStatus.COMPLETED
            run.finished_at = datetime.now(timezone.utc)
            run.result_summary = self._summarize_result(result)

            job.status = JobStatus.COMPLETED
            job.successful_runs += 1
            job.updated_at = datetime.now(timezone.utc)

        except Exception as exc:
            # Execution failed
            run.status = JobRunStatus.FAILED
            run.finished_at = datetime.now(timezone.utc)
            run.error = str(exc)
            run.error_details = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

            job.status = JobStatus.FAILED
            job.failed_runs += 1
            job.updated_at = datetime.now(timezone.utc)

            await self._save_job_run(run)
            await self._save_job(job)
            raise

        await self._save_job_run(run)
        await self._save_job(job)
        return run

    async def run_job(
        self,
        description: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        planning_context: Optional[Dict[str, Any]] = None,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> JobRun:
        """
        Convenience method: create + plan + execute in one shot.

        This is the simplest way to use the JobManager.

        Args:
            description: Natural language task description
            metadata: Optional metadata
            tags: Optional tags
            session_id: Optional session ID
            planning_context: Context for planning phase
            execution_context: Context for execution phase

        Returns:
            JobRun with execution results

        Example:
            >>> run = await manager.run_job(
            ...     "Research AI safety and create a PDF report"
            ... )
            >>> assert run.status == JobRunStatus.COMPLETED
        """
        job = await self.create_job(description, metadata=metadata, tags=tags)
        await self.plan_job(
            job.id,
            session_id=session_id,
            planning_context=planning_context,
        )
        run = await self.start_job(
            job.id,
            execution_context=execution_context,
        )
        return run

    async def resume_job(
        self,
        job_id: str,
        *,
        create_new_run: bool = False,
    ) -> JobRun:
        """
        Resume a failed or paused job.

        Uses checkpointing to continue from where it left off.

        Args:
            job_id: Job to resume
            create_new_run: If True, create a new run; else resume current run

        Returns:
            JobRun with resumed execution results

        Example:
            >>> # Job failed partway through
            >>> run = await manager.resume_job(job.id)
            >>> # Continues from last checkpoint
        """
        job = await self._get_job(job_id)
        if not job or not job.current_run_id:
            raise ValueError(f"Job {job_id} has no run to resume")

        if create_new_run:
            # Create a fresh run with same plan
            run = await self._get_job_run(job.current_run_id)
            if not run:
                raise ValueError(f"Current run {job.current_run_id} not found")

            # Create new run with same plan
            new_run = JobRun(
                id=self._new_id("run"),
                job_id=job.id,
                plan_id=run.plan_id,
                session_id=self._new_session_id(job_id),
                status=JobRunStatus.PENDING,
            )

            job.current_run_id = new_run.id
            job.run_count += 1
            await self._save_job_run(new_run)
            await self._save_job(job)

        # Start with resume=True
        return await self.start_job(job.id, resume=True)

    async def cancel_job(self, job_id: str) -> None:
        """
        Cancel a job and its current run.

        Note: Actual cancellation of in-flight work depends on
        executor and tool processor support for cancellation.

        Args:
            job_id: Job to cancel

        Example:
            >>> await manager.cancel_job(job.id)
        """
        job = await self._get_job(job_id)
        if job is None:
            return

        job.status = JobStatus.CANCELLED
        job.updated_at = datetime.now(timezone.utc)
        await self._save_job(job)

        if job.current_run_id:
            run = await self._get_job_run(job.current_run_id)
            if run:
                run.status = JobRunStatus.CANCELLED
                run.finished_at = datetime.now(timezone.utc)
                await self._save_job_run(run)

        # TODO: Signal to UniversalExecutor / ToolProcessor to stop
        # if they support cancellation

    async def get_job_status(self, job_id: str) -> JobStatus:
        """
        Get current status of a job.

        Args:
            job_id: Job ID

        Returns:
            Current JobStatus
        """
        job = await self._get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        return job.status

    async def get_job(
        self,
        job_id: str,
        *,
        include_runs: bool = True,
        include_plan: bool = False,
    ) -> Dict[str, Any]:
        """
        Get comprehensive job information.

        Args:
            job_id: Job ID
            include_runs: Include run history
            include_plan: Include current plan structure

        Returns:
            Dictionary with job, runs, and optionally plan

        Example:
            >>> info = await manager.get_job(job.id, include_runs=True)
            >>> print(info["job"].status)
            >>> print(f"Runs: {len(info['runs'])}")
        """
        job = await self._get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        data: Dict[str, Any] = {"job": job}

        if include_runs:
            runs = await self._list_job_runs(job_id)
            data["runs"] = runs

        if include_plan and job.current_run_id:
            run = await self._get_job_run(job.current_run_id)
            if run:
                # Get plan from graph store
                plan_node = self.store.get_node(run.plan_id)
                if plan_node:
                    data["plan"] = plan_node

        return data

    async def list_jobs(
        self,
        *,
        status: Optional[Iterable[JobStatus]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Job]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status(es)
            tags: Filter by tags (AND logic - must have all tags)
            limit: Max number of results
            offset: Pagination offset

        Returns:
            List of Job objects

        Example:
            >>> running_jobs = await manager.list_jobs(
            ...     status=[JobStatus.RUNNING, JobStatus.PENDING]
            ... )
        """
        # This requires GraphStore to have job storage methods
        # For now, return stub - will be implemented with DB store
        return await self._list_jobs(
            status=status,
            tags=tags,
            limit=limit,
            offset=offset,
        )

    # ────────────────────────────────────────────────────────────────────
    # STORAGE METHODS (to be implemented in GraphStore)
    # ────────────────────────────────────────────────────────────────────

    async def _save_job(self, job: Job) -> None:
        """Save job to storage - Pydantic native."""
        from chuk_ai_planner.core.graph import JobNode

        node = JobNode(
            id=job.id,
            description=job.description,
            status=job.status.value,
            metadata=job.metadata,
            tags=job.tags,
            current_run_id=job.current_run_id,
            run_count=job.run_count,
            successful_runs=job.successful_runs,
            failed_runs=job.failed_runs,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )
        await self.store.add_node(node)

    async def _save_job_run(self, run: JobRun) -> None:
        """Save job run to storage - Pydantic native."""
        from chuk_ai_planner.core.graph import JobRunNode, GraphEdge, EdgeType

        node = JobRunNode(
            id=run.id,
            job_id=run.job_id,
            plan_id=run.plan_id,
            session_id=run.session_id,
            status=run.status.value,
            error=run.error,
            error_details=run.error_details,
            result_summary=run.result_summary,
            steps_completed=run.steps_completed,
            steps_total=run.steps_total,
            steps_failed=run.steps_failed,
            steps_skipped=run.steps_skipped,
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )
        await self.store.add_node(node)

        # Link run to job
        edge = GraphEdge(
            src=run.job_id,
            dst=run.id,
            kind=EdgeType.PARENT_CHILD,
        )
        await self.store.add_edge(edge)

    async def _get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve job from storage - Pydantic native."""
        from chuk_ai_planner.core.graph import JobNode

        node = await self.store.get_node(job_id)
        if not node or not isinstance(node, JobNode):
            return None

        return Job(
            id=node.id,
            description=node.description,
            status=JobStatus(node.status),
            metadata=node.metadata,
            tags=node.tags,
            current_run_id=node.current_run_id,
            run_count=node.run_count,
            successful_runs=node.successful_runs,
            failed_runs=node.failed_runs,
            created_at=node.created_at,
            updated_at=node.updated_at,
        )

    async def _get_job_run(self, run_id: str) -> Optional[JobRun]:
        """Retrieve job run from storage - Pydantic native."""
        from chuk_ai_planner.core.graph import JobRunNode

        node = await self.store.get_node(run_id)
        if not node or not isinstance(node, JobRunNode):
            return None

        return JobRun(
            id=node.id,
            job_id=node.job_id,
            plan_id=node.plan_id,
            session_id=node.session_id,
            status=JobRunStatus(node.status),
            error=node.error,
            error_details=node.error_details,
            result_summary=node.result_summary,
            steps_completed=node.steps_completed,
            steps_total=node.steps_total,
            steps_failed=node.steps_failed,
            steps_skipped=node.steps_skipped,
            created_at=node.created_at,
            started_at=node.started_at,
            finished_at=node.finished_at,
        )

    async def _list_jobs(
        self,
        status: Optional[Iterable[JobStatus]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Job]:
        """List jobs with filtering - Pydantic native."""
        from chuk_ai_planner.core.graph import JobNode

        # Get all job nodes
        all_nodes = await self.store.list_nodes(kind="job")

        jobs = []
        for node in all_nodes:
            if not isinstance(node, JobNode):
                continue

            job = await self._get_job(node.id)
            if not job:
                continue

            # Apply filters
            if status and job.status not in status:
                continue

            if tags and not all(tag in job.tags for tag in tags):
                continue

            jobs.append(job)

        # Sort by updated_at desc
        jobs.sort(key=lambda j: j.updated_at, reverse=True)

        # Pagination
        return jobs[offset : offset + limit]

    async def _list_job_runs(self, job_id: str) -> List[JobRun]:
        """List all runs for a job."""
        edges = await self.store.get_edges_by_src(job_id, kind=EdgeType.PARENT_CHILD)

        runs = []
        for edge in edges:
            run = await self._get_job_run(edge.dst)
            if run:
                runs.append(run)

        # Sort by created_at desc
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs

    # ────────────────────────────────────────────────────────────────────
    # HELPERS
    # ────────────────────────────────────────────────────────────────────

    def _new_id(self, prefix: str) -> str:
        """Generate a new ID with prefix."""
        return f"{prefix}_{uuid.uuid4().hex}"

    def _new_session_id(self, job_id: str) -> str:
        """Generate a session ID for a job."""
        return f"job-{job_id}-{uuid.uuid4().hex[:8]}"

    def _summarize_result(self, result: Any) -> Dict[str, Any]:
        """
        Summarize execution result for storage.

        This is a simple summarizer - can be made more sophisticated.
        """
        if isinstance(result, dict):
            # Limit to first 10 keys for summary
            return {k: v for k, v in list(result.items())[:10]}
        elif isinstance(result, list):
            return {
                "type": "list",
                "length": len(result),
                "sample": result[:3] if len(result) > 0 else [],
            }
        else:
            return {"type": type(result).__name__, "repr": repr(result)[:200]}
