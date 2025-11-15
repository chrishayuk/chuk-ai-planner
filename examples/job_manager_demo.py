"""
Job Manager Demo
================

Demonstrates the high-level JobManager API for creating,
planning, and executing jobs.

This is the simplest way to use chuk-ai-planner.
"""

import asyncio
from chuk_ai_planner.jobs import JobManager, JobStatus
from chuk_ai_planner.agents.graph_plan_agent import GraphPlanAgent
from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor
from chuk_ai_planner.store.memory import InMemoryGraphStore


# ────────────────────────────────────────────────────────────────────
# SETUP
# ────────────────────────────────────────────────────────────────────


async def setup_manager():
    """Set up the job manager with planner and executor."""

    # Create graph store
    graph = InMemoryGraphStore()

    # Create planner (you'd configure this with your tools)
    planner = GraphPlanAgent(
        graph=graph,
        system_prompt="You are a helpful planning assistant.",
        validate_step=lambda step: True,  # Simple validation
        model="gpt-4o-mini",
    )

    # Create executor
    executor = UniversalExecutor(graph_store=graph)

    # Create job manager
    manager = JobManager(
        planner=planner,
        executor=executor,
        graph_store=graph,
    )

    return manager


# ────────────────────────────────────────────────────────────────────
# EXAMPLE 1: ONE-SHOT JOB
# ────────────────────────────────────────────────────────────────────


async def example_one_shot():
    """
    Simplest usage: describe what you want and get it done.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: One-Shot Job")
    print("=" * 70)

    manager = await setup_manager()

    # This is the simplest API - just describe what you want
    run = await manager.run_job(
        "Research climate change adaptation strategies and create a summary"
    )

    print("\nJob completed!")
    print(f"  Status: {run.status}")
    print(f"  Run ID: {run.id}")
    print(f"  Duration: {(run.finished_at - run.started_at).total_seconds():.2f}s")

    return run


# ────────────────────────────────────────────────────────────────────
# EXAMPLE 2: STEP-BY-STEP CONTROL
# ────────────────────────────────────────────────────────────────────


async def example_step_by_step():
    """
    More control: create, plan, and execute separately.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Step-by-Step Job Control")
    print("=" * 70)

    manager = await setup_manager()

    # Step 1: Create the job
    job = await manager.create_job(
        "Deploy application to production",
        metadata={"owner": "alice", "priority": "high"},
        tags=["deployment", "production"],
    )
    print(f"\n1. Job created: {job.id}")
    print(f"   Status: {job.status}")

    # Step 2: Plan the job (LLM generates execution plan)
    run = await manager.plan_job(job.id)
    print(f"\n2. Plan generated: {run.plan_id}")
    print(f"   Run ID: {run.id}")

    # At this point you could:
    # - Review the plan
    # - Modify it
    # - Get approval
    # - Wait for scheduled time

    # Step 3: Execute the plan
    result = await manager.start_job(job.id)
    print("\n3. Execution complete!")
    print(f"   Status: {result.status}")
    print(f"   Steps completed: {result.steps_completed}/{result.steps_total}")

    return result


# ────────────────────────────────────────────────────────────────────
# EXAMPLE 3: RESUME AFTER FAILURE
# ────────────────────────────────────────────────────────────────────


async def example_resume():
    """
    Demonstrate resuming a failed job.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Resume After Failure")
    print("=" * 70)

    manager = await setup_manager()

    # Create and start a job
    job = await manager.create_job(
        "Process large dataset and generate report",
        metadata={"dataset_size": "100GB"},
    )

    try:
        # This might fail partway through
        run = await manager.start_job(job.id)
    except Exception as e:
        print(f"\nJob failed: {e}")

        # Check status
        status = await manager.get_job_status(job.id)
        print(f"Job status: {status}")

        # Resume from checkpoint
        print("\nResuming job...")
        resumed_run = await manager.resume_job(job.id)
        print(f"Resumed! Status: {resumed_run.status}")

        return resumed_run


# ────────────────────────────────────────────────────────────────────
# EXAMPLE 4: JOB MONITORING
# ────────────────────────────────────────────────────────────────────


async def example_monitoring():
    """
    Monitor job status and progress.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Job Monitoring")
    print("=" * 70)

    manager = await setup_manager()

    # Create multiple jobs
    jobs = []
    for i in range(3):
        job = await manager.create_job(
            f"Task {i + 1}: Process batch {i + 1}",
            tags=["batch", f"batch-{i + 1}"],
        )
        jobs.append(job)
        # Start planning (but don't execute yet)
        await manager.plan_job(job.id)

    # List all jobs
    all_jobs = await manager.list_jobs()
    print(f"\nTotal jobs: {len(all_jobs)}")

    # List by status
    ready_jobs = await manager.list_jobs(status=[JobStatus.READY])
    print(f"Ready jobs: {len(ready_jobs)}")

    # Get detailed info for one job
    info = await manager.get_job(
        jobs[0].id,
        include_runs=True,
        include_plan=True,
    )
    print("\nJob Details:")
    print(f"  Description: {info['job'].description}")
    print(f"  Status: {info['job'].status}")
    print(f"  Runs: {len(info.get('runs', []))}")
    if "plan" in info:
        print(f"  Plan: {info['plan'].id}")

    return info


# ────────────────────────────────────────────────────────────────────
# EXAMPLE 5: CANCEL JOB
# ────────────────────────────────────────────────────────────────────


async def example_cancel():
    """
    Demonstrate cancelling a job.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Cancel Job")
    print("=" * 70)

    manager = await setup_manager()

    # Create a job
    job = await manager.create_job(
        "Long running analysis task",
        metadata={"estimated_duration": "2 hours"},
    )

    # Plan it
    await manager.plan_job(job.id)

    print(f"\nJob created: {job.id}")
    print(f"Status: {(await manager.get_job_status(job.id)).value}")

    # Cancel it
    await manager.cancel_job(job.id)
    print("\nJob cancelled!")
    print(f"Status: {(await manager.get_job_status(job.id)).value}")

    return job


# ────────────────────────────────────────────────────────────────────
# EXAMPLE 6: MULTI-TENANT USAGE
# ────────────────────────────────────────────────────────────────────


async def example_multi_tenant():
    """
    Demonstrate using metadata for multi-tenant scenarios.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Multi-Tenant Usage")
    print("=" * 70)

    manager = await setup_manager()

    # Create jobs for different users/teams
    alice_job = await manager.create_job(
        "Analyze Q4 sales data",
        metadata={
            "owner": "alice",
            "team": "analytics",
            "priority": "high",
            "cost_center": "CC-123",
        },
        tags=["sales", "q4", "analytics"],
    )

    bob_job = await manager.create_job(
        "Generate marketing report",
        metadata={
            "owner": "bob",
            "team": "marketing",
            "priority": "medium",
            "cost_center": "CC-456",
        },
        tags=["marketing", "report"],
    )

    print(f"\nAlice's job: {alice_job.id}")
    print(f"  Metadata: {alice_job.metadata}")
    print(f"  Tags: {alice_job.tags}")

    print(f"\nBob's job: {bob_job.id}")
    print(f"  Metadata: {bob_job.metadata}")
    print(f"  Tags: {bob_job.tags}")

    # Filter by team
    analytics_jobs = await manager.list_jobs(tags=["analytics"])
    print(f"\nAnalytics team jobs: {len(analytics_jobs)}")

    return alice_job, bob_job


# ────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────


async def main():
    """Run all examples."""

    print("\n" + "=" * 70)
    print("JOB MANAGER DEMO")
    print("=" * 70)
    print("\nThe JobManager provides a high-level API for orchestrating")
    print("AI-powered workflows. Just describe what you want, and the")
    print("system plans and executes it for you.")
    print("=" * 70)

    # Run examples
    await example_one_shot()
    await example_step_by_step()
    # await example_resume()  # Might fail without proper error simulation
    await example_monitoring()
    await example_cancel()
    await example_multi_tenant()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. JobManager provides Manus-style orchestration")
    print("  2. Natural language → Plan → Execute")
    print("  3. Resume after failures with checkpointing")
    print("  4. Monitor and manage multiple jobs")
    print("  5. Enterprise features (metadata, tags, multi-tenant)")
    print("\nThis is the easiest way to use chuk-ai-planner!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
