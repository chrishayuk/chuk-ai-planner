# examples/session_example_usage.py
"""
Demonstration script for managing Accounts → Projects → Sessions,
recording events, runs, and traversing session hierarchies
with the in-memory store.
"""

from uuid import uuid4
from datetime import datetime

# --- Storage providers ---
from a2a_session_manager.storage import InMemorySessionStore, SessionStoreProvider

# --- Account & Project layer ---
from a2a_accounts.models.access_control import AccessControlled
from a2a_accounts.models.project import Project
from a2a_accounts.models.account import Account
from a2a_accounts.models.access_levels import AccessLevel

# --- Session layer ---
from a2a_session_manager.models.session import Session
from a2a_session_manager.models.session_event import SessionEvent
from a2a_session_manager.models.event_type import EventType
from a2a_session_manager.models.event_source import EventSource
from a2a_session_manager.models.session_run import SessionRun, RunStatus

def main():
    # 1) Set up an in-memory session store
    store = InMemorySessionStore()
    SessionStoreProvider.set_store(store)
    print("🗄️  Initialized in-memory session store.")

    # 2) Create an Account
    acct = Account(name="Demo Corp", owner_user_id="alice")
    print(f"👤 Created Account: {acct.id} (owner_user_id={acct.owner_user_id})")

    # 3) Create a Project under that Account
    proj = Project(
        name="Alpha Project",
        account_id=acct.id,
        access_level=AccessLevel.SHARED,
        shared_with={acct.id, "bob"}
    )
    acct.add_project(proj)
    print(f"📁 Created Project: {proj.id} (access_level={proj.access_level.value})")
    print(f"   → Account {acct.id} now owns projects: {acct.project_ids}")

    # 4) Create a root Session under that Project
    root = Session()
    store.save(root)
    proj.add_session(root)
    print(f"💬 Created root Session: {root.id}")
    print(f"   → Project {proj.id} now has sessions: {proj.session_ids}")

    # 5) Record a couple of events
    e1 = SessionEvent(
        message="Hey, how are you?",
        source=EventSource.USER,
        type=EventType.MESSAGE
    )
    e2 = SessionEvent(
        message="I'm fine, thanks!",
        source=EventSource.LLM,
        type=EventType.MESSAGE
    )
    root.events.extend([e1, e2])
    print(f"   • Recorded {len(root.events)} events; last at {root.last_update_time}")

    # 6) Start and complete a run
    run = SessionRun()
    run.mark_running()
    root.runs.append(run)
    print(f"   • Started run {run.id} at {run.started_at} (status={run.status.value})")

    run.mark_completed()
    print(f"   • Completed run {run.id} at {run.ended_at} (status={run.status.value})")

    # 7) Fork a child Session
    child = Session(parent_id=root.id)
    store.save(child)
    # The model_validator will auto-sync root.child_ids
    print(f"🧒 Created child Session: {child.id}")
    print(f"   → root.child_ids = {root.child_ids}")
    print(f"   → child.ancestors = {[s.id for s in child.ancestors()]}")

    # 8) Check ACL at the project level
    print(f"🔒 Is project public? {proj.is_public}")
    print(f"🔑 Does '{acct.id}' have access? {proj.has_access(acct.id)}")
    print(f"🔑 Does 'eve' have access? {proj.has_access('eve')}")

if __name__ == "__main__":
    main()
