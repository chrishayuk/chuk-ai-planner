"""
Build a *minimal* prompt for the next LLM call from a Session.

Keeps
-----
• The first USER message (task).
• The latest assistant MESSAGE (with its content **always set to None**).
• For that assistant, every TOOL_CALL child as a “tool” role
  – only the call’s *result* dict is included.
• If there are no TOOL_CALL children yet, the latest SUMMARY retry note
  is appended instead.
"""

from __future__ import annotations
import json
from typing import List, Dict

from a2a_session_manager.models.session import Session
from a2a_session_manager.models.event_type import EventType
from a2a_session_manager.models.event_source import EventSource

__all__ = ["build_prompt_from_session"]


def build_prompt_from_session(session: Session) -> List[Dict[str, str]]:
    if not session.events:
        return []

    # -------------------------------------------------- #
    # First USER message (instruction / question)
    # -------------------------------------------------- #
    first_user = next(
        (
            e
            for e in session.events
            if e.type == EventType.MESSAGE and e.source == EventSource.USER
        ),
        None,
    )

    # -------------------------------------------------- #
    # Latest assistant MESSAGE (non-user)
    # -------------------------------------------------- #
    assistant_msg = next(
        (
            ev
            for ev in reversed(session.events)
            if ev.type == EventType.MESSAGE and ev.source != EventSource.USER
        ),
        None,
    )
    if assistant_msg is None:
        # Only the user message exists so far
        return [{"role": "user", "content": first_user.message["content"]}] if first_user else []

    # Children of that assistant
    children = [
        e
        for e in session.events
        if e.metadata.get("parent_event_id") == assistant_msg.id
    ]
    tool_calls = [c for c in children if c.type == EventType.TOOL_CALL]
    summaries  = [c for c in children if c.type == EventType.SUMMARY]

    # -------------------------------------------------- #
    # Assemble prompt
    # -------------------------------------------------- #
    prompt: List[Dict[str, str]] = []
    if first_user:
        prompt.append({"role": "user", "content": first_user.message["content"]})

    # ALWAYS add the assistant marker – but strip its free text
    prompt.append({"role": "assistant", "content": None})

    if tool_calls:
        for tc in tool_calls:
            prompt.append(
                {
                    "role": "tool",
                    "name": tc.message["tool"],
                    "content": json.dumps(tc.message["result"], default=str),
                }
            )
    elif summaries:
        prompt.append({"role": "system", "content": summaries[-1].message["note"]})

    return prompt
