"""
Demonstrate SessionAwareToolProcessor retry logic, event hierarchy,
and prompt-pruning with build_prompt_from_session().
"""

from __future__ import annotations
import asyncio, json, logging, pprint
from typing import Dict

# ── quiet logging ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

# ── sample tool registers itself ────────────────────────────────────────
from sample_tools import WeatherTool  # noqa: F401

# ── A2A + processor imports ────────────────────────────────────────────
from a2a_session_manager.storage import InMemorySessionStore, SessionStoreProvider
from a2a_session_manager.models.session import Session, SessionEvent
from a2a_session_manager.models.event_type import EventType
from a2a_session_manager.models.event_source import EventSource
from chuk_ai_planner.session_aware_tool_processor import SessionAwareToolProcessor
from chuk_ai_planner.prompt_builder import build_prompt_from_session

# -----------------------------------------------------------------------


async def main() -> None:
    # 1) Session setup
    store = InMemorySessionStore()
    SessionStoreProvider.set_store(store)
    session = Session()
    store.save(session)

    user_prompt = "Tell me the weather in London."
    session.events.append(
        SessionEvent(
            message={"content": user_prompt},
            type=EventType.MESSAGE,
            source=EventSource.USER,
        )
    )
    store.save(session)

    # 2) Processor with single retry
    processor = SessionAwareToolProcessor(
        session_id=session.id,
        enable_caching=False,
        enable_retries=True,
        max_llm_retries=1,
    )

    # 3) Fake LLM: first reply bad, second good
    attempts = 0

    async def fake_llm(_prompt: str) -> Dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:  # invalid
            return {"role": "assistant", "content": "Weather is nice!", "tool_calls": []}
        return {           # valid
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": '{"location": "London"}',
                }}],
        }

    assistant_reply = await fake_llm("first call")
    results = await processor.process_llm_message(assistant_reply, fake_llm)

    # 4) Results
    print("\nTool execution results:")
    for r in results:
        print(json.dumps(r.result.model_dump(), indent=2))

    # 5) Event hierarchy
    print("\nHierarchical Session Events:")
    def tree(evt: SessionEvent, depth=0):
        pad = "  " * depth
        print(f"{pad}• {evt.type.value:9} id={evt.id}")
        for ch in [e for e in session.events
                   if e.metadata.get("parent_event_id") == evt.id]:
            tree(ch, depth + 1)
    for root in [e for e in session.events if "parent_event_id" not in e.metadata]:
        tree(root)

    # 6) Next-turn prompt
    next_prompt = build_prompt_from_session(session)
    print("\nNext-turn prompt that will be sent to the LLM:")
    pprint.pp(next_prompt)


if __name__ == "__main__":
    asyncio.run(main())
