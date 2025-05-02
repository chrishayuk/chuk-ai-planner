# a2a_agent_core/examples/openai_tool_demo.py
"""
OpenAI-tools demo with session logging *and* hierarchical event
printing.  The SessionAwareToolProcessor stores:

   • one parent “batch” event
   • retry notices (if any)   → metadata.parent_event_id = <batch-id>
   • TOOL_CALL results        → metadata.parent_event_id = <batch-id>

This script renders that hierarchy in an indented tree.
"""

from __future__ import annotations

# ── quiet logging by default ───────────────────────────────────────
import logging, sys, asyncio, json, os, pprint
logging.basicConfig(level=logging.WARNING,
                    stream=sys.stdout,
                    format="%(levelname)s | %(message)s")
import chuk_tool_processor            # silence its handler
logging.getLogger("chuk_tool_processor").setLevel(logging.WARNING)

# ── OpenAI + env ───────────────────────────────────────────────────
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ── session & processor bits ───────────────────────────────────────
from a2a_session_manager.storage import InMemorySessionStore, SessionStoreProvider
from a2a_session_manager.models.session import Session, SessionEvent
from a2a_session_manager.models.event_type import EventType
from chuk_ai_planner.session_aware_tool_processor import SessionAwareToolProcessor
from chuk_tool_processor.registry.tool_export import openai_functions

# tools self-register
from sample_tools import WeatherTool, SearchTool, CalculatorTool  # noqa: F401

load_dotenv()


async def ensure_openai_ok() -> AsyncOpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in environment or .env")
    client = AsyncOpenAI(api_key=key)
    await client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": "ping"}]
    )
    return client


# ── tiny helper to pretty-print hierarchy ──────────────────────────
def print_event_tree(events: list[SessionEvent]) -> None:
    ids = {evt.id: evt for evt in events}
    children: dict[str, list[SessionEvent]] = {}
    for evt in events:
        parent = evt.metadata.get("parent_event_id")
        if parent:
            children.setdefault(parent, []).append(evt)

    def _dump(evt: SessionEvent, indent: int = 0):
        pad = "  " * indent
        print(f"{pad}• {evt.type.value:10}  id={evt.id}")
        if evt.type == EventType.TOOL_CALL:
            msg = evt.message or {}
            print(f"{pad}  ⇒ {msg.get('tool')}   "
                  f"error={msg.get('error')}")
        for ch in children.get(evt.id, []):
            _dump(ch, indent + 1)

    # roots = events without parent_event_id
    roots = [e for e in events if not e.metadata.get("parent_event_id")]
    for root in roots:
        _dump(root)


# ── main flow ───────────────────────────────────────────────────────
async def main() -> None:
    client = await ensure_openai_ok()

    # session & in-memory store
    SessionStoreProvider.set_store(InMemorySessionStore())
    session = Session(); SessionStoreProvider.get_store().save(session)

    processor = SessionAwareToolProcessor(session_id=session.id,
                                          enable_caching=True,
                                          enable_retries=True)

    prompt = (
        "I need to know if I should wear a jacket today in New York.\n"
        "Also, how much is 235.5 × 18.75?\n"
        "Finally, find a couple of pages on climate-change adaptation."
    )

    async def ask_llm(prompt_text: str):
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_text}],
            tools=openai_functions(),
            tool_choice="auto",
            temperature=0.7,
        )
        return resp.choices[0].message.model_dump()

    assistant_msg = await ask_llm(prompt)
    results = await processor.process_llm_message(assistant_msg, ask_llm)

    # ── show tool results ───────────────────────────────────────────
    print(f"\nExecuted {len(results)} tool calls")
    for r in results:
        print(f"\n⮑  {r.tool}")
        pprint.pp(r.result.model_dump())

    # ── show hierarchical events ───────────────────────────────────
    print("\nSession event tree:")
    print_event_tree(session.events)


if __name__ == "__main__":
    asyncio.run(main())
