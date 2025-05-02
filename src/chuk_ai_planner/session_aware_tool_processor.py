# a2a_agent_core/session_aware_tool_processor.py
"""
Session-aware ToolProcessor

• Wraps each end-to-end batch of tool calls in a `SessionRun`
• Creates ONE parent SessionEvent that contains the **assistant’s raw reply**
• Logs retries and each TOOL_CALL as children via `metadata.parent_event_id`
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Any, List

from chuk_tool_processor.core.processor import ToolProcessor
from chuk_tool_processor.models.tool_result import ToolResult

from a2a_session_manager.storage import SessionStoreProvider
from a2a_session_manager.models.session_event import SessionEvent
from a2a_session_manager.models.event_type import EventType
from a2a_session_manager.models.event_source import EventSource
from a2a_session_manager.models.session_run import SessionRun

_log = logging.getLogger(__name__)


class SessionAwareToolProcessor(ToolProcessor):
    def __init__(
        self,
        session_id: str,
        *,
        max_llm_retries: int = 2,
        llm_retry_prompt: str | None = None,
        **toolproc_kwargs,
    ):
        """
        Parameters
        ----------
        session_id
            The session into which events and runs are written
        max_llm_retries
            How many times to re-prompt the LLM if no tool_call is returned
        llm_retry_prompt
            Prompt appended on each retry (model receives only this text)
        **toolproc_kwargs
            Passed straight through to the ToolProcessor base-class
        """
        super().__init__(**toolproc_kwargs)
        self.session_id = session_id
        self.max_llm_retries = max_llm_retries
        self.llm_retry_prompt = (
            llm_retry_prompt
            or "Previous response contained no valid `tool_call`.\n"
               "Return ONLY a JSON block invoking one of the declared tools."
        )

    # ────────────────────────────────────────────────────────────
    async def process_llm_message(
        self,
        assistant_msg: dict,
        llm_call_fn: Callable[[str], Any],  # async: retry_prompt → assistant_msg dict
    ) -> List[ToolResult]:
        """
        Parse & execute the assistant message.  Retries the LLM until a
        parsable tool_call appears or `max_llm_retries` is reached.

        Child events are threaded under a **parent event** that stores the
        original assistant reply.
        """
        store = SessionStoreProvider.get_store()
        sess = store.get(self.session_id)
        if not sess:
            raise RuntimeError(f"Session {self.session_id!r} not found")

        # 1) SessionRun envelope
        run = SessionRun()
        run.mark_running()
        sess.runs.append(run)
        store.save(sess)

        # 2) Parent event = raw assistant reply
        parent_evt = SessionEvent(
            message=assistant_msg,
            type=EventType.MESSAGE,
            source=EventSource.SYSTEM,
        )
        sess.events.append(parent_evt)
        store.save(sess)
        parent_id = parent_evt.id

        attempt = 0
        blob = json.dumps(assistant_msg)

        while True:
            # try to parse + execute
            results = await self.process_text(blob)
            if results:                                   # success
                run.mark_completed()
                store.save(sess)
                for res in results:
                    self._child_event(EventType.TOOL_CALL,
                                      res.model_dump(),
                                      parent_id)
                return results

            # no tool_calls → retry or fail
            if attempt >= self.max_llm_retries:
                run.mark_failed()
                store.save(sess)
                self._child_event(EventType.MESSAGE,
                                  {"error": "Max LLM retries exceeded"},
                                  parent_id)
                raise RuntimeError("Max LLM retries exceeded")

            attempt += 1
            _log.info("Retrying LLM for valid tool_call (attempt %d)", attempt)
            self._child_event(EventType.SUMMARY,
                              {"note": "Retry due to unparsable tool_call",
                               "attempt": attempt},
                              parent_id)

            assistant_msg = await llm_call_fn(self.llm_retry_prompt)
            blob = json.dumps(assistant_msg)

    # ───────────────────────── helpers ──────────────────────────
    def _child_event(
        self,
        etype: EventType,
        message: Any,
        parent_id: str,
    ) -> None:
        """Append a child SessionEvent linked to *parent_id*."""
        store = SessionStoreProvider.get_store()
        sess = store.get(self.session_id)
        if not sess:
            _log.warning("Session %s disappeared while logging", self.session_id)
            return
        sess.events.append(
            SessionEvent(
                message=message,
                type=etype,
                source=EventSource.SYSTEM,
                metadata={"parent_event_id": parent_id},
            )
        )
        store.save(sess)
