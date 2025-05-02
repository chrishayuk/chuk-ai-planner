# src/chuk_ai_planner/processor.py

"""
Graph-Aware Tool Processor (refactored to use GraphNodeManager and PlanExecutor)
"""

import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Callable, Optional
from uuid import uuid4

from chuk_ai_planner.graph.node_manager import GraphNodeManager
from chuk_ai_planner.planner.plan_executor import PlanExecutor
from chuk_ai_planner.models import (
    NodeKind,
    AssistantMessage,
    ToolCall,
    TaskRun,
    Summary
)
from chuk_ai_planner.models.edges import EdgeKind, ParentChildEdge
from a2a_session_manager.storage import SessionStoreProvider
from a2a_session_manager.models.session_event import SessionEvent
from a2a_session_manager.models.event_type import EventType
from a2a_session_manager.models.event_source import EventSource
from a2a_session_manager.models.session_run import SessionRun
from chuk_tool_processor.models.tool_result import ToolResult
from .store.base import GraphStore

_log = logging.getLogger(__name__)


class GraphAwareToolProcessor:
    """
    Tool processor using GraphNodeManager for CRUD and PlanExecutor for plan execution.
    """
    def __init__(
        self,
        session_id: str,
        graph_store: GraphStore,
        *,
        max_llm_retries: int = 2,
        llm_retry_prompt: Optional[str] = None,
        enable_caching: bool = True,
        enable_retries: bool = True
    ):
        self.session_id = session_id
        self.graph_store = graph_store
        self.node_mgr = GraphNodeManager(graph_store)
        self.plan_executor = PlanExecutor(graph_store)
        self.max_llm_retries = max_llm_retries
        self.llm_retry_prompt = (
            llm_retry_prompt or
            "Previous response contained no valid `tool_call`. "
            "Return ONLY a JSON block invoking one of the declared tools."
        )
        self.enable_caching = enable_caching
        self.enable_retries = enable_retries
        self.tool_registry: Dict[str, Callable] = {}
        self._cache: Dict[str, Any] = {}

        # detect an appropriate error event type
        self._error_event_type = next(
            (et for et in EventType if et.name in ('ERROR', 'FAILURE', 'EXCEPTION')),
            EventType.MESSAGE
        )

    def register_tool(self, name: str, fn: Callable):
        """Register a tool function for use in processing."""
        self.tool_registry[name] = fn

    async def process_llm_message(
        self,
        assistant_msg: Dict[str, Any],
        llm_call_fn: Callable[[str], Any],
        assistant_node_id: Optional[str] = None
    ) -> List[ToolResult]:
        """
        Process tool calls from an LLM message, with retry logic,
        and record both session events and graph nodes.
        """
        store = SessionStoreProvider.get_store()
        session = store.get(self.session_id)
        if not session:
            raise RuntimeError(f"Session {self.session_id} not found")

        # start a new SessionRun
        run = SessionRun()
        run.mark_running()
        session.runs.append(run)
        store.save(session)

        # record the assistant message event
        evt = SessionEvent(
            message=assistant_msg,
            type=EventType.MESSAGE,
            source=EventSource.SYSTEM
        )
        session.events.append(evt)
        store.save(session)
        parent_id = evt.id

        # update the assistant_message node if provided
        if assistant_node_id:
            self.node_mgr.update_assistant_node(assistant_node_id, assistant_msg)

        # retry loop for missing tool_calls
        attempt = 0
        while True:
            tool_calls = assistant_msg.get('tool_calls', [])
            if tool_calls:
                results: List[ToolResult] = []
                for tc in tool_calls:
                    res = await self._process_single_tool_call(tc, parent_id, assistant_node_id)
                    results.append(res)
                run.mark_completed()
                store.save(session)
                return results

            # no tool_calls found => retry or fail
            if attempt >= self.max_llm_retries:
                run.mark_failed('Max LLM retries exceeded')
                store.save(session)
                self._create_child_event(
                    self._error_event_type,
                    {'error': 'Max LLM retries exceeded'},
                    parent_id
                )
                raise RuntimeError('Max LLM retries exceeded')

            attempt += 1
            self._create_child_event(
                EventType.SUMMARY,
                {'note': 'Retry due to missing tool calls', 'attempt': attempt},
                parent_id
            )
            assistant_msg = await llm_call_fn(self.llm_retry_prompt)

    async def _process_single_tool_call(
        self,
        tool_call: Dict[str, Any],
        parent_event_id: str,
        assistant_node_id: Optional[str]
    ) -> ToolResult:
        """
        Execute one tool call, record session events, update the graph,
        and return a ToolResult.
        """
        fn_data = tool_call.get('function', {})
        tool_name = fn_data.get('name')
        args_json = fn_data.get('arguments', '{}')
        try:
            args = json.loads(args_json)
        except json.JSONDecodeError:
            args = {'raw_text': args_json}
        call_id = tool_call.get('id', uuid4().hex)

        # caching check
        cache_key = None
        if self.enable_caching:
            cache_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                self._create_child_event(
                    EventType.TOOL_CALL,
                    {'tool': tool_name, 'args': args, 'result': cached, 'cached': True},
                    parent_event_id
                )
                if assistant_node_id:
                    tool_node = self.node_mgr.create_tool_call_node(
                        tool_name, args, cached, assistant_node_id, is_cached=True
                    )
                    self.node_mgr.create_task_run_node(tool_node.id, True, 'cached')
                return ToolResult(id=call_id, tool=tool_name, args=args, result=cached)

        # execute the actual tool function
        tool_fn = self.tool_registry.get(tool_name)
        if not tool_fn:
            raise ValueError(f"Unknown tool: {tool_name}")

        try:
            result = await tool_fn(args)
            success, error = True, None
        except Exception as ex:
            result, success, error = None, False, str(ex)

        # cache on success
        if success and cache_key:
            self._cache[cache_key] = result

        # record in session events
        self._create_child_event(
            EventType.TOOL_CALL,
            {'tool': tool_name, 'args': args, 'result': result, 'error': error},
            parent_event_id
        )

        # update graph nodes
        if assistant_node_id:
            tool_node = self.node_mgr.create_tool_call_node(
                tool_name, args, result, assistant_node_id, error=error
            )
            self.node_mgr.create_task_run_node(tool_node.id, success, error)

        return ToolResult(
            id=call_id, tool=tool_name, args=args, result=result, error=error
        )

    def _create_child_event(
        self,
        event_type: EventType,
        message: Dict[str, Any],
        parent_id: str
    ) -> SessionEvent:
        """
        Emit a session event as a child of the given parent_id and persist.
        """
        store = SessionStoreProvider.get_store()
        session = store.get(self.session_id)
        evt = SessionEvent(
            message=message,
            type=event_type,
            source=EventSource.SYSTEM,
            metadata={'parent_event_id': parent_id}
        )
        session.events.append(evt)
        store.save(session)
        return evt
    
    async def process_plan(
            self,
            plan_node_id: str,
            assistant_node_id: str,
            llm_call_fn: Callable[[str], Any],
            *,
            on_step: Callable[[str, List[ToolResult]], bool] | None = None,   # NEW
        ) -> List[ToolResult]:
            """
            Execute a PlanNode.

            Parameters
            ----------
            on_step
                Optional callback run *after each PlanStep*:

                    keep_running = on_step(step_id, tool_results)

                • Return ``False`` to abort remaining steps.
                • Return ``True``/``None`` (or omit the param) to continue.
            """
            store   = SessionStoreProvider.get_store()
            session = store.get(self.session_id)
            if not session:
                raise RuntimeError(f"Session {self.session_id} not found")

            run = SessionRun(); run.mark_running()
            session.runs.append(run); store.save(session)

            parent_evt = SessionEvent(
                message={'plan_id': plan_node_id},
                type=EventType.SUMMARY,
                source=EventSource.SYSTEM,
                metadata={'description': 'Plan execution started'},
            )
            session.events.append(parent_evt); store.save(session)
            parent_id = parent_evt.id

            steps = self.plan_executor.get_plan_steps(plan_node_id)
            if not steps:
                raise ValueError(f"No steps found for plan {plan_node_id}")

            batches      = self.plan_executor.determine_execution_order(steps)
            all_results: List[ToolResult] = []

            # we keep batching, but execute steps sequentially inside the batch
            for batch in batches:
                for step_id in batch:
                    res_list = await self.plan_executor.execute_step(
                        step_id,
                        assistant_node_id,
                        parent_id,
                        self._create_child_event,
                        self._process_single_tool_call,
                    )
                    all_results.extend(res_list)

                    # ── per-step callback ───────────────────────────────
                    if on_step and on_step(step_id, res_list) is False:
                        run.mark_completed(); store.save(session)
                        return all_results
                    # ─────────────────────────────────────────────────────

            run.mark_completed(); store.save(session)
            summary_evt = SessionEvent(
                message={
                    'plan_id': plan_node_id,
                    'steps_executed': len(steps),
                    'tools_executed': len(all_results),
                },
                type=EventType.SUMMARY,
                source=EventSource.SYSTEM,
                metadata={'parent_event_id': parent_id},
            )
            session.events.append(summary_evt); store.save(session)
            return all_results