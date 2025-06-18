# chuk_ai_planner/processor.py
"""
Graph-Aware Tool Processor
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
from chuk_ai_session_manager.session_storage import setup_chuk_sessions_storage
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.session_run import SessionRun
from chuk_tool_processor.models.tool_result import ToolResult
from .store.base import GraphStore

_log = logging.getLogger(__name__)


class GraphAwareToolProcessor:
    """
    Tool processor using GraphNodeManager for CRUD and PlanExecutor for plan execution.
    Updated to work with the latest chuk_tool_processor.
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
        self._executor = None

        # detect an appropriate error event type
        self._error_event_type = next(
            (et for et in EventType if et.name in ('ERROR', 'FAILURE', 'EXCEPTION')),
            EventType.MESSAGE
        )

    async def _get_tool_executor(self):
        """Get or initialize a tool executor with the current registry."""
        if not hasattr(self, '_executor') or self._executor is None:
            try:
                # Try to import and use the new tool processor
                from chuk_tool_processor.registry import get_default_registry
                from chuk_tool_processor.execution.strategies.inprocess_strategy import InProcessStrategy
                from chuk_tool_processor.execution.tool_executor import ToolExecutor
                
                registry = await get_default_registry()
                strategy = InProcessStrategy(registry)
                self._executor = ToolExecutor(registry=registry, strategy=strategy)
            except ImportError:
                self._executor = None
        
        return self._executor

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
        # Create a session for this operation
        session = await Session.create()
        
        # start a new SessionRun
        run = SessionRun()
        run.mark_running()
        session.add_run(run)

        # record the assistant message event
        evt = SessionEvent(
            message=assistant_msg,
            type=EventType.MESSAGE,
            source=EventSource.SYSTEM
        )
        session.add_event(evt)
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
                return results

            # no tool_calls found => retry or fail
            if attempt >= self.max_llm_retries:
                run.mark_failed('Max LLM retries exceeded')
                self._create_child_event(
                    session,
                    self._error_event_type,
                    {'error': 'Max LLM retries exceeded'},
                    parent_id
                )
                raise RuntimeError('Max LLM retries exceeded')

            attempt += 1
            self._create_child_event(
                session,
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

        # First, try to use the new tool executor
        executor = await self._get_tool_executor()
        if executor is not None:
            try:
                from chuk_tool_processor.models.tool_call import ToolCall as ProcessorToolCall
                
                # Create a ToolCall for the executor
                tc = ProcessorToolCall(
                    id=call_id,
                    tool=tool_name,
                    arguments=args
                )
                
                # Execute the tool
                results = await executor.execute([tc])
                if results and len(results) > 0:
                    result = results[0].result
                    error = results[0].error
                    success = not bool(error)
                    
                    # Cache on success
                    if success and cache_key:
                        self._cache[cache_key] = result
                    
                    # Update graph nodes
                    if assistant_node_id:
                        tool_node = self.node_mgr.create_tool_call_node(
                            tool_name, args, result, assistant_node_id, error=error
                        )
                        self.node_mgr.create_task_run_node(tool_node.id, success, error)
                    
                    return ToolResult(
                        id=call_id, tool=tool_name, args=args, result=result, error=error
                    )
            except Exception as ex:
                _log.warning(f"Error using tool executor: {ex}. Falling back to registered tool.")
                # Fall through to the registered tool approach
        
        # Fall back to using the registered tool
        tool_fn = self.tool_registry.get(tool_name)
        if not tool_fn:
            raise ValueError(f"Unknown tool: {tool_name}")

        try:
            result = await tool_fn(args)
            success, error = True, None
        except Exception as ex:
            result, success, error = None, False, str(ex)

        # Cache on success
        if success and cache_key:
            self._cache[cache_key] = result

        # Update graph nodes
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
        session: Session,
        event_type: EventType,
        message: Dict[str, Any],
        parent_id: str
    ) -> SessionEvent:
        """
        Emit a session event as a child of the given parent_id.
        """
        evt = SessionEvent(
            message=message,
            type=event_type,
            source=EventSource.SYSTEM,
            metadata={'parent_event_id': parent_id}
        )
        session.add_event(evt)
        return evt
    
    async def process_plan(
        self,
        plan_node_id: str,
        assistant_node_id: str,
        llm_call_fn: Callable[[str], Any],
        *,
        on_step: Callable[[str, List[ToolResult]], bool] | None = None,
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
        # Create a session for this operation
        session = await Session.create()

        # Create a new SessionRun
        run = SessionRun()
        run.mark_running()
        session.add_run(run)

        # Create parent event
        parent_evt = SessionEvent(
            message={'plan_id': plan_node_id},
            type=EventType.SUMMARY,
            source=EventSource.SYSTEM,
            metadata={'description': 'Plan execution started'},
        )
        session.add_event(parent_evt)
        parent_id = parent_evt.id

        # Get steps
        steps = self.plan_executor.get_plan_steps(plan_node_id)
        if not steps:
            raise ValueError(f"No steps found for plan {plan_node_id}")

        # Determine execution order
        batches = self.plan_executor.determine_execution_order(steps)
        all_results: List[ToolResult] = []

        # Execute steps in batches
        for batch in batches:
            for step_id in batch:
                res_list = await self.plan_executor.execute_step(
                    step_id,
                    assistant_node_id,
                    parent_id,
                    lambda event_type, message, parent_id: self._create_child_event(session, event_type, message, parent_id),
                    self._process_single_tool_call,
                )
                all_results.extend(res_list)

                # Per-step callback
                if on_step and on_step(step_id, res_list) is False:
                    run.mark_completed()
                    return all_results

        # Complete the run
        run.mark_completed()
        
        # Create summary event
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
        session.add_event(summary_evt)
        
        return all_results