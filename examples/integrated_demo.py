"""
Comprehensive example integrating:
1. Account/Project/Session management
2. Graph-based conversation model
3. Tool execution with session awareness
4. Prompt building for multi-turn conversations
"""

import asyncio
import logging
import json
from uuid import uuid4
from datetime import datetime, timezone
from typing import Dict, List, Any

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# --- Account & Project imports ---
from a2a_accounts.models.access_control import AccessControlled
from a2a_accounts.models.project import Project
from a2a_accounts.models.account import Account
from a2a_accounts.models.access_levels import AccessLevel

# --- Session Manager imports ---
from a2a_session_manager.storage import InMemorySessionStore, SessionStoreProvider
from a2a_session_manager.models.session import Session
from a2a_session_manager.models.session_event import SessionEvent
from a2a_session_manager.models.event_type import EventType
from a2a_session_manager.models.event_source import EventSource
from a2a_session_manager.models.session_run import SessionRun, RunStatus

# --- Graph Model imports ---
from chuk_ai_planner.models import (
    NodeKind, GraphNode, SessionNode, PlanNode, PlanStep,
    UserMessage, AssistantMessage, ToolCall, TaskRun, Summary
)
from chuk_ai_planner.models.edges import (
    EdgeKind, GraphEdge, ParentChildEdge, NextEdge, PlanEdge, StepEdge 
)

# --- Tool Processor imports ---
from chuk_ai_planner.session_aware_tool_processor import SessionAwareToolProcessor

# --- Simulated LLM API ---
async def simulate_llm_call(prompt_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simulate an LLM API call."""
    # In a real implementation, this would call OpenAI or another LLM API
    print("\n🤖 Simulated LLM prompted with:")
    for msg in prompt_messages:
        role = msg["role"]
        content = msg.get("content", "")
        if content and len(content) > 50:
            content = content[:50] + "..."
        print(f"  {role}: {content}")
    
    # For demo purposes, return a fixed response with tool calls
    user_msg = next((m for m in prompt_messages if m["role"] == "user"), {}).get("content", "")
    
    # If the prompt contains weather question
    if "weather" in user_msg.lower():
        return {
            "content": "I'll check the weather for you.",
            "tool_calls": [
                {
                    "id": f"call_{uuid4()}",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": json.dumps({"location": "New York"})
                    }
                }
            ]
        }
    # If the prompt contains math question
    elif any(x in user_msg.lower() for x in ["calculate", "math", "multiply", "divide"]):
        return {
            "content": "I'll calculate that for you.",
            "tool_calls": [
                {
                    "id": f"call_{uuid4()}",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": json.dumps({
                            "operation": "multiply", 
                            "a": 235.5, 
                            "b": 18.75
                        })
                    }
                }
            ]
        }
    # Default response
    else:
        return {
            "content": "I'm not sure how to help with that specific request.",
            "tool_calls": []
        }

# --- Simulated tool implementations ---
async def execute_weather_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate weather tool execution."""
    location = args.get("location", "Unknown")
    return {
        "temperature": 22.5,
        "conditions": "Partly Cloudy",
        "humidity": 65.0,
        "location": location
    }

async def execute_calculator_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate calculator tool execution."""
    op = args.get("operation")
    a = args.get("a", 0)
    b = args.get("b", 0)
    
    if op == "multiply":
        return {"operation": op, "result": a * b}
    elif op == "add":
        return {"operation": op, "result": a + b}
    else:
        return {"operation": op, "error": "Unsupported operation"}

# --- Tool registry (simplified) ---
TOOL_REGISTRY = {
    "weather": execute_weather_tool,
    "calculator": execute_calculator_tool
}

# --- Prompt building from session ---
def build_prompt_from_session(session: Session) -> List[Dict[str, Any]]:
    """Build a minimal prompt for the next LLM call from a Session."""
    if not session.events:
        return []

    # First USER message
    first_user = next(
        (
            e
            for e in session.events
            if e.type == EventType.MESSAGE and e.source == EventSource.USER
        ),
        None,
    )

    # Latest assistant MESSAGE
    assistant_msg = next(
        (
            ev
            for ev in reversed(session.events)
            if ev.type == EventType.MESSAGE and ev.source != EventSource.USER
        ),
        None,
    )
    
    if assistant_msg is None:
        return [{"role": "user", "content": first_user.message}] if first_user else []

    # Children of that assistant
    children = [
        e
        for e in session.events
        if e.metadata.get("parent_event_id") == assistant_msg.id
    ]
    tool_calls = [c for c in children if c.type == EventType.TOOL_CALL]

    # Assemble prompt
    prompt = []
    if first_user:
        prompt.append({"role": "user", "content": first_user.message})

    # Add the assistant marker with content set to None
    prompt.append({"role": "assistant", "content": None})

    # Add tool results
    for tc in tool_calls:
        prompt.append(
            {
                "role": "tool",
                "name": tc.message["tool"],
                "content": json.dumps(tc.message["result"])
            }
        )

    return prompt

# --- Process tool calls from LLM message ---
async def process_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process tool calls from an LLM message."""
    results = []
    
    for call in tool_calls:
        try:
            function_data = call.get("function", {})
            tool_name = function_data.get("name")
            args = json.loads(function_data.get("arguments", "{}"))
            
            # Look up the tool implementation
            tool_fn = TOOL_REGISTRY.get(tool_name)
            if not tool_fn:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            # Execute the tool
            result = await tool_fn(args)
            
            results.append({
                "tool": tool_name,
                "args": args,
                "result": result,
                "error": None
            })
        except Exception as e:
            results.append({
                "tool": tool_name if 'tool_name' in locals() else "unknown",
                "args": args if 'args' in locals() else {},
                "result": None,
                "error": str(e)
            })
    
    return results

# --- Integrated demonstration ---
async def main():
    print("🚀 Starting integrated demo of Session + Graph + Tools")
    
    # 1. Initialize storage
    store = InMemorySessionStore()
    SessionStoreProvider.set_store(store)
    print("🗄️  Initialized in-memory session store")
    
    # 2. Set up account and project structure
    acct = Account(name="Demo User", owner_user_id="user1")
    proj = Project(
        name="Weather Assistant",
        account_id=acct.id,
        access_level=AccessLevel.PRIVATE
    )
    acct.add_project(proj)
    print(f"👤 Created account '{acct.name}' with project '{proj.name}'")
    
    # 3. Create a session
    session = Session()
    store.save(session)
    proj.add_session(session)
    print(f"💬 Created session {session.id}")
    
    # 4. Initialize the graph structure
    graph_nodes = []
    graph_edges = []
    
    # Create session node in graph
    session_node = SessionNode(
        data={"session_manager_id": session.id}
    )
    graph_nodes.append(session_node)
    print(f"📊 Created graph session node {session_node.id}")
    
    # 5. Add user message
    user_content = "What's the weather like in New York, and can you calculate 235.5 × 18.75?"
    
    # Add to session manager
    user_event = SessionEvent(
        message=user_content,
        source=EventSource.USER,
        type=EventType.MESSAGE
    )
    session.events.append(user_event)
    store.save(session)
    
    # Add to graph
    user_msg_node = UserMessage(
        data={"content": user_content}
    )
    graph_nodes.append(user_msg_node)
    
    # Connect user message to session
    session_to_user = ParentChildEdge(
        src=session_node.id,
        dst=user_msg_node.id
    )
    graph_edges.append(session_to_user)
    print(f"✏️ Added user message to session and graph")
    
    # 6. Create a plan
    plan_node = PlanNode(
        data={"description": "Answer weather query and perform calculation"}
    )
    graph_nodes.append(plan_node)
    
    # Connect plan to session
    session_to_plan = ParentChildEdge(
        src=session_node.id,
        dst=plan_node.id
    )
    graph_edges.append(session_to_plan)
    
    # 7. Create plan steps
    steps = [
        PlanStep(data={"description": "Check the weather in New York", "index": 1}),
        PlanStep(data={"description": "Calculate 235.5 × 18.75", "index": 2}),
        PlanStep(data={"description": "Provide final response", "index": 3})
    ]
    
    # Add steps to graph and connect to plan
    for i, step in enumerate(steps):
        graph_nodes.append(step)
        
        # Connect step to plan
        plan_to_step = ParentChildEdge(
            src=plan_node.id,
            dst=step.id
        )
        graph_edges.append(plan_to_step)
        
        # Connect step to previous step (if applicable)
        if i > 0:
            step_to_step = StepEdge(
                src=steps[i-1].id,
                dst=step.id
            )
            graph_edges.append(step_to_step)
    
    print(f"📝 Created plan with {len(steps)} steps")
    
    # 8. Generate LLM response - simulate assistant message
    prompt = build_prompt_from_session(session)
    llm_response = await simulate_llm_call(prompt)
    
    # 9. Process tool calls
    tool_calls = llm_response.get("tool_calls", [])
    tool_results = await process_tool_calls(tool_calls)
    
    # Add assistant response to session
    assistant_event = SessionEvent(
        message=llm_response,
        source=EventSource.LLM,
        type=EventType.MESSAGE
    )
    session.events.append(assistant_event)
    store.save(session)
    
    # Add assistant message to graph
    assistant_msg_node = AssistantMessage(
        data={
            "content": llm_response.get("content"),
            "tool_calls": tool_calls
        }
    )
    graph_nodes.append(assistant_msg_node)
    
    # Connect assistant message
    session_to_assistant = ParentChildEdge(
        src=session_node.id,
        dst=assistant_msg_node.id
    )
    user_to_assistant = NextEdge(
        src=user_msg_node.id,
        dst=assistant_msg_node.id
    )
    graph_edges.append(session_to_assistant)
    graph_edges.append(user_to_assistant)
    
    print(f"🤖 Added assistant response with {len(tool_calls)} tool calls")
    
    # 10. Add tool calls and results to session and graph
    for i, (tool_call, result) in enumerate(zip(tool_calls, tool_results)):
        # Add to session - Session-aware format
        tool_event = SessionEvent(
            message=result,
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL,
            metadata={"parent_event_id": assistant_event.id}
        )
        session.events.append(tool_event)
        
        # Add to graph
        tool_node = ToolCall(
            data={
                "name": result["tool"],
                "args": result["args"],
                "result": result["result"]
            }
        )
        graph_nodes.append(tool_node)
        
        # Connect tool call
        assistant_to_tool = ParentChildEdge(
            src=assistant_msg_node.id,
            dst=tool_node.id
        )
        graph_edges.append(assistant_to_tool)
        
        # Connect tool to plan step
        step_to_tool = PlanEdge(
            src=steps[i].id,
            dst=tool_node.id
        )
        graph_edges.append(step_to_tool)
        
        # Add task run
        task_run = TaskRun(
            data={
                "tool": result["tool"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": result["error"] is None
            }
        )
        graph_nodes.append(task_run)
        
        # Connect task run
        tool_to_task = ParentChildEdge(
            src=tool_node.id,
            dst=task_run.id
        )
        graph_edges.append(tool_to_task)
    
    store.save(session)
    print(f"🔧 Added {len(tool_results)} tool executions to session and graph")
    
    # 11. Create summary
    summary_content = "The weather in New York is 22.5°F and Partly Cloudy. The calculation 235.5 × 18.75 equals 4415.625."
    summary_node = Summary(
        data={"content": summary_content}
    )
    graph_nodes.append(summary_node)
    
    # Connect summary
    plan_to_summary = PlanEdge(
        src=steps[2].id,
        dst=summary_node.id
    )
    graph_edges.append(plan_to_summary)
    
    # 12. Final assistant message
    final_msg_node = AssistantMessage(
        data={"content": summary_content}
    )
    graph_nodes.append(final_msg_node)
    
    # Connect final message
    session_to_final = ParentChildEdge(
        src=session_node.id,
        dst=final_msg_node.id
    )
    assistant_to_final = NextEdge(
        src=assistant_msg_node.id,
        dst=final_msg_node.id
    )
    graph_edges.append(session_to_final)
    graph_edges.append(assistant_to_final)
    
    print(f"📊 Added summary and final response")
    
    # 13. Print session structure
    print("\n==== SESSION STRUCTURE ====")
    print(f"Session ID: {session.id}")
    print(f"Events: {len(session.events)}")
    for i, event in enumerate(session.events):
        indent = "  " if event.metadata.get("parent_event_id") else ""
        print(f"{indent}• {event.type.value:10} id={event.id}")
        
        if event.type == EventType.TOOL_CALL:
            tool_name = event.message.get("tool", "unknown")
            has_error = event.message.get("error") is not None
            error_str = "error=Yes" if has_error else "error=None"
            print(f"    ⇒ {tool_name:10} {error_str}")
    
    # 14. Print graph structure
    print("\n==== GRAPH STRUCTURE ====")
    print(f"Nodes: {len(graph_nodes)}")
    print(f"Edges: {len(graph_edges)}")
    
    # Find all direct children of session
    session_children = [e for e in graph_edges if e.src == session_node.id]
    print(f"\nSession node: {session_node!r}")
    
    for edge in session_children:
        child = next((n for n in graph_nodes if n.id == edge.dst), None)
        if child:
            print(f"├── {child.kind.value}: {child!r}")
    
    # Print the plan structure
    print(f"\nPlan: {plan_node!r}")
    
    for i, step in enumerate(steps):
        is_last = i == len(steps) - 1
        prefix = "└──" if is_last else "├──"
        print(f"{prefix} Step {i+1}: {step.data.get('description')}")
        
        # Find executions linked to this step
        exec_edges = [e for e in graph_edges if e.src == step.id and isinstance(e, PlanEdge)]
        for j, exec_edge in enumerate(exec_edges):
            is_last_exec = j == len(exec_edges) - 1
            exec_prefix = "    └──" if is_last_exec else "    ├──"
            
            execution = next((n for n in graph_nodes if n.id == exec_edge.dst), None)
            if execution:
                print(f"{exec_prefix} Execution: {execution!r}")
                
                if execution.kind == NodeKind.TOOL_CALL:
                    print(f"        └── {execution.data.get('name', 'unnamed')}")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    asyncio.run(main())