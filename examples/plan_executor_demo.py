#!/usr/bin/env python
# examples/plan_executor_demo.py
"""
Original Plan Executor E2E Demo
===============================

Demonstrates the core functionality of the original PlanExecutor:
1. Creating a plan with the Plan DSL
2. Adding tool calls to plan steps
3. Using PlanExecutor to get steps and determine execution order
4. Executing steps with proper session event handling

This shows the lower-level API compared to UniversalExecutor.
"""

import asyncio
import json
import pprint
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Core imports
from chuk_ai_planner.planner import Plan, PlanExecutor
from chuk_ai_planner.models import ToolCall, NodeKind
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind
from chuk_ai_planner.store.memory import InMemoryGraphStore

# Session management - using chuk-ai-session-manager  
from chuk_ai_session_manager.models.session import Session, SessionEvent
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore, setup_chuk_sessions_storage


# --------------------------------------------------------------------------- Mock Tools
async def weather_tool_call(tool_call_dict: Dict[str, Any], parent_event_id: str, assistant_node_id: str) -> Dict[str, Any]:
    """Process weather tool call"""
    args = json.loads(tool_call_dict["function"]["arguments"])
    location = args.get("location", "Unknown")
    
    print(f"🌤️  Getting weather for: {location}")
    
    # Mock weather data
    weather_data = {
        "New York": {"temperature": 72, "conditions": "Partly cloudy", "humidity": 65},
        "London": {"temperature": 62, "conditions": "Rainy", "humidity": 80},
        "Tokyo": {"temperature": 78, "conditions": "Sunny", "humidity": 70},
    }
    
    result = weather_data.get(location, {"temperature": 75, "conditions": "Unknown", "humidity": 50})
    
    return {
        "id": tool_call_dict["id"],
        "result": result,
        "success": True
    }


async def calculation_tool_call(tool_call_dict: Dict[str, Any], parent_event_id: str, assistant_node_id: str) -> Dict[str, Any]:
    """Process calculation tool call"""
    args = json.loads(tool_call_dict["function"]["arguments"])
    operation = args.get("operation")
    a = float(args.get("a", 0))
    b = float(args.get("b", 0))
    
    print(f"🧮 Calculating: {a} {operation} {b}")
    
    if operation == "add":
        result = a + b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        result = a / b if b != 0 else "Error: Division by zero"
    else:
        result = f"Error: Unknown operation {operation}"
    
    return {
        "id": tool_call_dict["id"],
        "result": {"calculation": f"{a} {operation} {b}", "answer": result},
        "success": True
    }


async def search_tool_call(tool_call_dict: Dict[str, Any], parent_event_id: str, assistant_node_id: str) -> Dict[str, Any]:
    """Process search tool call"""
    args = json.loads(tool_call_dict["function"]["arguments"])
    query = args.get("query", "")
    
    print(f"🔍 Searching for: {query}")
    
    # Mock search results
    results = [
        {"title": f"Top result for {query}", "url": f"https://example.com/{query.replace(' ', '-')}"},
        {"title": f"Research on {query}", "url": f"https://research.com/{query.replace(' ', '_')}"},
    ]
    
    return {
        "id": tool_call_dict["id"],
        "result": {"query": query, "results": results, "count": len(results)},
        "success": True
    }


# --------------------------------------------------------------------------- Session & Event Management
class SessionEventManager:
    """Manages session events for plan execution"""
    
    def __init__(self):
        self.session = None
        self.events = []
    
    async def initialize_session(self):
        """Initialize session and session store"""
        # Set up session store using the correct API
        setup_chuk_sessions_storage(sandbox_id="plan-executor-demo", default_ttl_hours=2)
        
        # Create session
        self.session = await Session.create()
        
        print(f"📋 Session initialized: {self.session.id}")
    
    def create_child_event(self, event_type, message: Dict[str, Any], parent_event_id: str):
        """Create a child event using the session manager API"""
        # Create event using SessionEvent
        event = SessionEvent(
            message=message,
            source=EventSource.SYSTEM,
            type=event_type,
        )
        
        # Add metadata for parent relationship if provided
        if parent_event_id:
            event.metadata = {"parent_event_id": parent_event_id}
        
        self.events.append(event)
        
        # Log the event
        if event_type == EventType.SUMMARY:
            status = message.get("status", "")
            step_desc = message.get("description", "")
            print(f"📝 Event: {step_desc} - {status}")
        
        return event


# --------------------------------------------------------------------------- Tool Registry
class ToolRegistry:
    """Simple tool registry for routing tool calls"""
    
    def __init__(self):
        self.tools = {
            "weather": weather_tool_call,
            "calculator": calculation_tool_call, 
            "search": search_tool_call,
        }
    
    async def process_tool_call(self, tool_call: Dict[str, Any], parent_event_id: str, assistant_node_id: str) -> Any:
        """Route and execute tool call"""
        tool_name = tool_call["function"]["name"]
        
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        handler = self.tools[tool_name]
        return await handler(tool_call, parent_event_id, assistant_node_id)


# --------------------------------------------------------------------------- Plan Creation
def create_demo_plan() -> tuple[Plan, str]:
    """Create a demo plan with multiple steps and dependencies"""
    
    # Create plan with custom graph store
    graph_store = InMemoryGraphStore()
    plan = Plan("Multi-Step Data Analysis", graph=graph_store)
    
    # Build plan structure using fluent interface
    plan.step("Get weather data for New York")
    plan.up()
    
    plan.step("Get weather data for London") 
    plan.up()
    
    plan.step("Calculate temperature difference", after=["1", "2"])  # Depends on both weather steps
    plan.up()
    
    plan.step("Search for climate information", after=["3"])  # Depends on calculation
    plan.up()
    
    plan.step("Generate final summary", after=["4"])  # Depends on search
    plan.up()
    
    # Save the plan to generate IDs and structure
    plan_id = plan.save()
    
    return plan, plan_id


def add_tool_calls_to_plan(plan: Plan) -> Dict[str, str]:
    """Add tool calls to plan steps and return step_id -> tool mapping"""
    
    # Get all plan steps
    steps = []
    for node in plan.graph.nodes.values():
        if node.kind == NodeKind.PLAN_STEP:
            steps.append(node)
    
    # Sort by index to match our plan structure
    steps.sort(key=lambda n: n.data.get("index", ""))
    
    step_tools = {}
    
    for i, step in enumerate(steps):
        step_id = step.id
        index = step.data.get("index")
        
        # Create appropriate tool call based on step
        if index == "1":  # NYC weather
            tool_call = ToolCall(data={
                "name": "weather",
                "args": {"location": "New York"}
            })
            step_tools[step_id] = "weather (NYC)"
            
        elif index == "2":  # London weather  
            tool_call = ToolCall(data={
                "name": "weather", 
                "args": {"location": "London"}
            })
            step_tools[step_id] = "weather (London)"
            
        elif index == "3":  # Temperature calculation
            tool_call = ToolCall(data={
                "name": "calculator",
                "args": {"operation": "add", "a": 72, "b": -62}  # NYC - London (mock)
            })
            step_tools[step_id] = "calculator"
            
        elif index == "4":  # Climate search
            tool_call = ToolCall(data={
                "name": "search",
                "args": {"query": "climate temperature differences New York London"}
            })
            step_tools[step_id] = "search"
            
        elif index == "5":  # Summary calculation
            tool_call = ToolCall(data={
                "name": "calculator", 
                "args": {"operation": "multiply", "a": 2, "b": 3}  # Mock summary calc
            })
            step_tools[step_id] = "calculator (summary)"
        
        else:
            continue
        
        # Add tool call to graph and link to step
        plan.graph.add_node(tool_call)
        plan.graph.add_edge(GraphEdge(
            kind=EdgeKind.PLAN_LINK,
            src=step_id,
            dst=tool_call.id
        ))
    
    return step_tools


# --------------------------------------------------------------------------- Main Demo
async def main():
    """Main demo function"""
    print("🚀 Original Plan Executor E2E Demo")
    print("=" * 50)
    
    # 1. Create plan
    print("\n📋 STEP 1: CREATING PLAN...")
    plan, plan_id = create_demo_plan()
    
    print(f"Plan created with ID: {plan_id}")
    print("\nPlan structure:")
    print(plan.outline())
    
    # 2. Add tool calls
    print("\n🔧 STEP 2: ADDING TOOL CALLS...")
    step_tools = add_tool_calls_to_plan(plan)
    
    print("Tool calls added:")
    for step_id, tool_desc in step_tools.items():
        print(f"  {step_id[:8]} -> {tool_desc}")
    
    # 3. Initialize executor and session
    print("\n⚙️  STEP 3: INITIALIZING EXECUTOR...")
    executor = PlanExecutor(plan.graph)
    session_manager = SessionEventManager()
    tool_registry = ToolRegistry()
    
    await session_manager.initialize_session()
    
    # 4. Get plan steps
    print("\n📊 STEP 4: ANALYZING PLAN STRUCTURE...")
    steps = executor.get_plan_steps(plan_id)
    
    print(f"Found {len(steps)} plan steps:")
    for step in steps:
        index = step.data.get("index")
        desc = step.data.get("description")
        print(f"  {index}: {desc} (id: {step.id[:8]})")
    
    # 5. Determine execution order
    print("\n🔄 STEP 5: DETERMINING EXECUTION ORDER...")
    batches = executor.determine_execution_order(steps)
    
    print(f"Execution will proceed in {len(batches)} batches:")
    for i, batch in enumerate(batches, 1):
        print(f"  Batch {i}: {len(batch)} steps")
        for step_id in batch:
            # Find step by ID
            step = next((s for s in steps if s.id == step_id), None)
            if step:
                index = step.data.get("index")
                desc = step.data.get("description")
                print(f"    {index}: {desc}")
    
    # 6. Execute plan
    print("\n🏃 STEP 6: EXECUTING PLAN...")
    
    assistant_node_id = str(uuid.uuid4())
    parent_event_id = str(uuid.uuid4())
    
    all_results = []
    
    for batch_num, batch in enumerate(batches, 1):
        print(f"\n--- Executing Batch {batch_num} ---")
        
        # Execute all steps in batch (could be parallel)
        batch_results = []
        for step_id in batch:
            print(f"\nExecuting step {step_id[:8]}...")
            
            try:
                step_results = await executor.execute_step(
                    step_id=step_id,
                    assistant_node_id=assistant_node_id,
                    parent_event_id=parent_event_id,
                    create_child_event=session_manager.create_child_event,
                    process_tool_call=tool_registry.process_tool_call
                )
                
                batch_results.extend(step_results)
                print(f"✅ Step completed with {len(step_results)} results")
                
            except Exception as e:
                print(f"❌ Step failed: {e}")
                batch_results.append({"error": str(e)})
        
        all_results.extend(batch_results)
    
    # 7. Display results
    print("\n" + "=" * 50)
    print("EXECUTION RESULTS")
    print("=" * 50)
    
    print(f"\nTotal results: {len(all_results)}")
    for i, result in enumerate(all_results, 1):
        print(f"\n--- Result {i} ---")
        pprint.pprint(result, width=80, sort_dicts=False)
    
    # 8. Display session events
    print("\n" + "=" * 50)
    print("SESSION EVENTS")
    print("=" * 50)
    
    print(f"\nTotal events: {len(session_manager.events)}")
    for event in session_manager.events:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Current time as placeholder
        event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)
        message_summary = str(event.message)[:100] + "..." if len(str(event.message)) > 100 else str(event.message)
        print(f"{timestamp} [{event_type}] {message_summary}")
    
    # 9. Summary
    print("\n🎉 Demo completed successfully!")
    print(f"   - Executed {len(steps)} plan steps")
    print(f"   - Processed {len(batches)} execution batches")
    print(f"   - Generated {len(all_results)} tool results")
    print(f"   - Created {len(session_manager.events)} session events")


# --------------------------------------------------------------------------- Entry Point
if __name__ == "__main__":
    asyncio.run(main())