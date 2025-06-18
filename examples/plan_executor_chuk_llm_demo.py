#!/usr/bin/env python
# examples/plan_executor_chuk_llm_demo.py
"""
Plan Executor LLM-to-Execution Demo (chuk_llm version)
=====================================================

End-to-end demo using the original PlanExecutor that showcases:
1. Converting a natural language task to a JSON plan via chuk_llm
2. Creating a Plan using the original Plan DSL
3. Manually adding tool calls to plan steps
4. Executing the plan with the original PlanExecutor
5. Session management and event tracking

This demonstrates the lower-level workflow compared to UniversalExecutor,
now using chuk_llm instead of OpenAI directly.
"""

import argparse
import asyncio
import json
import os
import pprint
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Import the original Plan implementation
from chuk_ai_planner.planner import Plan, PlanExecutor
from chuk_ai_planner.models import ToolCall, NodeKind
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind
from chuk_ai_planner.store.memory import InMemoryGraphStore

# Session management - fixed imports
from chuk_ai_session_manager.models.session import Session, SessionEvent
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.session_storage import get_backend, ChukSessionsStore, setup_chuk_sessions_storage

# chuk_llm imports
from dotenv import load_dotenv
load_dotenv()

try:
    from chuk_llm import ask, ask_openai, ask_anthropic, ask_openai_gpt4o_mini, ask_anthropic_sonnet
except ImportError:
    print("❌ chuk_llm not available. Install with: pip install chuk-llm")
    exit(1)


# -------------------------------------------------------------------- Mock Tool Implementations
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
        "Sydney": {"temperature": 68, "conditions": "Clear", "humidity": 60},
        "Cairo": {"temperature": 90, "conditions": "Hot", "humidity": 30},
    }
    
    result = weather_data.get(location, {"temperature": 75, "conditions": "Unknown", "humidity": 50})
    
    return {
        "id": tool_call_dict["id"],
        "result": result,
        "success": True
    }


async def calculator_tool_call(tool_call_dict: Dict[str, Any], parent_event_id: str, assistant_node_id: str) -> Dict[str, Any]:
    """Process calculation tool call"""
    args = json.loads(tool_call_dict["function"]["arguments"])
    operation = args.get("operation")
    a = float(args.get("a", 0))
    b = float(args.get("b", 0))
    
    print(f"🧮 Calculating: {a} {operation} {b}")
    
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
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
        {"title": f"Guide to {query}", "url": f"https://guide.com/{query.replace(' ', '-')}"},
    ]
    
    return {
        "id": tool_call_dict["id"],
        "result": {"query": query, "results": results, "count": len(results)},
        "success": True
    }


async def coffee_tool_call(tool_call_dict: Dict[str, Any], parent_event_id: str, assistant_node_id: str) -> Dict[str, Any]:
    """Process coffee-related tool calls"""
    tool_name = tool_call_dict["function"]["name"]
    args = json.loads(tool_call_dict["function"]["arguments"]) if tool_call_dict["function"]["arguments"] else {}
    
    print(f"☕ {tool_name}: {args}")
    
    coffee_actions = {
        "grind_beans": "Beans ground successfully to a medium-fine consistency",
        "boil_water": "Water boiled to 200°F, perfect for brewing coffee",
        "brew_coffee": "Coffee brewed perfectly, strong aroma and rich taste",
        "clean_station": "Coffee station cleaned and ready for next use"
    }
    
    message = coffee_actions.get(tool_name, f"Unknown coffee action: {tool_name}")
    
    return {
        "id": tool_call_dict["id"],
        "result": {"status": "success", "message": message},
        "success": True
    }


# -------------------------------------------------------------------- Tool Registry
class ToolRegistry:
    """Tool registry for routing tool calls to handlers"""
    
    def __init__(self):
        self.tools = {
            "weather": weather_tool_call,
            "calculator": calculator_tool_call,
            "search": search_tool_call,
            "grind_beans": coffee_tool_call,
            "boil_water": coffee_tool_call,
            "brew_coffee": coffee_tool_call,
            "clean_station": coffee_tool_call,
        }
    
    async def process_tool_call(self, tool_call: Dict[str, Any], parent_event_id: str, assistant_node_id: str) -> Any:
        """Route and execute tool call"""
        tool_name = tool_call["function"]["name"]
        
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        handler = self.tools[tool_name]
        return await handler(tool_call, parent_event_id, assistant_node_id)


# -------------------------------------------------------------------- Session Management
class SessionEventManager:
    """Manages session events for plan execution"""
    
    def __init__(self):
        self.session = None
        self.events = []
    
    async def initialize_session(self):
        """Initialize session and session store"""
        # Set up session store using the correct API
        setup_chuk_sessions_storage(sandbox_id="plan-executor-chuk-llm-demo", default_ttl_hours=2)
        
        # Create session
        self.session = await Session.create()
        
        print(f"📋 Session initialized: {self.session.id}")
    
    def create_child_event(self, event_type: EventType, message: Dict[str, Any], parent_event_id: str):
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


# -------------------------------------------------------------------- LLM helpers
LLM_SYSTEM_MSG = (
    "You are an assistant that converts a natural-language task into a JSON "
    "plan. Return ONLY valid JSON!\n"
    "Schema:\n"
    "{\n"
    '  "title": str,\n'
    '  "steps": [              // ordered list\n'
    '    {"title": str, "tool": str, "args": {}, "depends_on": [indices]},\n'
    "    ...\n"
    "  ]\n"
    "}\n"
    "Indices start at 1 in the final output.\n"
    "Available tools: weather, calculator, search, grind_beans, boil_water, brew_coffee, clean_station\n"
    "The 'tool' field is required and must be one of the available tools.\n"
    "The 'args' field should contain the appropriate arguments for the tool."
)


async def call_llm_live(task: str, provider="openai") -> Dict[str, Any]:
    """Call chuk_llm to generate a plan."""
    try:
        print(f"📡 Calling {provider} via chuk_llm to generate plan...")
        
        # Detailed system message for better plan generation
        system_message = (
            "You are an assistant that converts a natural-language task into a JSON "
            "plan. Return ONLY valid JSON!\n\n"
            "Schema:\n"
            "{\n"
            '  "title": str,\n'
            '  "steps": [              // ordered list\n'
            '    {"title": str, "tool": str, "args": {}, "depends_on": [indices]},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "1. Indices start at 1 in the final output\n"
            "2. The 'depends_on' field should list step indices (numbers) that must complete before this step\n"
            "3. Make sure all JSON is valid with proper syntax and no trailing commas\n"
            "4. Include ONLY the JSON in your response, nothing else\n\n"
            "Available tools and their arguments:\n"
            "- weather: {\"location\": \"city name\"}\n"
            "- calculator: {\"operation\": \"add|subtract|multiply|divide\", \"a\": number, \"b\": number}\n"
            "- search: {\"query\": \"search query string\"}\n"
            "- grind_beans: {}\n"
            "- boil_water: {}\n"
            "- brew_coffee: {}\n"
            "- clean_station: {}\n"
        )
        
        # Use the appropriate chuk_llm function based on provider
        full_prompt = f"{system_message}\n\nUser request: {task}"
        
        if provider == "openai":
            # Use specific OpenAI model
            content = await ask_openai_gpt4o_mini(full_prompt)
        elif provider == "anthropic":
            # Use specific Anthropic model
            content = await ask_anthropic_sonnet(full_prompt)
        else:
            # Use generic ask function
            content = await ask(full_prompt, provider=provider)
        
        # Try to extract JSON if there's any surrounding text
        try:
            import re
            json_match = re.search(r'```(?:json)?(.*?)```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
            
            # Another common pattern is JSON without code blocks
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
                
            return json.loads(content)
        except (json.JSONDecodeError, AttributeError):
            # If we can't extract JSON, try to parse the whole response
            return json.loads(content)
            
    except Exception as e:
        print(f"❌ Error calling chuk_llm: {e}")
        print("📋 Falling back to simulated response...")
        # Fall back to simulation if API call fails
        return await call_llm_sim(task)


async def call_llm_sim(task: str) -> Dict[str, Any]:
    """
    Simulation that returns a fixed plan (no actual LLM call)
    """
    print("🤖 Simulating LLM response (no API call)...")
    print(f"   Task: {task}")
    print("   LLM system prompt:", LLM_SYSTEM_MSG[:100] + "...")
    
    return {
        "title": "Coffee, Weather, Calculation, and Search",
        "steps": [
            {
                "title": "Grind coffee beans",
                "tool": "grind_beans",
                "args": {},
                "depends_on": []
            },
            {
                "title": "Boil water",
                "tool": "boil_water",
                "args": {},
                "depends_on": []
            },
            {
                "title": "Brew coffee",
                "tool": "brew_coffee",
                "args": {},
                "depends_on": [1, 2]
            },
            {
                "title": "Check weather in New York",
                "tool": "weather",
                "args": {"location": "New York"},
                "depends_on": [3]
            },
            {
                "title": "Multiply 235.5 × 18.75",
                "tool": "calculator",
                "args": {"operation": "multiply", "a": 235.5, "b": 18.75},
                "depends_on": [3]
            },
            {
                "title": "Search for climate-adaptation info",
                "tool": "search",
                "args": {"query": "climate change adaptation"},
                "depends_on": [4, 5]
            },
            {
                "title": "Clean coffee station",
                "tool": "clean_station",
                "args": {},
                "depends_on": [3]
            }
        ]
    }


# -------------------------------------------------------------------- Plan conversion
def convert_to_plan(llm_json: Dict[str, Any]) -> tuple[Plan, str, Dict[int, str]]:
    """Convert LLM-generated JSON to a Plan using the original DSL."""
    # Create a new plan with custom graph store
    graph_store = InMemoryGraphStore()
    plan = Plan(llm_json["title"], graph=graph_store)
    
    # Create a mapping of LLM step index to step IDs for dependencies
    step_id_map = {}
    
    # Build the plan structure using the fluent interface
    for i, step_data in enumerate(llm_json["steps"], 1):
        title = step_data["title"]
        depends_on = step_data.get("depends_on", [])
        
        # Convert depends_on indices to step IDs (if we have them)
        after_ids = []
        for dep_idx in depends_on:
            if dep_idx in step_id_map:
                after_ids.append(step_id_map[dep_idx])
        
        # Add the step (note: after parameter expects step indices, not IDs)
        # For now, we'll add dependencies manually after the plan is saved
        plan.step(title).up()
    
    # Save the plan to generate step IDs
    plan_id = plan.save()
    
    # Get all plan steps to create the mapping
    steps = []
    for node in plan.graph.nodes.values():
        if node.kind == NodeKind.PLAN_STEP:
            steps.append(node)
    
    # Sort by index to match our step order
    steps.sort(key=lambda n: int(n.data.get("index", "0")))
    
    # Create mapping from LLM step index to actual step ID
    for i, step in enumerate(steps, 1):
        step_id_map[i] = step.id
    
    # Now add dependencies using the proper step IDs
    for i, step_data in enumerate(llm_json["steps"], 1):
        step_id = step_id_map.get(i)
        if not step_id:
            continue
            
        depends_on = step_data.get("depends_on", [])
        for dep_idx in depends_on:
            dep_id = step_id_map.get(dep_idx)
            if dep_id:
                # Add step order edge
                plan.graph.add_edge(GraphEdge(
                    kind=EdgeKind.STEP_ORDER,
                    src=dep_id,
                    dst=step_id
                ))
    
    return plan, plan_id, step_id_map


def add_tool_calls_to_plan(plan: Plan, llm_json: Dict[str, Any], step_id_map: Dict[int, str]) -> None:
    """Add tool calls to plan steps based on LLM JSON."""
    
    for i, step_data in enumerate(llm_json["steps"], 1):
        step_id = step_id_map.get(i)
        if not step_id:
            continue
            
        tool_name = step_data.get("tool")
        tool_args = step_data.get("args", {})
        
        if tool_name:
            # Create tool call node
            tool_call = ToolCall(data={
                "name": tool_name,
                "args": tool_args
            })
            plan.graph.add_node(tool_call)
            
            # Link step to tool call
            plan.graph.add_edge(GraphEdge(
                kind=EdgeKind.PLAN_LINK,
                src=step_id,
                dst=tool_call.id
            ))


# -------------------------------------------------------------------- main flow
async def main(live: bool = False, provider: str = "openai") -> None:
    print("🚀 Plan Executor chuk_llm-to-Execution Demo\n" + "=" * 50)
    
    # Allow the user to customize the task via command line
    parser = argparse.ArgumentParser(description="chuk_llm to Plan Executor Demo")
    parser.add_argument("--task", type=str, 
                        default="I need a plan that first prepares coffee "
                                "then checks today's weather in New York, multiplies 235.5×18.75, "
                                "and finally searches for pages on climate-change adaptation.",
                        help="The natural language task to convert to a plan")
    parser.add_argument("--live", action="store_true",
                        help="Use real chuk_llm API instead of simulation")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "auto"],
                        help="LLM provider to use (openai, anthropic, or auto)")
    args = parser.parse_args()
    
    # Check for required API keys
    if args.live:
        required_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        
        if args.provider != "auto" and not os.getenv(required_keys.get(args.provider)):
            print(f"Error: To use --live with {args.provider}, set the {required_keys[args.provider]} environment variable")
            exit(1)
    
    task = args.task

    # Step 1: Get LLM-generated plan
    print("\n📝 STEP 1: GENERATING PLAN FROM chuk_llm...\n")
    if live:
        try:
            llm_json = await call_llm_live(task, provider)
        except Exception as e:
            print(f"❌ Error with chuk_llm: {e}")
            print("📋 Falling back to simulated response...")
            llm_json = await call_llm_sim(task)
    else:
        llm_json = await call_llm_sim(task)
    
    print(f"\nLLM Response:")
    print(json.dumps(llm_json, indent=2))

    # Step 2: Convert to Plan
    print("\n🔄 STEP 2: CONVERTING TO PLAN DSL...\n")
    try:
        plan, plan_id, step_id_map = convert_to_plan(llm_json)
        
        print("Plan Structure (before tool calls):")
        print(plan.outline())
        
        # Add tool calls to the plan
        add_tool_calls_to_plan(plan, llm_json, step_id_map)
        
        print("\nPlan conversion completed!")
        print(f"- Plan ID: {plan_id}")
        print(f"- Steps created: {len(step_id_map)}")
        print(f"- Tool calls added to steps")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Error converting LLM output to Plan: {e}")
        return

    # Step 3: Set up executor and session
    print("\n⚙️ STEP 3: SETTING UP PLAN EXECUTOR...\n")
    
    # Initialize components
    executor = PlanExecutor(plan.graph)
    session_manager = SessionEventManager()
    tool_registry = ToolRegistry()
    
    # Initialize session
    await session_manager.initialize_session()
    
    print("Plan Executor initialized with:")
    print(f"- Graph store with {len(plan.graph.nodes)} nodes")
    print(f"- Tool registry with {len(tool_registry.tools)} tools")
    print(f"- Session management ready")

    # Step 4: Analyze plan structure
    print("\n📊 STEP 4: ANALYZING PLAN STRUCTURE...\n")
    
    # Get plan steps
    steps = executor.get_plan_steps(plan_id)
    print(f"Found {len(steps)} plan steps:")
    for step in steps:
        index = step.data.get("index")
        desc = step.data.get("description")
        print(f"  {index}: {desc} (id: {step.id[:8]})")
    
    # Determine execution order
    batches = executor.determine_execution_order(steps)
    print(f"\nExecution will proceed in {len(batches)} batches:")
    for i, batch in enumerate(batches, 1):
        print(f"  Batch {i}: {len(batch)} steps")
        for step_id in batch:
            # Find step by ID
            step = next((s for s in steps if s.id == step_id), None)
            if step:
                index = step.data.get("index")
                desc = step.data.get("description")
                print(f"    {index}: {desc}")

    # Step 5: Execute the plan
    print("\n🏃 STEP 5: EXECUTING PLAN...\n")
    
    assistant_node_id = str(uuid.uuid4())
    parent_event_id = str(uuid.uuid4())
    
    all_results = []
    
    try:
        for batch_num, batch in enumerate(batches, 1):
            print(f"\n--- Executing Batch {batch_num} ---")
            
            # Execute all steps in batch (could be parallel in real implementation)
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
        
        print("\n✅ Plan execution completed successfully!\n")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Error during plan execution: {e}")
        return

    # Step 6: Display results
    print("\n" + "=" * 60)
    print("EXECUTION RESULTS")
    print("=" * 60)
    
    print(f"\nTotal results: {len(all_results)}")
    for i, result in enumerate(all_results, 1):
        print(f"\n--- Result {i} ---")
        pprint.pprint(result, width=100, sort_dicts=False)
    
    # Save results to file
    try:
        output_file = "plan_executor_chuk_llm_results.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {output_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results to file: {e}")
    
    # Step 7: Display session events
    print("\n" + "=" * 60)
    print("SESSION EVENTS")
    print("=" * 60)
    
    print(f"\nTotal events: {len(session_manager.events)}")
    for i, event in enumerate(session_manager.events, 1):
        timestamp = datetime.now().strftime("%H:%M:%S")
        event_type = event.type.value
        message_summary = str(event.message)[:80] + "..." if len(str(event.message)) > 80 else str(event.message)
        print(f"{i:2d}. {timestamp} [{event_type:8}] {message_summary}")
    
    # Step 8: Summary
    print(f"\n🎉 Demo completed successfully!")
    print(f"   - Generated plan from natural language task using chuk_llm")
    print(f"   - Used {provider} provider via chuk_llm")
    print(f"   - Created {len(steps)} plan steps with dependencies")
    print(f"   - Executed {len(batches)} batches with proper ordering")
    print(f"   - Processed {len(all_results)} tool results")
    print(f"   - Generated {len(session_manager.events)} session events")
    print(f"   - Demonstrated lower-level Plan Executor API with chuk_llm")


# -------------------------------------------------------------------- entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chuk_llm to Plan Executor Demo")
    parser.add_argument("--live", action="store_true",
                        help="Use real chuk_llm API instead of simulation")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "auto"],
                        help="LLM provider to use (openai, anthropic, or auto)")
    parser.add_argument("--task", type=str, 
                        default="I need a plan that first prepares coffee "
                                "then checks today's weather in New York, multiplies 235.5×18.75, "
                                "and finally searches for pages on climate-change adaptation.",
                        help="The natural language task to convert to a plan")
    args = parser.parse_args()
    
    # Check for required API keys
    if args.live:
        required_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        
        if args.provider != "auto" and not os.getenv(required_keys.get(args.provider)):
            print(f"Error: To use --live with {args.provider}, set the {required_keys[args.provider]} environment variable")
            exit(1)
    
    asyncio.run(main(args.live, args.provider))