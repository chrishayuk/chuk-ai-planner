#!/usr/bin/env python
"""
demo_llm_plan.py
================

Natural-language task → LLM-generated JSON plan → executed plan
"""

from __future__ import annotations
import argparse, asyncio, json, os, re, uuid
from typing import Dict, Any, List
import inspect

# ── Explicitly import the sample tools ──────────────────────────────────
# We'll import them but modify how we use them
from chuk_ai_planner.store.base import GraphStore
from sample_tools import WeatherTool, CalculatorTool, SearchTool

# ── A2A plumbing -----------------------------------------------------
from chuk_session_manager.storage import InMemorySessionStore, SessionStoreProvider
from chuk_session_manager.models.session import Session
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.planner import Plan
from chuk_ai_planner.models import ToolCall
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind
from chuk_ai_planner.utils.visualization import print_session_events, print_graph_structure

from dotenv import load_dotenv
load_dotenv()

# ── OpenAI (optional) -------------------------------------------------
from chuk_ai_planner.demo.llm_simulator import simulate_llm_call
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore

# ── Simple wrapper for sample tools ----------------------------------
class ToolWrapper:
    """Wrapper to standardize tool interface."""
    
    def __init__(self, tool_class):
        """Initialize with a tool class."""
        self.tool_class = tool_class
        
    async def run(self, **kwargs):
        """Run the tool with given arguments."""
        # Create an instance of the tool
        instance = self.tool_class()
        
        # Check if the instance has an arun method (async)
        if hasattr(instance, 'arun') and callable(instance.arun):
            return await instance.arun(**kwargs)
        
        # Check if the instance has a run method (sync)
        if hasattr(instance, 'run') and callable(instance.run):
            return instance.run(**kwargs)
        
        # If neither method exists, try to call the instance itself
        if callable(instance):
            result = instance(**kwargs)
            if inspect.iscoroutine(result):
                return await result
            return result
        
        raise ValueError(f"Unable to execute tool {self.tool_class.__name__}")

# ── Simple coffee tools ----------------------------------------------
class CoffeeTool:
    """Generic coffee preparation tool."""
    
    async def run(self):
        """Perform a coffee-related task."""
        return {"status": "completed", "message": "Coffee task completed successfully"}

# ── LLM helpers ------------------------------------------------------
LLM_SYSTEM_MSG = (
    "You are an assistant that converts a natural-language task into a JSON "
    "plan. Return ONLY valid JSON!\n"
    "Schema:\n"
    "{\n"
    '  "title": str,\n'
    '  "steps": [              // ordered list\n'
    '    {"title": str, "depends_on": [indices]},\n'
    "    ...\n"
    "  ]\n"
    "}\n"
    "Indices start at 1 in the final output."
)

async def call_llm_live(task: str) -> Dict[str, Any]:
    if not AsyncOpenAI:
        raise RuntimeError("openai package not installed")
    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_MSG},
            {"role": "user", "content": task},
        ],
    )
    return json.loads(resp.choices[0].message.content)

async def call_llm_sim(task: str) -> Dict[str, Any]:
    """
    Crude rule-based stub that returns a fixed three-step plan for any prompt.
    """
    _ = await simulate_llm_call(task)  # just to show the prompt
    return {
        "title": "Weather → Calc → Search",
        "steps": [
            {"title": "Check weather in New York",           "depends_on": []},
            {"title": "Multiply 235.5 × 18.75",              "depends_on": []},
            {"title": "Search climate-adaptation info",      "depends_on": []},
        ],
    }

# ── Simple plan execution --------------------------------------------
async def execute_step_tool_calls(graph_store: GraphStore, step_id: str) -> List[Dict]:
    """
    Execute all tool calls linked to a step.
    Returns a list of result dictionaries.
    """
    results = []
    
    # Map tool names to wrapped tool classes
    tools = {
        'weather': ToolWrapper(WeatherTool),
        'calculator': ToolWrapper(CalculatorTool),
        'search': ToolWrapper(SearchTool),
        'grind_beans': CoffeeTool(),
        'boil_water': CoffeeTool(),
        'brew_coffee': CoffeeTool(),
        'clean_station': CoffeeTool(),
    }
    
    # Find all tool calls for this step
    for edge in graph_store.get_edges(src=step_id, kind=EdgeKind.PLAN_LINK):
        tool_node = graph_store.get_node(edge.dst)
        if not tool_node or tool_node.kind.value != "tool_call":
            continue
        
        # Get tool info
        tool_name = tool_node.data.get("name")
        args = tool_node.data.get("args", {})
        
        try:
            # Get the tool
            if tool_name in tools:
                tool = tools[tool_name]
                
                # Special handling for calculator
                if tool_name == 'calculator':
                    # Fix parameters if needed - map x/y to a/b
                    if 'x' in args and 'y' in args and 'a' not in args and 'b' not in args:
                        args = {
                            'a': args['x'],
                            'b': args['y'],
                            'operation': args.get('operation', 'add')
                        }
                
                # Execute the tool
                result = await tool.run(**args)
                
                # Update the tool node with the result
                updated_tool = ToolCall(
                    id=tool_node.id,
                    data={
                        **tool_node.data,
                        "result": result
                    }
                )
                graph_store.update_node(updated_tool)
                
                # Add to results
                results.append({
                    "id": tool_node.id,
                    "tool": tool_name,
                    "args": args,
                    "result": result
                })
            else:
                # Unknown tool
                error = f"Unknown tool: {tool_name}"
                print(f"Warning: {error}")
                
                # Update the tool node with the error
                updated_tool = ToolCall(
                    id=tool_node.id,
                    data={
                        **tool_node.data,
                        "error": error
                    }
                )
                graph_store.update_node(updated_tool)
                
                # Add to results
                results.append({
                    "id": tool_node.id,
                    "tool": tool_name,
                    "args": args,
                    "error": error
                })
                
        except Exception as ex:
            print(f"Error executing {tool_name}: {ex}")
            
            # Update the tool node with the error
            updated_tool = ToolCall(
                id=tool_node.id,
                data={
                    **tool_node.data,
                    "error": str(ex)
                }
            )
            graph_store.update_node(updated_tool)
            
            # Add to results
            results.append({
                "id": tool_node.id,
                "tool": tool_name,
                "args": args,
                "error": str(ex)
            })
    
    return results

async def execute_plan_directly(graph_store: GraphStore, plan_id: str) -> List[Dict]:
    """
    Simple implementation to execute a plan directly.
    """
    # Import the PlanExecutor to get steps and execution order
    from chuk_ai_planner.planner.plan_executor import PlanExecutor
    executor = PlanExecutor(graph_store)
    
    # Get steps
    steps = executor.get_plan_steps(plan_id)
    if not steps:
        raise ValueError(f"No steps found for plan {plan_id}")
    
    # Determine execution order
    batches = executor.determine_execution_order(steps)
    all_results = []
    
    # Execute steps in order
    for batch_idx, batch in enumerate(batches):
        print(f"Executing batch {batch_idx+1}/{len(batches)}...")
        
        for step_idx, step_id in enumerate(batch):
            step = graph_store.get_node(step_id)
            print(f"  Step {step.data.get('index')}: {step.data.get('description')}")
            
            # Execute all tool calls for this step
            step_results = await execute_step_tool_calls(graph_store, step_id)
            all_results.extend(step_results)
    
    return all_results

# -------------------------------------------------------------------- main flow
async def main(live: bool) -> None:
    task = (
        "I need a short plan that first prepares coffee "
        "then checks today's weather in New York, multiplies 235.5×18.75, "
        "and finally searches for pages on climate-change adaptation."
    )

    llm_json = await (call_llm_live if live else call_llm_sim)(task)

    # 1) build Plan DSL -----------------------------------------------
    plan = Plan(llm_json["title"])
    for i, step in enumerate(llm_json["steps"], 1):
        deps = [str(d) for d in step.get("depends_on", [])]
        plan.step(step["title"], after=deps).up()
    plan_id = plan.save()

    print("\n📋  PLAN GENERATED BY LLM\n")
    print(plan.outline(), "\n")

    # 2) naive tool-mapping heuristic ---------------------------------
    title_map = {
        re.compile(r"weather", re.I):   ("weather",    {"location": "New York"}),
        re.compile(r"multiply", re.I):  ("calculator", {"operation": "multiply",
                                                       "a": 235.5, "b": 18.75}),
        re.compile(r"search", re.I):    ("search",     {"query":
                                                        "climate change adaptation"}),
        re.compile(r"grind", re.I):     ("grind_beans",  {}),
        re.compile(r"boil", re.I):      ("boil_water",   {}),
        re.compile(r"brew", re.I):      ("brew_coffee",  {}),
        re.compile(r"clean", re.I):     ("clean_station",{}),
    }

    idx2id = {n.data["index"]: n.id
              for n in plan.graph.nodes.values()
              if n.__class__.__name__ == "PlanStep"}

    for idx, node_id in idx2id.items():
        title = next(n.data["description"]
                     for n in plan.graph.nodes.values()
                     if n.id == node_id)
        for pattern, (tool, args) in title_map.items():
            if pattern.search(title):
                tc = ToolCall(data={"name": tool, "args": args})
                plan.graph.add_node(tc)
                plan.graph.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK,
                                              src=node_id, dst=tc.id))
                break

    # 3) execution ---------------------------------------------------
    # Initialize session store with proper async handling
    store = InMemorySessionStore()
    SessionStoreProvider.set_store(store)
    session = Session()
    
    # Handle potentially async save
    save_result = store.save(session)
    if asyncio.iscoroutine(save_result):
        await save_result

    # Execute plan directly
    results = await execute_plan_directly(plan.graph, plan_id)

    print("\n✅  TOOL RESULTS\n")
    for r in results:
        tool_name = r.get("tool", "unknown")
        if "error" in r:
            print(f"• {tool_name} (ERROR)\n  {r['error']}\n")
        else:
            print(f"• {tool_name}\n{json.dumps(r.get('result', {}), indent=2)}\n")

    # Display graph structure
    print_graph_structure(plan.graph)

# -------------------------------------------------------------------- entry-point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Use real OpenAI rather than the simulator")
    args = parser.parse_args()

    if args.live and not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY or omit --live")

    asyncio.run(main(args.live)