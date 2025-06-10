"""
chuk_generic_executor_fixed.py - Generic plan executor that works with Chuk AI Planner system

This file demonstrates how to integrate our generic, domain-agnostic approach
with the existing Chuk AI Planner system, including LLM-driven plan generation.
"""

import asyncio
import json
import os
from typing import Dict, Any, List, Optional, Callable, Awaitable

# Chuk AI Planner imports
from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.models import ToolCall
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.processor import GraphAwareToolProcessor
from chuk_ai_planner.utils.visualization import print_session_events, print_graph_structure

# A2A session manager imports 
from a2a_session_manager.storage import InMemorySessionStore, SessionStoreProvider
from a2a_session_manager.models.session import Session

# OpenAI for LLM integration
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# ----------------------------------------------------------------
# Plan Database - Registry for storing and retrieving plans
# ----------------------------------------------------------------

class PlanDatabase:
    """
    Plan database that stores plans and tools.
    
    This works with the Chuk AI Planner system structure but provides
    a generic interface for storing and retrieving plans.
    """
    
    def __init__(self):
        """Initialize an empty plan database"""
        # Plans stored by ID
        self.plans = {}
        # Tools registry
        self.tools = {}
        # Shared graph store
        self.graph_store = InMemoryGraphStore()
    
    def store_plan(self, plan: UniversalPlan) -> str:
        """
        Store a plan in the database
        
        Parameters
        ----------
        plan : UniversalPlan
            The plan to store
            
        Returns
        -------
        str
            The ID of the stored plan
        """
        # Ensure the plan is saved to the graph store
        if not plan._indexed:
            plan.save()
        
        # Store the plan in our registry
        self.plans[plan.id] = plan
        
        return plan.id
    
    def get_plan(self, plan_id: str) -> Optional[UniversalPlan]:
        """
        Retrieve a plan from the database
        
        Parameters
        ----------
        plan_id : str
            ID of the plan to retrieve
            
        Returns
        -------
        Optional[UniversalPlan]
            The plan, or None if not found
        """
        return self.plans.get(plan_id)
    
    def list_plans(self, tag: Optional[str] = None) -> List[str]:
        """
        List all plan IDs, optionally filtered by tag
        
        Parameters
        ----------
        tag : Optional[str]
            Tag to filter plans by
            
        Returns
        -------
        List[str]
            List of plan IDs
        """
        if tag:
            return [
                plan_id for plan_id, plan in self.plans.items()
                if tag in plan.tags
            ]
        return list(self.plans.keys())
    
    def register_tool(self, tool_name: str, tool_fn: Callable) -> None:
        """
        Register a tool function
        
        Parameters
        ----------
        tool_name : str
            Name of the tool
        tool_fn : Callable
            Function that implements the tool
        """
        self.tools[tool_name] = tool_fn
        print(f"Registered tool: {tool_name}")
    
    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """
        Get a tool function by name
        
        Parameters
        ----------
        tool_name : str
            Name of the tool
            
        Returns
        -------
        Optional[Callable]
            The tool function, or None if not found
        """
        return self.tools.get(tool_name)


# ----------------------------------------------------------------
# Generic Executor - Works with Chuk AI Planner system
# ----------------------------------------------------------------

class GenericExecutor:
    """
    Generic executor that can run any plan from the database
    
    This integrates the generic approach with the Chuk AI Planner system.
    """
    
    def __init__(self, plan_db: PlanDatabase):
        """
        Initialize the executor
        
        Parameters
        ----------
        plan_db : PlanDatabase
            Database to retrieve plans and tools from
        """
        self.plan_db = plan_db
        
        # Set up session for Chuk AI Planner
        SessionStoreProvider.set_store(InMemorySessionStore())
        self.session = Session()
        SessionStoreProvider.get_store().save(self.session)
        
        # Create processor
        self.processor = GraphAwareToolProcessor(
            self.session.id,
            self.plan_db.graph_store
        )
        
        # Register tools with the processor
        for tool_name, tool_fn in self.plan_db.tools.items():
            self.processor.register_tool(tool_name, tool_fn)
    
    async def execute_plan(
        self, 
        plan_id: str, 
        variables: Optional[Dict[str, Any]] = None,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a plan from the database
        
        Parameters
        ----------
        plan_id : str
            ID of the plan to execute
        variables : Optional[Dict[str, Any]]
            Initial variables for the plan (not used directly by processor,
            but could be used to set plan variables before execution)
        debug : bool
            Whether to print debug information
            
        Returns
        -------
        Dict[str, Any]
            Execution results
        """
        # Get the plan
        plan = self.plan_db.get_plan(plan_id)
        if not plan:
            return {"success": False, "error": f"Plan not found: {plan_id}"}
        
        print(f"Executing plan: {plan.title}")
        
        # If variables are provided, set them in the plan
        if variables:
            for key, value in variables.items():
                plan.set_variable(key, value)
                
            # Make sure to save the plan with updated variables
            plan.save()
        
        # Execute the plan
        try:
            # Use the Chuk AI processor to execute the plan
            results = await self.processor.process_plan(
                plan_node_id=plan_id,
                assistant_node_id="assistant",
                llm_call_fn=lambda _: None
            )
            
            if debug:
                # Print debug information
                print_session_events(self.session)
                print_graph_structure(self.plan_db.graph_store)
            
            # Extract variables from results
            variables = {}
            for result in results:
                # Get tool name
                tool_name = result.tool
                
                # Store result in variables
                var_name = f"result_{tool_name}"
                variables[var_name] = result.result
            
            return {
                "success": True,
                "plan_id": plan_id,
                "results": results,
                "variables": variables
            }
            
        except Exception as e:
            print(f"Error executing plan {plan_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "plan_id": plan_id
            }


# ----------------------------------------------------------------
# LLM-Driven Plan Generation
# ----------------------------------------------------------------

async def generate_plan_from_llm(
    task: str, 
    system_msg: Optional[str] = None,
    available_tools: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
    use_sim: bool = False
) -> Dict[str, Any]:
    """
    Generate a plan from a natural language task using LLM
    
    Parameters
    ----------
    task : str
        Natural language task description
    system_msg : Optional[str]
        System message for the LLM, or None to use default
    available_tools : Optional[List[str]]
        List of available tools, or None to use default
    model : str
        LLM model to use
    use_sim : bool
        Whether to use simulated LLM response instead of real API
        
    Returns
    -------
    Dict[str, Any]
        LLM-generated plan in JSON format
    """
    if use_sim:
        # Return a simulated response
        return {
            "title": "Generated Task Plan",
            "steps": [
                {
                    "title": "Check the weather in London",
                    "tool": "weather",
                    "args": {"location": "London"},
                    "depends_on": []
                },
                {
                    "title": "Calculate 123.45 * 67.89",
                    "tool": "calculator",
                    "args": {"operation": "multiply", "a": 123.45, "b": 67.89},
                    "depends_on": []
                },
                {
                    "title": "Search for renewable energy sources",
                    "tool": "search",
                    "args": {"query": "renewable energy sources"},
                    "depends_on": []
                }
            ]
        }
    
    # Build the system message
    if system_msg is None:
        tools_str = ", ".join(available_tools) if available_tools else "search, calculator, weather"
        system_msg = (
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
            f"Indices start at 1 in the final output.\n"
            f"Available tools: {tools_str}\n"
            "The 'tool' field is required and must be one of the available tools.\n"
            "The 'args' field should contain the appropriate arguments for the tool."
        )
    
    if not AsyncOpenAI:
        raise RuntimeError("OpenAI package not installed")
    
    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": task},
        ],
    )
    
    return json.loads(resp.choices[0].message.content)


def convert_to_universal_plan(
    llm_json: Dict[str, Any], 
    graph_store: Optional[InMemoryGraphStore] = None
) -> UniversalPlan:
    """
    Convert LLM-generated JSON to a UniversalPlan
    
    Parameters
    ----------
    llm_json : Dict[str, Any]
        LLM-generated plan
    graph_store : Optional[InMemoryGraphStore]
        Graph store to use, or None to create a new one
        
    Returns
    -------
    UniversalPlan
        Created plan
    """
    # Create a new universal plan
    plan = UniversalPlan(
        title=llm_json["title"],
        description="Generated from LLM input",
        tags=["llm-generated"],
        graph=graph_store
    )
    
    # Add metadata about the source
    plan.add_metadata("source", "llm")
    plan.add_metadata("generation_time", str(asyncio.get_running_loop().time()))
    
    # Create a mapping of LLM step index to Plan step ID
    step_ids = {}
    
    # First pass: Create all steps without tool links
    for i, step_data in enumerate(llm_json["steps"], 1):
        title = step_data["title"]
        step_index = plan.add_step(title, parent=None)
        
        # Get the step node
        step_id = None
        for node in plan._graph.nodes.values():
            if node.__class__.__name__ == "PlanStep" and node.data.get("index") == step_index:
                step_id = node.id
                break
        
        if step_id:
            step_ids[i] = step_id
    
    # Now add tool calls and dependencies
    for i, step_data in enumerate(llm_json["steps"], 1):
        step_id = step_ids.get(i)
        if not step_id:
            continue
            
        # Create tool call
        tool = step_data.get("tool")
        args = step_data.get("args", {})
        
        if tool:
            # Create and link tool call
            tool_call = ToolCall(data={"name": tool, "args": args})
            plan._graph.add_node(tool_call)
            plan._graph.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step_id, dst=tool_call.id))
            
            # Store result variable using a custom edge
            result_var = f"result_{i}"
            plan._graph.add_edge(GraphEdge(
                kind=EdgeKind.CUSTOM,
                src=step_id,
                dst=tool_call.id,
                data={"type": "result_variable", "variable": result_var}
            ))
        
        # Add dependencies
        for dep_idx in step_data.get("depends_on", []):
            dep_id = step_ids.get(dep_idx)
            if dep_id:
                plan._graph.add_edge(GraphEdge(
                    kind=EdgeKind.STEP_ORDER,
                    src=dep_id,
                    dst=step_id
                ))
    
    return plan


# ----------------------------------------------------------------
# End-to-End Demo Implementation
# ----------------------------------------------------------------

class ToolExecutorFactory:
    """Factory for creating tool executors"""
    
    @staticmethod
    def create_executor(name: str) -> Callable:
        """Create an executor for a tool"""
        async def executor(args):
            print(f"Executing tool: {name} with args: {args}")
            
            if name == "search":
                query = args.get("query", "unknown")
                return {
                    "results": [
                        {"title": f"Result for {query}", "url": f"https://example.com/search?q={query}"},
                        {"title": f"Another result for {query}", "url": f"https://example.com/search?q={query}&page=2"}
                    ],
                    "query": query
                }
            
            elif name == "calculator":
                operation = args.get("operation", "")
                a = float(args.get("a", 0))
                b = float(args.get("b", 0))
                
                result = 0
                if operation == "add":
                    result = a + b
                elif operation == "subtract":
                    result = a - b
                elif operation == "multiply":
                    result = a * b
                elif operation == "divide":
                    result = a / b if b != 0 else "Error: Division by zero"
                
                return {"result": result, "operation": operation}
            
            elif name == "weather":
                location = args.get("location", "Unknown")
                return {
                    "temperature": 72,
                    "conditions": "Partly cloudy",
                    "humidity": 65, 
                    "location": location
                }
            
            elif name == "database_query":
                query = args.get("query", "")
                return {
                    "results": [
                        {"id": 1, "name": "Sample Result 1"},
                        {"id": 2, "name": "Sample Result 2"}
                    ],
                    "count": 2,
                    "query": query
                }
            
            elif name == "translate":
                text = args.get("text", "")
                target_lang = args.get("target_language", "en")
                return {
                    "original": text,
                    "translated": f"Translated {text} to {target_lang}",
                    "language": target_lang
                }
            
            # Default fallback
            return {"status": f"Executed {name}", "args": args}
        
        return executor


async def run_llm_plan_demo():
    """Run a demo of the LLM-driven plan generation and execution"""
    print("=== LLM-DRIVEN PLAN GENERATION AND EXECUTION DEMO ===")
    
    # Define the natural language task
    task = (
        "I need a plan that checks the weather in London, "
        "calculates 123.45 * 67.89, and searches for information "
        "about renewable energy sources."
    )
    
    print(f"\nTask: {task}")
    
    # Define available tools
    available_tools = ["search", "calculator", "weather", "database_query", "translate"]
    
    # Create plan database
    db = PlanDatabase()
    
    # Register tools with the database
    tool_factory = ToolExecutorFactory()
    for tool_name in available_tools:
        db.register_tool(tool_name, tool_factory.create_executor(tool_name))
    
    # Create executor
    executor = GenericExecutor(db)
    
    # Generate plan from LLM
    print("\nGenerating plan from LLM...")
    try:
        # Try to use real LLM
        use_sim = not (AsyncOpenAI and os.getenv("OPENAI_API_KEY"))
        if use_sim:
            print("(Using simulated LLM response - no API key found)")
        
        llm_json = await generate_plan_from_llm(
            task,
            available_tools=available_tools,
            use_sim=use_sim
        )
        
        print("\nLLM-generated plan:")
        print(json.dumps(llm_json, indent=2))
        
        # Convert to UniversalPlan
        print("\nConverting to UniversalPlan...")
        plan = convert_to_universal_plan(llm_json, db.graph_store)
        
        # Store the plan in the database
        db.store_plan(plan)
        
        # Print plan outline
        print("\nPlan structure:")
        print(plan.outline())
        
        # Execute the plan
        print("\nExecuting the plan...")
        result = await executor.execute_plan(plan.id, debug=True)
        
        # Print results
        if result.get("success", False):
            print("\nExecution successful!")
            print("\nResults:")
            for r in result.get("results", []):
                print(f"• {r.tool}:")
                print(json.dumps(r.result, indent=2))
        else:
            print(f"\nExecution failed: {result.get('error')}")
    
    except Exception as e:
        print(f"Error in LLM plan demo: {e}")


# ----------------------------------------------------------------
# Direct Plan Creation Demo
# ----------------------------------------------------------------

async def run_direct_plan_demo():
    """Run a demo with directly created plans (no LLM)"""
    print("\n=== DIRECT PLAN CREATION AND EXECUTION DEMO ===")
    
    # Create plan database
    db = PlanDatabase()
    
    # Register tools
    tool_factory = ToolExecutorFactory()
    db.register_tool("search", tool_factory.create_executor("search"))
    db.register_tool("calculator", tool_factory.create_executor("calculator"))
    db.register_tool("weather", tool_factory.create_executor("weather"))
    
    # Create a plan directly
    print("\nCreating plan directly...")
    plan = UniversalPlan(
        title="Research and Calculate Plan",
        description="A plan to search for information and perform calculations",
        tags=["research", "calculation"],
        graph=db.graph_store
    )
    
    # Add steps
    s1 = plan.add_tool_step(
        title="Search for renewable energy",
        tool="search",
        args={"query": "renewable energy benefits"},
        result_variable="search_results"
    )
    
    s2 = plan.add_tool_step(
        title="Calculate energy efficiency",
        tool="calculator",
        args={"operation": "multiply", "a": 0.85, "b": 100},
        result_variable="efficiency_calculation"
    )
    
    s3 = plan.add_tool_step(
        title="Check weather in solar farm location",
        tool="weather",
        args={"location": "Phoenix, AZ"},
        result_variable="weather_data",
        depends_on=[s1, s2]
    )
    
    # Save and store the plan
    plan.save()
    db.store_plan(plan)
    
    # Print plan structure
    print("\nPlan structure:")
    print(plan.outline())
    
    # Create executor
    executor = GenericExecutor(db)
    
    # Execute the plan
    print("\nExecuting the plan...")
    result = await executor.execute_plan(plan.id)
    
    # Print results
    if result.get("success", False):
        print("\nExecution successful!")
        print("\nResults:")
        for r in result.get("results", []):
            print(f"• {r.tool}:")
            print(json.dumps(r.result, indent=2))
    else:
        print(f"\nExecution failed: {result.get('error')}")


# ----------------------------------------------------------------
# Main Demo Runner
# ----------------------------------------------------------------

async def main():
    """Run all demos"""
    # Run LLM-driven plan demo
    await run_llm_plan_demo()
    
    # Run direct plan creation demo
    await run_direct_plan_demo()


if __name__ == "__main__":
    asyncio.run(main())