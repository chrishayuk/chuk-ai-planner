# src/chuk_ai_planner/planner/universal_plan_executor.py
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable

# planner
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.processor import GraphAwareToolProcessor

# plan
from .universal_plan import UniversalPlan
from .plan_executor import PlanExecutor

class UniversalExecutor:
    """
    Enhanced executor that works with UniversalPlan and integrates with GraphAwareToolProcessor
    """
    
    def __init__(self):
        # Create session and graph store
        from a2a_session_manager.storage import InMemorySessionStore, SessionStoreProvider
        from a2a_session_manager.models.session import Session
        
        try:
            SessionStoreProvider.get_store()
        except:
            SessionStoreProvider.set_store(InMemorySessionStore())
        
        self.session = Session()
        SessionStoreProvider.get_store().save(self.session)
        
        # Create graph store and processor
        self.graph_store = InMemoryGraphStore()
        self.processor = GraphAwareToolProcessor(
            self.session.id,
            self.graph_store,
            enable_caching=True,
            enable_retries=True
        )
        
        # Create plan executor
        self.plan_executor = PlanExecutor(self.graph_store)
        
        # Registry for tools and functions
        self.tool_registry = {}
        self.function_registry = {}
        
        # Create assistant node for tool processing
        self.assistant_node_id = str(uuid.uuid4())
    
    def register_tool(self, name: str, fn: Callable) -> None:
        """Register a tool with the executor"""
        self.tool_registry[name] = fn
        self.processor.register_tool(name, fn)
    
    def register_function(self, name: str, fn: Callable) -> None:
        """Register a function with the executor"""
        self.function_registry[name] = fn
        
        # For functions, we also create a wrapper tool that the processor can use
        async def function_wrapper(args):
            function_name = args.get("function")
            function_args = args.get("args", {})
            
            if function_name not in self.function_registry:
                raise ValueError(f"Unknown function: {function_name}")
                
            func = self.function_registry[function_name]
            
            # Handle both async and sync functions
            if asyncio.iscoroutinefunction(func):
                return await func(**function_args)
            else:
                return func(**function_args)
        
        self.processor.register_tool("function", function_wrapper)
    
    async def execute_plan(self, plan: UniversalPlan, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a UniversalPlan with variable resolution"""
        # Ensure the plan is saved
        if not plan._indexed:
            plan.save()
        
        # Initialize execution context with variables
        context = {
            "variables": variables or {},
            "results": {}
        }
        
        # Merge plan variables with provided variables
        for name, value in plan.variables.items():
            if name not in context["variables"]:
                context["variables"][name] = value
        
        # Define LLM call function (not actually used in plan execution)
        async def mock_llm_call_fn(prompt):
            return {"content": "Mock response", "tool_calls": []}
        
        # Execute the plan using the processor
        try:
            tool_results = await self.processor.process_plan(
                plan.id, 
                self.assistant_node_id,
                mock_llm_call_fn,
                on_step=lambda step_id, results: self._process_step_results(step_id, results, context)
            )
            
            # Process final results
            return {
                "success": True,
                "results": context["results"],
                "variables": context["variables"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": context["results"],
                "variables": context["variables"]
            }
    
    def _process_step_results(self, step_id, results, context):
        """Process results from a step execution and update the context"""
        # Store results in the context
        context["results"][step_id] = results
        
        # Extract variables based on result_variable edges
        for edge in self.graph_store.get_edges(src=step_id, kind="RESULT_VARIABLE"):
            variable_name = edge.data.get("variable")
            if variable_name and results:
                # For simplicity, use the first result if there are multiple
                result = results[0] if isinstance(results, list) else results
                context["variables"][variable_name] = result.get("result", result)
        
        # Continue execution
        return True
    
    async def execute_plan_by_id(self, plan_id: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a plan by ID"""
        # Find the plan node
        plan_node = None
        for node in self.graph_store.nodes.values():
            if node.id == plan_id:
                plan_node = node
                break
        
        if not plan_node:
            raise ValueError(f"Plan not found: {plan_id}")
        
        # Create a temporary UniversalPlan wrapper
        plan = UniversalPlan(
            title=plan_node.data.get("title", "Untitled Plan"),
            id=plan_id,
            graph=self.graph_store
        )
        
        # Execute the plan
        return await self.execute_plan(plan, variables)