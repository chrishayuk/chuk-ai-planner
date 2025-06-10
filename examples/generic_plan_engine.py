"""
generic_plan_engine.py - Domain-agnostic plan executor

This module demonstrates a completely generic plan execution system that
can run any type of plan stored in a database without knowing anything
about the domain specifics.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

class PlanDatabase:
    """
    Generic plan database that stores and retrieves plans
    
    This is a simplified in-memory implementation, but in a real system
    this would connect to a database like MongoDB, PostgreSQL, etc.
    """
    
    def __init__(self):
        """Initialize an empty plan database"""
        # In-memory storage for plans
        self.plans = {}
        # Tool registry
        self.tools = {}
    
    def store_plan(self, plan_id: str, plan_data: Dict[str, Any]) -> None:
        """
        Store a plan in the database
        
        Parameters
        ----------
        plan_id : str
            Unique identifier for the plan
        plan_data : Dict[str, Any]
            Complete plan data including steps, metadata, etc.
        """
        self.plans[plan_id] = plan_data
        logger.info(f"Stored plan: {plan_id}")
    
    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a plan from the database
        
        Parameters
        ----------
        plan_id : str
            ID of the plan to retrieve
            
        Returns
        -------
        Optional[Dict[str, Any]]
            The plan data, or None if not found
        """
        plan = self.plans.get(plan_id)
        if not plan:
            logger.warning(f"Plan not found: {plan_id}")
        return plan
    
    def list_plans(self, domain: Optional[str] = None) -> List[str]:
        """
        List all plan IDs, optionally filtered by domain
        
        Parameters
        ----------
        domain : Optional[str]
            Domain to filter plans by
            
        Returns
        -------
        List[str]
            List of plan IDs
        """
        if domain:
            return [
                plan_id for plan_id, plan in self.plans.items()
                if domain in plan.get("domains", [])
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
        logger.info(f"Registered tool: {tool_name}")
    
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
        tool = self.tools.get(tool_name)
        if not tool:
            logger.warning(f"Tool not found: {tool_name}")
        return tool

class GenericPlanExecutor:
    """
    Generic plan executor that can run any plan from the database
    
    This executor knows nothing about the domain specifics - it simply
    executes steps, resolves variables, and handles dependencies based
    on the plan structure.
    """
    
    def __init__(self, plan_db: PlanDatabase):
        """
        Initialize the plan executor
        
        Parameters
        ----------
        plan_db : PlanDatabase
            Database to retrieve plans and tools from
        """
        self.plan_db = plan_db
    
    async def execute_plan(
        self, 
        plan_id: str, 
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a plan from the database
        
        Parameters
        ----------
        plan_id : str
            ID of the plan to execute
        variables : Optional[Dict[str, Any]]
            Initial variables for the plan
            
        Returns
        -------
        Dict[str, Any]
            Execution results
        """
        # Get the plan
        plan = self.plan_db.get_plan(plan_id)
        if not plan:
            return {"success": False, "error": f"Plan not found: {plan_id}"}
        
        logger.info(f"Executing plan: {plan.get('title', plan_id)}")
        
        # Initialize variables
        context = {
            "variables": {**(variables or {}), **(plan.get("variables", {}))},
            "results": {}
        }
        
        try:
            # Get steps from the plan
            steps = plan.get("steps", [])
            
            # Determine execution order based on dependencies
            execution_order = self._determine_execution_order(steps)
            
            # Execute steps in order
            for step_batch in execution_order:
                # Execute steps in this batch in parallel
                tasks = [
                    self._execute_step(steps[step_idx], context)
                    for step_idx in step_batch
                ]
                await asyncio.gather(*tasks)
            
            return {
                "success": True,
                "plan_id": plan_id,
                "variables": context["variables"],
                "results": context["results"]
            }
        
        except Exception as e:
            logger.error(f"Error executing plan {plan_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "plan_id": plan_id
            }
    
    def _determine_execution_order(self, steps: List[Dict[str, Any]]) -> List[List[int]]:
        """
        Determine the execution order of steps based on dependencies
        
        This uses a topological sort to create batches of steps that can
        be executed in parallel.
        
        Parameters
        ----------
        steps : List[Dict[str, Any]]
            List of steps
            
        Returns
        -------
        List[List[int]]
            List of batches, where each batch is a list of step indices
        """
        # Build dependency graph
        dependencies = {}
        dependents = {}
        
        for i, step in enumerate(steps):
            deps = step.get("depends_on", [])
            # Map step IDs to indices if needed
            if deps and isinstance(deps[0], str):
                # Convert step IDs to indices
                deps = [
                    j for j, s in enumerate(steps)
                    if s.get("id") in deps
                ]
            
            dependencies[i] = set(deps)
            
            # Build reverse mapping
            for dep in deps:
                if dep not in dependents:
                    dependents[dep] = set()
                dependents[dep].add(i)
        
        # Initialize steps with no dependencies
        ready = [i for i, deps in dependencies.items() if not deps]
        
        # Build execution batches
        batches = []
        while ready:
            batches.append(ready)
            
            # Find steps that become ready after this batch
            next_ready = []
            for step_idx in ready:
                for dependent in dependents.get(step_idx, set()):
                    dependencies[dependent].discard(step_idx)
                    if not dependencies[dependent]:
                        next_ready.append(dependent)
            
            ready = next_ready
        
        return batches
    
    async def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Execute a single step
        
        Parameters
        ----------
        step : Dict[str, Any]
            Step to execute
        context : Dict[str, Any]
            Execution context with variables
            
        Returns
        -------
        Any
            Step result
        """
        step_id = step.get("id")
        step_type = step.get("type", "logic")
        step_title = step.get("title", f"Step {step_id}")
        
        logger.info(f"Executing step: {step_title} (type: {step_type})")
        
        if step_type == "tool":
            # Execute tool step
            result = await self._execute_tool_step(step, context)
        elif step_type == "subplan":
            # Execute subplan step
            result = await self._execute_subplan_step(step, context)
        else:
            # Execute logic step
            result = self._execute_logic_step(step, context)
        
        # Store result
        if step_id:
            context["results"][step_id] = result
        
        # Update variables if result_variable is specified
        result_var = step.get("result_variable")
        if result_var:
            context["variables"][result_var] = result
        
        return result
    
    async def _execute_tool_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Execute a tool step
        
        Parameters
        ----------
        step : Dict[str, Any]
            Tool step to execute
        context : Dict[str, Any]
            Execution context
            
        Returns
        -------
        Any
            Tool result
        """
        tool_name = step.get("tool")
        args = step.get("args", {})
        
        # Resolve variables in args
        resolved_args = self._resolve_variables(args, context["variables"])
        
        # Get the tool function
        tool_fn = self.plan_db.get_tool(tool_name)
        if not tool_fn:
            raise ValueError(f"Tool not found: {tool_name}")
        
        # Execute the tool
        if asyncio.iscoroutinefunction(tool_fn):
            result = await tool_fn(resolved_args)
        else:
            result = tool_fn(resolved_args)
        
        return result
    
    async def _execute_subplan_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Execute a subplan step
        
        Parameters
        ----------
        step : Dict[str, Any]
            Subplan step to execute
        context : Dict[str, Any]
            Execution context
            
        Returns
        -------
        Any
            Subplan result
        """
        plan_id = step.get("plan_id")
        args = step.get("args", {})
        
        # Resolve variables in args
        resolved_args = self._resolve_variables(args, context["variables"])
        
        # Execute the subplan
        result = await self.execute_plan(plan_id, resolved_args)
        
        return result
    
    def _execute_logic_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Execute a logic step
        
        This is for steps that don't execute tools or subplans, but still
        need to be tracked in the plan structure.
        
        Parameters
        ----------
        step : Dict[str, Any]
            Logic step to execute
        context : Dict[str, Any]
            Execution context
            
        Returns
        -------
        Any
            Logic step result
        """
        step_id = step.get("id", "unknown")
        metadata = step.get("metadata", {})
        
        # Simply return a status object - in a real system, this might
        # do more complex processing
        return {
            "step_id": step_id,
            "executed": True,
            "step_type": "logic",
            "metadata": metadata
        }
    
    def _resolve_variables(self, template: Any, variables: Dict[str, Any]) -> Any:
        """
        Resolve variables in a template
        
        Parameters
        ----------
        template : Any
            Template with possible variable references
        variables : Dict[str, Any]
            Dictionary of variables
            
        Returns
        -------
        Any
            Template with variables resolved
        """
        if isinstance(template, str):
            # Handle pattern ${varname} or ${varname.attr1.attr2}
            if template.startswith("${") and template.endswith("}"):
                var_path = template[2:-1]
                return self._get_nested_value(var_path, variables)
            
            # Replace ${varname} patterns within the string
            import re
            def replace_var(match):
                var_path = match.group(1)
                value = self._get_nested_value(var_path, variables)
                # Convert to string if needed
                if not isinstance(value, str):
                    value = str(value)
                return value
            
            return re.sub(r'\${([^}]+)}', replace_var, template)
        
        elif isinstance(template, dict):
            return {k: self._resolve_variables(v, variables) for k, v in template.items()}
        
        elif isinstance(template, list):
            return [self._resolve_variables(item, variables) for item in template]
        
        return template
    
    def _get_nested_value(self, path: str, variables: Dict[str, Any]) -> Any:
        """
        Get a value from nested dictionaries using a dot-separated path
        
        Parameters
        ----------
        path : str
            Dot-separated path (e.g., "user.address.city")
        variables : Dict[str, Any]
            Dictionary of variables
            
        Returns
        -------
        Any
            Value at the path
        """
        parts = path.split(".")
        value = variables
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                # Return None if the path doesn't exist
                return None
        
        return value

# Example usage
async def run_example():
    # Create a plan database
    db = PlanDatabase()
    
    # Register example tools
    db.register_tool("add_numbers", lambda args: {"result": args.get("a", 0) + args.get("b", 0)})
    
    async def fetch_data(args):
        # Simulate a network request
        await asyncio.sleep(0.5)
        return {"data": f"Data for {args.get('id', 'unknown')}"}
    
    db.register_tool("fetch_data", fetch_data)
    
    async def process_data(args):
        # Simulate data processing
        await asyncio.sleep(0.5)
        return {"processed": True, "original": args.get("data"), "length": len(str(args.get("data", "")))}
    
    db.register_tool("process_data", process_data)
    
    # Store some example plans
    
    # 1. Simple math plan
    math_plan = {
        "id": "math_plan_001",
        "title": "Basic Math Operations",
        "description": "Performs basic math operations",
        "domains": ["math", "calculation"],
        "variables": {"default_value": 100},
        "steps": [
            {
                "id": "step1",
                "type": "tool",
                "title": "Add two numbers",
                "tool": "add_numbers",
                "args": {"a": 5, "b": 10},
                "result_variable": "addition_result"
            },
            {
                "id": "step2",
                "type": "tool",
                "title": "Add with a variable",
                "tool": "add_numbers",
                "args": {"a": "${addition_result.result}", "b": "${default_value}"},
                "result_variable": "final_result",
                "depends_on": ["step1"]
            }
        ]
    }
    
    db.store_plan("math_plan_001", math_plan)
    
    # 2. Data processing plan
    data_plan = {
        "id": "data_plan_001",
        "title": "Data Fetching and Processing",
        "description": "Fetches and processes data",
        "domains": ["data", "processing"],
        "variables": {"target_id": "default_id"},
        "steps": [
            {
                "id": "fetch_step",
                "type": "tool",
                "title": "Fetch data",
                "tool": "fetch_data",
                "args": {"id": "${target_id}"},
                "result_variable": "fetched_data"
            },
            {
                "id": "process_step",
                "type": "tool",
                "title": "Process data",
                "tool": "process_data",
                "args": {"data": "${fetched_data.data}"},
                "result_variable": "processed_data",
                "depends_on": ["fetch_step"]
            }
        ]
    }
    
    db.store_plan("data_plan_001", data_plan)
    
    # 3. Composite plan using a subplan
    composite_plan = {
        "id": "composite_plan_001",
        "title": "Composite Plan with Subplan",
        "description": "Demonstrates using a subplan",
        "domains": ["composite", "demo"],
        "variables": {"input_value": "test_data"},
        "steps": [
            {
                "id": "logic_step",
                "type": "logic",
                "title": "Prepare for processing",
                "metadata": {"purpose": "setup"}
            },
            {
                "id": "data_step",
                "type": "subplan",
                "title": "Process data using subplan",
                "plan_id": "data_plan_001",
                "args": {"target_id": "${input_value}"},
                "result_variable": "data_results",
                "depends_on": ["logic_step"]
            },
            {
                "id": "math_step",
                "type": "subplan",
                "title": "Perform math operations",
                "plan_id": "math_plan_001",
                "args": {"default_value": 50},
                "result_variable": "math_results"
            },
            {
                "id": "final_step",
                "type": "logic",
                "title": "Combine results",
                "metadata": {"purpose": "integration"},
                "depends_on": ["data_step", "math_step"]
            }
        ]
    }
    
    db.store_plan("composite_plan_001", composite_plan)
    
    # Create a plan executor
    executor = GenericPlanExecutor(db)
    
    # Execute plans
    print("\n=== Executing Math Plan ===")
    math_result = await executor.execute_plan("math_plan_001")
    print(f"Math Plan Result: {json.dumps(math_result, indent=2)}")
    
    print("\n=== Executing Data Plan ===")
    data_result = await executor.execute_plan("data_plan_001", {"target_id": "customer_123"})
    print(f"Data Plan Result: {json.dumps(data_result, indent=2)}")
    
    print("\n=== Executing Composite Plan ===")
    composite_result = await executor.execute_plan("composite_plan_001", {"input_value": "special_case"})
    print(f"Composite Plan Result: {json.dumps(composite_result, indent=2)}")

# Run the example
if __name__ == "__main__":
    asyncio.run(run_example())