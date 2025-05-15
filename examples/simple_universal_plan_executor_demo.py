#!/usr/bin/env python
# examples/universal_executor_demo.py
"""
Universal Executor Demo – final working version
==============================================

• Simple, standalone executor that properly handles variable dependencies
• Sequential execution that respects step ordering
• Direct function execution with no intermediate layers
• Proper variable resolution
"""

import asyncio
import json
import pprint
from typing import Any, Dict, List

from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.models.edges import EdgeKind


# --------------------------------------------------------------------------- custom executor
class SimpleExecutor:
    """A simplified executor that handles variable dependencies correctly."""
    
    def __init__(self):
        self.graph_store = None
        self.tool_registry = {}
        self.function_registry = {}
    
    def register_tool(self, name, fn):
        self.tool_registry[name] = fn
    
    def register_function(self, name, fn):
        self.function_registry[name] = fn
    
    def _resolve_vars(self, value, variables):
        """Resolve any variable references in value using the variables dict."""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]  # Remove ${ and }
            if var_name in variables:
                print(f"DEBUG: Resolved ${{{var_name}}} -> {type(variables[var_name])}")
                return variables[var_name]
            print(f"DEBUG: Variable not found: {var_name}")
            return value
        
        if isinstance(value, dict):
            return {k: self._resolve_vars(v, variables) for k, v in value.items()}
        
        if isinstance(value, list):
            return [self._resolve_vars(item, variables) for item in value]
        
        return value
    
    async def execute_plan(self, plan: UniversalPlan, variables=None):
        """Execute the plan sequentially with variable substitution."""
        self.graph_store = plan.graph
        
        # Initialize context with plan variables and any provided variables
        context = {
            "variables": {**plan.variables, **(variables or {})},
            "results": {}
        }
        
        # Debug plan variables
        print("\nDEBUG: Plan variables:")
        for k, v in context["variables"].items():
            print(f"  {k}: {type(v)}")
            if isinstance(v, (list, dict)):
                print(f"    Value: {v}")
        
        try:
            # Get all steps
            steps = []
            for node in self.graph_store.nodes.values():
                if node.kind.value == "plan_step":
                    steps.append(node)
            
            # Build dependency map
            step_dependencies = {}
            for step in steps:
                deps = set()
                # Get explicit dependencies from STEP_ORDER edges
                for edge in self.graph_store.get_edges(dst=step.id, kind=EdgeKind.STEP_ORDER):
                    deps.add(edge.src)
                step_dependencies[step.id] = deps
            
            # Sort steps topologically - this gives us a safe execution order
            sorted_steps = self._topological_sort(steps, step_dependencies)
            
            # Execute steps in order
            for step in sorted_steps:
                print(f"\nDEBUG: Executing step: {step.data.get('description')}")
                await self._execute_step(step, context)
            
            return {"success": True, **context}
        
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(exc), **context}
    
    def _topological_sort(self, steps, dependencies):
        """Sort steps based on dependencies."""
        # Create a mapping from step ID to step object
        id_to_step = {step.id: step for step in steps}
        
        # Track visited and temp markers for cycle detection
        visited = set()
        temp_mark = set()
        
        # Result list
        sorted_steps = []
        
        def visit(step_id):
            if step_id in temp_mark:
                raise ValueError(f"Dependency cycle detected involving step {step_id}")
            
            if step_id not in visited:
                temp_mark.add(step_id)
                
                # Visit dependencies
                for dep_id in dependencies.get(step_id, set()):
                    visit(dep_id)
                
                temp_mark.remove(step_id)
                visited.add(step_id)
                
                # Add to result
                if step_id in id_to_step:
                    sorted_steps.append(id_to_step[step_id])
        
        # Visit all steps
        for step in steps:
            if step.id not in visited:
                visit(step.id)
        
        # Reverse the list to get the correct order
        return sorted_steps
    
    async def _execute_step(self, step, context):
        """Execute a single step and update context."""
        step_id = step.id
        
        # Find tool calls for this step
        results = []
        for edge in self.graph_store.get_edges(src=step_id, kind=EdgeKind.PLAN_LINK):
            tool_node = self.graph_store.get_node(edge.dst)
            if tool_node and tool_node.kind.value == "tool_call":
                # Get tool info
                tool_name = tool_node.data.get("name")
                args = tool_node.data.get("args", {})
                
                # Resolve variables in args
                resolved_args = self._resolve_vars(args, context["variables"])
                print(f"DEBUG: {tool_name} with resolved args: {resolved_args}")
                
                # Execute the appropriate function
                if tool_name == "function":
                    # Handle function calls
                    fn_name = resolved_args.get("function")
                    fn_args = resolved_args.get("args", {})
                    
                    # Ensure function args are fully resolved
                    fn_args = self._resolve_vars(fn_args, context["variables"])
                    print(f"DEBUG: Function {fn_name} with resolved args: {fn_args}")
                    
                    fn = self.function_registry.get(fn_name)
                    if fn is None:
                        raise ValueError(f"Unknown function: {fn_name}")
                    
                    # Call function with args
                    if asyncio.iscoroutinefunction(fn):
                        result = await fn(**fn_args)
                    else:
                        result = fn(**fn_args)
                else:
                    # Handle regular tools
                    fn = self.tool_registry.get(tool_name)
                    if fn is None:
                        raise ValueError(f"Unknown tool: {tool_name}")
                    
                    if asyncio.iscoroutinefunction(fn):
                        result = await fn(resolved_args)
                    else:
                        result = fn(resolved_args)
                
                results.append(result)
                
                # Store in variable immediately for subsequent steps
                for var_edge in self.graph_store.get_edges(src=step_id, kind=EdgeKind.CUSTOM):
                    if var_edge.data.get("type") == "result_variable":
                        var_name = var_edge.data.get("variable")
                        if var_name:
                            context["variables"][var_name] = result
                            print(f"DEBUG: Set variable {var_name} = {type(result)}")
        
        # Store results in context
        if results:
            context["results"][step_id] = results
        
        return results


# --------------------------------------------------------------------------- custom tools / fns
async def batch_weather_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get weather data for multiple locations."""
    print(f"DEBUG: batch_weather_tool received args: {args}")
    
    # Get locations from args
    locations = args.get("locations", [])
    print(f"DEBUG: locations: {locations}")
    
    # Ensure we have a list
    if not isinstance(locations, list):
        locations = [str(locations)]
    
    # Sample weather data
    weather_data = {
        "New York": {"temperature": 72, "conditions": "Partly cloudy", "humidity": 65},
        "London":   {"temperature": 62, "conditions": "Rainy",          "humidity": 80},
        "Tokyo":    {"temperature": 78, "conditions": "Sunny",          "humidity": 70},
        "Sydney":   {"temperature": 68, "conditions": "Clear",          "humidity": 60},
        "Cairo":    {"temperature": 90, "conditions": "Hot",            "humidity": 30},
    }
    
    # Get weather for each location
    results = {}
    for loc in locations:
        results[loc] = weather_data.get(loc, {"temperature": 75, "conditions": "Unknown", "humidity": 50})
    
    return {"results": results}


def analyze_weather_function(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze weather data to extract statistics."""
    print(f"DEBUG: analyze_weather_function received: {type(weather_data)}")
    
    # Extract results from the weather data
    results = weather_data.get("results", weather_data)
    
    # Calculate statistics
    n = len(results)
    if n == 0:
        return {
            "average_temperature": 0,
            "average_humidity": 0,
            "most_common_condition": "Unknown",
            "condition_distribution": {},
            "locations_analyzed": 0
        }
    
    # Extract data
    temperatures = []
    humidities = []
    conditions = {}
    
    for location, data in results.items():
        # Extract temperature
        temp = data.get("temperature")
        if temp is not None:
            temperatures.append(temp)
        
        # Extract humidity
        humidity = data.get("humidity")
        if humidity is not None:
            humidities.append(humidity)
        
        # Extract condition
        condition = data.get("conditions", "Unknown")
        conditions[condition] = conditions.get(condition, 0) + 1
    
    # Calculate averages
    avg_temp = sum(temperatures) / len(temperatures) if temperatures else 0
    avg_humidity = sum(humidities) / len(humidities) if humidities else 0
    
    # Find most common condition
    most_common = max(conditions.items(), key=lambda x: x[1])[0] if conditions else "Unknown"
    
    return {
        "average_temperature": round(avg_temp, 1),
        "average_humidity": round(avg_humidity, 1),
        "most_common_condition": most_common,
        "condition_distribution": conditions,
        "locations_analyzed": n,
    }


def create_report_function(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create a report from analysis data."""
    print(f"DEBUG: create_report_function received: {type(analysis)}")
    
    # Create the report
    return {
        "title": "Global Weather Analysis Report",
        "summary": (
            f"{analysis.get('locations_analyzed', 0)} cities analysed. "
            f"Avg T = {analysis.get('average_temperature', 'N/A')} °F, "
            f"Avg RH = {analysis.get('average_humidity', 'N/A')} %. "
            f"Most common: {analysis.get('most_common_condition', 'Unknown')}."
        ),
        "details": analysis,
    }


def format_visualization_function(weather_data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Format data for visualization."""
    print(f"DEBUG: format_visualization_function received: {type(weather_data)}, {type(analysis)}")
    
    # Extract results safely
    results = weather_data.get("results", weather_data)
    
    # Create temperature data for visualization
    temps = []
    for location, data in results.items():
        temps.append({"location": location, "temperature": data.get("temperature", 0)})
    
    # Sort temperatures
    temps.sort(key=lambda x: x["temperature"], reverse=True)
    
    # Create condition data for visualization
    conds = []
    for condition, count in analysis.get("condition_distribution", {}).items():
        conds.append({"condition": condition, "count": count})
    
    return {
        "title": "Global Weather Visualization",
        "temperature_data": temps,
        "condition_data": conds,
    }


# --------------------------------------------------------------------------- plan factory
def make_plan(store=None) -> UniversalPlan:
    """Create a weather analysis plan with proper variable references."""
    from chuk_ai_planner.store.memory import InMemoryGraphStore
    
    if store is None:
        store = InMemoryGraphStore()
        
    plan = UniversalPlan(
        title="Global Weather Analysis",
        description="Analyse weather for multiple cities",
        tags=["weather", "analysis", "demo"],
        graph=store,
    )
    
    # Define the target cities directly in the plan
    target_cities = ["New York", "London", "Tokyo", "Sydney", "Cairo"]
    plan.set_variable("target_cities", target_cities)
    plan.save()

    # Add steps with dependencies
    s1 = plan.add_tool_step(
        "Collect Weather Data",
        tool="batch_weather",
        args={"locations": target_cities},  # Direct reference
        result_variable="weather_data",
    )
    s2 = plan.add_function_step(
        "Analyze Weather Data",
        function="analyze_weather",
        args={"weather_data": "${weather_data}"},
        result_variable="analysis",
        depends_on=[s1],
    )
    s3 = plan.add_function_step(
        "Generate Weather Report",
        function="create_report",
        args={"analysis": "${analysis}"},
        result_variable="report",
        depends_on=[s2],
    )
    s4 = plan.add_function_step(
        "Format Visualization Data",
        function="format_visualization",
        args={"weather_data": "${weather_data}", "analysis": "${analysis}"},
        result_variable="viz",
        depends_on=[s2],
    )
    plan.save()
    return plan


# --------------------------------------------------------------------------- main
async def main():
    print("🌤️  Universal Executor Demo\n" + "=" * 35)

    # Create a simple executor
    executor = SimpleExecutor()
    
    # Create plan with the executor's graph store
    plan = make_plan(executor.graph_store)

    # Register tools / functions
    executor.register_tool("batch_weather", batch_weather_tool)
    executor.register_function("analyze_weather", analyze_weather_function)
    executor.register_function("create_report", create_report_function)
    executor.register_function("format_visualization", format_visualization_function)

    print("\n▶️ Executing …")
    res = await executor.execute_plan(plan)

    if not res["success"]:
        print(f"\n❌ Plan failed: {res['error']}")
        return

    # Debug raw variables
    print("\nDEBUG: Variables in result:")
    for k, v in res["variables"].items():
        print(f"  {k}: {type(v)}")
        if hasattr(v, '__repr__'):
            preview = str(v)[:50] + "..." if len(str(v)) > 50 else str(v)
            print(f"    Preview: {preview}")

    # Extract results
    weather_data = res["variables"].get("weather_data", {})
    analysis = res["variables"].get("analysis", {})
    report = res["variables"].get("report", {})
    viz = res["variables"].get("viz", {})

    print("\n✅ Success!\n")

    print("=== WEATHER DATA ===")
    pprint.pprint(weather_data, width=100, sort_dicts=False)

    print("\n=== ANALYSIS ===")
    pprint.pprint(analysis, width=100, sort_dicts=False)

    print("\n=== REPORT ===")
    pprint.pprint(report, width=100, sort_dicts=False)

    print("\n=== VIZ DATA ===")
    pprint.pprint(viz, width=100, sort_dicts=False)

    try:
        with open("weather_analysis_results.json", "w") as fp:
            json.dump(res["variables"], fp, indent=2, default=str)
        print("\n💾  Results written to weather_analysis_results.json")
    except Exception as e:
        print(f"\n❌ Failed to write results: {e}")


# --------------------------------------------------------------------------- entry point
if __name__ == "__main__":
    asyncio.run(main())