#!/usr/bin/env python
# examples/universal_plan_executor_tool_processor.py
"""
Registry-driven UniversalPlan demo with chuk_tool_processor
==========================================================

• Uses proper chuk_tool_processor API with UniversalPlan
• Builds a three-step plan using the fluent interface
• Executes it with UniversalExecutor using proper tool registration
• Demonstrates variable resolution and dependency management
• Shows both InProcess and Subprocess strategies
• Compares UniversalExecutor vs PlanExecutor approaches
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

# Universal plan imports
from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.utils.pretty import clr

# chuk_tool_processor imports (proper API)
from chuk_tool_processor.registry import register_tool, initialize
from chuk_tool_processor.models.tool_call import ToolCall
from chuk_tool_processor.execution.strategies.inprocess_strategy import InProcessStrategy
from chuk_tool_processor.execution.strategies.subprocess_strategy import SubprocessStrategy
from chuk_tool_processor.execution.tool_executor import ToolExecutor

# ───────────────────── Tool implementations (using decorators) ──────
@register_tool(name="weather", namespace="demo")
class WeatherTool:
    """Weather tool implementation"""
    
    async def execute(self, location: str = "Unknown") -> Dict[str, Any]:
        """Get weather for a location"""
        print(f"🌤️ Getting weather for {location}...")
        
        # Mock weather data
        weather_data = {
            "New York": {"temperature": 72, "conditions": "Partly cloudy", "humidity": 65},
            "London": {"temperature": 62, "conditions": "Rainy", "humidity": 80},
            "Tokyo": {"temperature": 78, "conditions": "Sunny", "humidity": 70},
        }
        
        return weather_data.get(location, {"temperature": 75, "conditions": "Unknown", "humidity": 50})


@register_tool(name="calculator", namespace="demo") 
class CalculatorTool:
    """Calculator tool implementation"""
    
    async def execute(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        """Perform a calculation"""
        print(f"🧮 Calculating: {a} {operation} {b}")
        
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return {"result": result}


@register_tool(name="search", namespace="demo")
class SearchTool:
    """Search tool implementation"""
    
    async def execute(self, query: str = "") -> Dict[str, Any]:
        """Search for information"""
        print(f"🔍 Searching for: {query}")
        
        # Mock search results
        return {
            "query": query,
            "results": [
                {"title": f"Result for {query}", "url": f"https://example.com/search?q={query}"},
                {"title": f"Guide to {query}", "url": f"https://guide.com/{query}"},
                {"title": f"Research on {query}", "url": f"https://research.org/{query}"}
            ]
        }


@register_tool(name="analyzer", namespace="demo")
class AnalyzerTool:
    """Analysis tool that uses variable references"""
    
    async def execute(self, weather_data: Dict[str, Any], calculation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze weather and calculation data"""
        print(f"📊 Analyzing weather and calculation data...")
        
        # Extract values
        temperature = weather_data.get("temperature", 0)
        calc_result = calculation_result.get("result", 0)
        
        # Perform analysis
        analysis = {
            "temperature_category": "hot" if temperature > 75 else "moderate" if temperature > 60 else "cold",
            "calculation_magnitude": "large" if calc_result > 1000 else "small",
            "combined_score": temperature + (calc_result / 100),
            "recommendation": f"Temperature is {temperature}°F, calculation yielded {calc_result}"
        }
        
        return analysis


# ───────────────────── Universal Tool Registry Integration ──────────
class UniversalToolRegistry:
    """Universal tool registry that works with chuk_tool_processor and UniversalExecutor"""
    
    def __init__(self):
        self.chuk_registry = None
        self.strategy = None
        self.executor = None
    
    async def initialize(self, strategy_name: str = "inprocess"):
        """Initialize the registry and tool processor"""
        print(f"🔧 Initializing Universal tool registry with {strategy_name} strategy...")
        
        # Initialize chuk_tool_processor registry
        self.chuk_registry = await initialize()
        
        # Set up strategy
        if strategy_name == "inprocess":
            self.strategy = InProcessStrategy(self.chuk_registry, default_timeout=10.0, max_concurrency=4)
        elif strategy_name == "subprocess":
            self.strategy = SubprocessStrategy(self.chuk_registry, max_workers=4, default_timeout=10.0)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Create tool executor
        self.executor = ToolExecutor(registry=self.chuk_registry, strategy=self.strategy)
        
        print("✅ Universal tool registry initialized with tools:")
        print("   - weather (demo namespace)")
        print("   - calculator (demo namespace)")
        print("   - search (demo namespace)")
        print("   - analyzer (demo namespace)")
    
    async def create_universal_executor(self, graph_store=None):
        """Create a UniversalExecutor with this registry's tools"""
        # Create Universal Executor
        universal_executor = UniversalExecutor(graph_store=graph_store)
        
        # Register tools with the universal executor using proper async wrappers
        async def weather_wrapper(args: Dict[str, Any]) -> Any:
            print(f"🔄 [WRAPPER] Calling weather tool with args: {args}")
            call = ToolCall(tool="weather", namespace="demo", arguments=args)
            results = await self.executor.execute([call])
            if results and results[0].error is None:
                result = results[0].result
                print(f"✅ [WRAPPER] Weather tool completed successfully: {result}")
                return result
            else:
                raise RuntimeError(f"Weather tool failed: {results[0].error if results else 'No results'}")
        
        async def calculator_wrapper(args: Dict[str, Any]) -> Any:
            print(f"🔄 [WRAPPER] Calling calculator tool with args: {args}")
            call = ToolCall(tool="calculator", namespace="demo", arguments=args)
            results = await self.executor.execute([call])
            if results and results[0].error is None:
                result = results[0].result
                print(f"✅ [WRAPPER] Calculator tool completed successfully: {result}")
                return result
            else:
                raise RuntimeError(f"Calculator tool failed: {results[0].error if results else 'No results'}")
        
        async def search_wrapper(args: Dict[str, Any]) -> Any:
            print(f"🔄 [WRAPPER] Calling search tool with args: {args}")
            call = ToolCall(tool="search", namespace="demo", arguments=args)
            results = await self.executor.execute([call])
            if results and results[0].error is None:
                result = results[0].result
                print(f"✅ [WRAPPER] Search tool completed successfully: {result}")
                return result
            else:
                raise RuntimeError(f"Search tool failed: {results[0].error if results else 'No results'}")
        
        async def analyzer_wrapper(args: Dict[str, Any]) -> Any:
            print(f"🔄 [WRAPPER] Calling analyzer tool with args: {args}")
            call = ToolCall(tool="analyzer", namespace="demo", arguments=args)
            results = await self.executor.execute([call])
            if results and results[0].error is None:
                result = results[0].result
                print(f"✅ [WRAPPER] Analyzer tool completed successfully: {result}")
                return result
            else:
                raise RuntimeError(f"Analyzer tool failed: {results[0].error if results else 'No results'}")
        
        # Register tools with the universal executor
        universal_executor.register_tool("weather", weather_wrapper)
        universal_executor.register_tool("calculator", calculator_wrapper)
        universal_executor.register_tool("search", search_wrapper)
        universal_executor.register_tool("analyzer", analyzer_wrapper)
        
        print(f"🔗 Registered {len(['weather', 'calculator', 'search', 'analyzer'])} tools with UniversalExecutor")
        
        return universal_executor
    
    async def shutdown(self):
        """Shutdown the tool processor"""
        if self.executor and self.strategy:
            await self.strategy.shutdown()


# ───────────────────── Build Universal Plan ───────────────────────
def build_universal_plan() -> UniversalPlan:
    """Build a Universal Plan using the fluent interface"""
    print(clr("🟢  BUILD UNIVERSAL PLAN\n", "1;32"))
    
    # Create Universal Plan with custom graph store
    graph_store = InMemoryGraphStore()
    plan = UniversalPlan(
        title="Enhanced Daily Helper",
        description="Universal plan with variable resolution and dependencies",
        graph=graph_store,
        tags=["demo", "universal", "registry"]
    )
    
    # Set initial variables
    plan.set_variable("target_location", "New York")
    plan.set_variable("calc_a", 235.5)
    plan.set_variable("calc_b", 18.75)
    plan.set_variable("search_topic", "climate change adaptation")
    
    # Add metadata
    plan.add_metadata("created_by", "universal_registry_demo")
    plan.add_metadata("execution_strategy", "chuk_tool_processor")
    
    # Build plan using fluent interface with direct method calls
    step1_id = plan.add_tool_step(
        title="Get weather data",
        tool="weather",
        args={"location": "${target_location}"},
        result_variable="weather_data"
    )
    
    step2_id = plan.add_tool_step(
        title="Perform calculation",
        tool="calculator", 
        args={"operation": "multiply", "a": "${calc_a}", "b": "${calc_b}"},
        result_variable="calculation_result"
    )
    
    step3_id = plan.add_tool_step(
        title="Search for information",
        tool="search",
        args={"query": "${search_topic}"},
        result_variable="search_results"
    )
    
    step4_id = plan.add_tool_step(
        title="Analyze combined data",
        tool="analyzer",
        args={
            "weather_data": "${weather_data}",
            "calculation_result": "${calculation_result}"
        },
        result_variable="analysis_results",
        depends_on=[step1_id, step2_id]  # Analysis depends on weather and calculation
    )
    
    # Save the plan
    plan.save()
    
    # Display plan structure
    print("Universal Plan Structure:")
    print(plan.outline())
    print()
    
    # Display step IDs for debugging
    print("Step IDs created:")
    print(f"  - Weather step: {step1_id[:8]}")
    print(f"  - Calculator step: {step2_id[:8]}")
    print(f"  - Search step: {step3_id[:8]}")
    print(f"  - Analyzer step: {step4_id[:8]} (depends on weather & calc)")
    print()
    
    # Display plan details
    plan_dict = plan.to_dict()
    print("Plan Details:")
    print(f"  - Title: {plan_dict['title']}")
    print(f"  - Description: {plan_dict['description']}")
    print(f"  - Tags: {plan_dict['tags']}")
    print(f"  - Variables: {len(plan_dict['variables'])} set")
    print(f"  - Steps: {len(plan_dict['steps'])} configured")
    print()
    
    return plan


# ───────────────────── Execute Universal Plan ──────────────────────
async def execute_universal_plan(plan: UniversalPlan, tool_registry: UniversalToolRegistry, strategy_name: str):
    """Execute the universal plan with the specified strategy"""
    print(clr(f"🛠  EXECUTE UNIVERSAL PLAN ({strategy_name.upper()})", "1;34"))
    
    # Create universal executor with the tool registry
    executor = await tool_registry.create_universal_executor(plan.graph)
    
    # Execute the plan
    try:
        print(f"🏃 Starting plan execution...")
        print(f"📋 Initial variables: {list(plan.variables.keys())}")
        print("=" * 50)
        
        result = await executor.execute_plan(plan)
        
        print("=" * 50)
        print(f"🏁 Plan execution completed!")
        
        if result["success"]:
            print(f"✅ Universal plan executed successfully with {strategy_name} strategy!")
            
            # Display results
            print(clr("\n🎉  UNIVERSAL PLAN RESULTS", "1;32"))
            
            variables = result["variables"]
            
            # Categorize variables
            input_vars = ["target_location", "calc_a", "calc_b", "search_topic"]
            output_vars = [k for k in variables.keys() if k not in input_vars]
            
            # Show input variables
            print(f"\n🔸 Input Variables ({len(input_vars)}):")
            for var_name in input_vars:
                if var_name in variables:
                    print(f"   - {var_name}: {variables[var_name]}")
            
            # Show output variables with detailed formatting
            print(f"\n🔸 Output Variables ({len(output_vars)} generated):")
            for var_name in output_vars:
                print(f"\n   📄 {var_name}:")
                value = variables[var_name]
                if isinstance(value, (dict, list)):
                    # Pretty print with indentation
                    import pprint
                    lines = pprint.pformat(value, width=70, sort_dicts=False).split('\n')
                    for line in lines:
                        print(f"      {line}")
                else:
                    print(f"      {value}")
            
            # Show execution results if available
            if "results" in result:
                print(clr("\n📊 EXECUTION RESULTS", "1;32"))
                for step_id, step_results in result["results"].items():
                    print(f"\nStep {step_id[:8]}:")
                    for i, step_result in enumerate(step_results):
                        print(f"  Result {i+1}: {step_result}")
            
            return result
        else:
            print(f"❌ Universal plan execution failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Error executing universal plan: {e}")
        return None


# ───────────────────── Strategy Comparison ─────────────────────────
async def compare_strategies(plan: UniversalPlan):
    """Compare different execution strategies with the Universal Plan"""
    print(clr("\n🔄 UNIVERSAL PLAN STRATEGY COMPARISON", "1;35"))
    
    strategies = ["inprocess", "subprocess"]
    results = {}
    
    for strategy_name in strategies:
        print(f"\n=== {strategy_name.upper()} STRATEGY ===")
        
        # Create tool registry for this strategy
        tool_registry = UniversalToolRegistry()
        
        try:
            await tool_registry.initialize(strategy_name)
            
            # Execute plan with this strategy
            result = await execute_universal_plan(plan, tool_registry, strategy_name)
            
            if result:
                results[strategy_name] = {
                    "success": True,
                    "variables_count": len(result["variables"]),
                    "results_count": len(result.get("results", {}))
                }
                print(f"{strategy_name.capitalize()} strategy: ✅ Success")
            else:
                results[strategy_name] = {"success": False}
                print(f"{strategy_name.capitalize()} strategy: ❌ Failed")
                
        except Exception as e:
            print(f"{strategy_name.capitalize()} strategy: ❌ Error - {e}")
            results[strategy_name] = {"success": False, "error": str(e)}
        finally:
            await tool_registry.shutdown()
    
    # Compare results
    print(clr("\n📊 STRATEGY COMPARISON SUMMARY", "1;33"))
    for strategy, result in results.items():
        status = "✅ Success" if result.get("success") else "❌ Failed"
        print(f"  {strategy.capitalize():12} - {status}")
        if result.get("success"):
            print(f"                - Variables: {result.get('variables_count', 0)}")
            print(f"                - Results: {result.get('results_count', 0)}")


# ───────────────────── Variable Resolution Demo ────────────────────
async def demonstrate_variable_resolution():
    """Demonstrate advanced variable resolution capabilities"""
    print(clr("\n🔗 VARIABLE RESOLUTION DEMONSTRATION", "1;36"))
    
    # Create a plan with complex variable references
    plan = UniversalPlan(
        title="Variable Resolution Demo",
        description="Demonstrates nested variable resolution"
    )
    
    # Set up nested variables
    plan.set_variable("config", {
        "api": {
            "endpoint": "api.weather.com",
            "port": 443
        },
        "user": {
            "id": "demo_user",
            "preferences": {
                "units": "fahrenheit",
                "detail_level": "high"
            }
        }
    })
    
    plan.set_variable("template", "Weather for ${config.user.id} from ${config.api.endpoint}:${config.api.port}")
    
    # Show variable resolution
    print("Variables set:")
    print(f"  - config.api.endpoint: {plan.variables['config']['api']['endpoint']}")
    print(f"  - config.user.id: {plan.variables['config']['user']['id']}")
    print(f"  - template: {plan.variables['template']}")
    
    print("\nVariable resolution would resolve:")
    print(f"  '${{config.user.id}}' → '{plan.variables['config']['user']['id']}'")
    print(f"  '${{config.api.endpoint}}' → '{plan.variables['config']['api']['endpoint']}'")


# ───────────────────── Main Demo Function ──────────────────────────
async def main() -> None:
    """Run the Universal Plan registry demo"""
    print(clr("🚀 Universal Plan with chuk_tool_processor Registry Demo", "1;36"))
    print("=" * 70)
    
    # Build the universal plan
    plan = build_universal_plan()
    
    # Demo variable resolution
    await demonstrate_variable_resolution()
    
    # Execute plan with default strategy
    print(clr("\n🏃 EXECUTING WITH DEFAULT STRATEGY", "1;34"))
    tool_registry = UniversalToolRegistry()
    
    try:
        await tool_registry.initialize("inprocess")
        result = await execute_universal_plan(plan, tool_registry, "inprocess")
        
        if result:
            print(clr(f"\n📈 DETAILED EXECUTION SUMMARY", "1;33"))
            print(f"   - Plan: {plan.title}")
            print(f"   - Steps: {len(plan.to_dict()['steps'])}")
            print(f"   - Input Variables: {len([k for k in result['variables'].keys() if k.startswith(('target_', 'calc_', 'search_'))])}")
            print(f"   - Output Variables: {len([k for k in result['variables'].keys() if not k.startswith(('target_', 'calc_', 'search_'))])}")
            print(f"   - Strategy: chuk_tool_processor InProcess")
            print(f"   - Success: ✅")
            
            # Show variable summary
            print(f"\n📋 Variable Summary:")
            for var_name, var_value in result['variables'].items():
                if isinstance(var_value, dict):
                    print(f"   - {var_name}: {type(var_value).__name__} with {len(var_value)} keys")
                elif isinstance(var_value, (list, tuple)):
                    print(f"   - {var_name}: {type(var_value).__name__} with {len(var_value)} items")
                else:
                    print(f"   - {var_name}: {type(var_value).__name__} = {str(var_value)[:50]}{'...' if len(str(var_value)) > 50 else ''}")
        
    finally:
        await tool_registry.shutdown()
    
    # Compare strategies
    await compare_strategies(plan)
    
    print(clr("\n🎊 UNIVERSAL PLAN DEMO COMPLETE", "1;32"))
    print("Key features demonstrated:")
    print("  ✅ Universal Plan fluent interface")
    print("  ✅ Variable resolution with ${} syntax")
    print("  ✅ chuk_tool_processor integration")
    print("  ✅ Multiple execution strategies")
    print("  ✅ Dependency management")
    print("  ✅ Professional logging and error handling")


if __name__ == "__main__":
    asyncio.run(main())