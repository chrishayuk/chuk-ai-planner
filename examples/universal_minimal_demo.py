#!/usr/bin/env python
"""
examples/universal_minimal_demo.py
A minimal example of using UniversalPlan and UniversalExecutor
"""

import argparse
import asyncio
import logging

from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Simple tools
async def hello_tool(args):
    """A simple tool that says hello."""
    name = args.get("name", "World")
    logger.info(f"👋 Hello, {name}!")
    return {"message": f"Hello, {name}!"}

def summarize_function():
    """A simple function that generates a summary."""
    logger.info("📝 Creating summary...")
    return {"summary": "This is a simple summary."}

# Main function
async def run_demo(topic):
    """Run a minimal demo of UniversalPlan and UniversalExecutor."""
    logger.info(f"🚀 Starting minimal demo for: {topic}")
    
    # Create executor
    executor = UniversalExecutor()
    
    # Register tools and functions
    executor.register_tool("hello", hello_tool)
    executor.register_function("summarize", summarize_function)
    
    # Create a simple plan
    plan = UniversalPlan(
        title=f"Simple Demo for {topic}",
        description="A minimal demo of UniversalPlan and UniversalExecutor",
        tags=["demo"]
    )
    
    # Add a simple step
    s1 = plan.add_tool_step(
        title="Say hello",
        tool="hello",
        args={"name": topic},
        result_variable="greeting"
    )
    
    # Add a summary step
    s2 = plan.add_function_step(
        title="Generate summary",
        function="summarize",
        args={},  # No arguments needed
        depends_on=[s1],
        result_variable="summary"
    )
    
    # Save the plan
    plan.save()
    logger.info(f"Plan created: {plan.outline()}")
    
    # Execute the plan
    try:
        result = await executor.execute_plan(plan)
        
        if not result.get("success", False):
            logger.error(f"❌ Error: {result.get('error')}")
            return
        
        # Get results
        greeting = result.get("variables", {}).get("greeting", {}).get("message", "No greeting")
        summary = result.get("variables", {}).get("summary", {}).get("summary", "No summary")
        
        # Display results
        logger.info("\n✅ Plan executed successfully")
        logger.info(f"Greeting: {greeting}")
        logger.info(f"Summary: {summary}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

# CLI interface
def main():
    parser = argparse.ArgumentParser(description="Minimal UniversalPlan Demo")
    parser.add_argument("name", help="Name to greet")
    args = parser.parse_args()
    
    asyncio.run(run_demo(args.name))

if __name__ == "__main__":
    main()