#!/usr/bin/env python
# examples/universal_plan_demo.py
"""
universal_plan_demo.py
======================

Example of creating a research plan using the enhanced UniversalPlan class
without involving the executor or other components.
"""

from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.models import ToolCall
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind

from typing import Dict, List, Any, Optional
import json
import uuid

def create_research_plan() -> UniversalPlan:
    """Create a comprehensive research plan using UniversalPlan"""
    # Create the main research plan
    plan = UniversalPlan(
        title="Climate Impact Research",
        description="Research the impact of climate change on coastal cities",
        tags=["research", "climate", "coastal"]
    )
    
    # Add initial variables
    plan.set_variable("max_rounds", 3)
    plan.set_variable("max_sources", 5)
    
    # Add metadata
    plan.add_metadata("created_by", "demo_script")
    plan.add_metadata("priority", "high")
    
    # Setup initial search
    plan.step("Initial Research")
    
    # Add steps directly with tool/function/subplan calls
    initial_research_idx = "1"  # Root step index
    
    # Add search steps for climate impact data
    search_step_idx = plan.add_tool_step(
        title="Search for climate impact data",
        tool="search",
        args={"query": "climate change impact coastal cities data"},
        result_variable="initial_search"
    )
    
    # Add search steps for scientific papers
    scientific_step_idx = plan.add_tool_step(
        title="Search for scientific papers",
        tool="search",
        args={"query": "scientific papers climate change sea level rise coastal cities"},
        result_variable="scientific_search"
    )
    
    # Add function step to extract key sources
    extract_step_idx = plan.add_function_step(
        title="Extract key sources",
        function="extract_best_sources",
        args={
            "search_results": "${initial_search}",
            "scientific_results": "${scientific_search}",
            "max_sources": "${max_sources}"
        },
        result_variable="key_sources"
    )
    
    # Add content exploration section
    plan.step("Content Exploration")
    content_exploration_idx = "2"  # Second top-level step
    
    # Add step to process sources
    process_step_idx = plan.add_function_step(
        title="Process each source",
        function="process_sources",
        args={"sources": "${key_sources}"},
        result_variable="source_contents"
    )
    
    # Add step to analyze content
    analyze_step_idx = plan.add_function_step(
        title="Analyze source content",
        function="analyze_content",
        args={"contents": "${source_contents}"},
        result_variable="content_analysis"
    )
    
    # Add deeper research section
    plan.step("Deeper Research")
    deeper_research_idx = "3"  # Third top-level step
    
    # Add subplan step
    subplan_step_idx = plan.add_plan_step(
        title="Explore impact categories",
        plan_id="impact_categories_plan",
        args={
            "base_analysis": "${content_analysis}",
            "query": "climate impact categories coastal cities"
        },
        result_variable="impact_categories"
    )
    
    # Add step to generate focused queries
    queries_step_idx = plan.add_function_step(
        title="Generate focused queries",
        function="generate_queries",
        args={
            "analysis": "${content_analysis}",
            "categories": "${impact_categories}"
        },
        result_variable="focused_queries"
    )
    
    # Add step for focused searches
    focused_search_idx = plan.add_tool_step(
        title="Execute focused searches",
        tool="batch_search",
        args={"queries": "${focused_queries}"},
        result_variable="focused_results"
    )
    
    # Add synthesis section
    plan.step("Synthesis")
    synthesis_idx = "4"  # Fourth top-level step
    
    # Add step to synthesize findings
    synthesize_idx = plan.add_function_step(
        title="Synthesize research findings",
        function="synthesize_findings",
        args={
            "initial_analysis": "${content_analysis}",
            "category_results": "${impact_categories}",
            "focused_results": "${focused_results}"
        },
        result_variable="research_synthesis"
    )
    
    # Add step to generate report
    report_idx = plan.add_function_step(
        title="Generate comprehensive report",
        function="generate_report",
        args={"synthesis": "${research_synthesis}"},
        result_variable="final_report"
    )
    
    return plan


def describe_plan(plan: UniversalPlan) -> str:
    """Return a detailed description of the plan"""
    lines = [
        f"Plan: {plan.title}",
        f"Description: {plan.description}",
        f"ID: {plan.id}",
        f"Tags: {', '.join(plan.tags)}",
        f"Variables: {json.dumps(plan.variables, indent=2)}",
        f"Metadata: {json.dumps(plan.metadata, indent=2)}",
        "\nSteps:"
    ]
    
    # Display the plan structure
    for node in plan._graph.nodes.values():
        if node.__class__.__name__ == "PlanStep":
            index = node.data.get("index", "")
            title = node.data.get("description", "")
            
            # Find tool calls for this step
            tool_info = ""
            for edge in plan._graph.get_edges(src=node.id, kind=EdgeKind.PLAN_LINK):
                tool_node = plan._graph.get_node(edge.dst)
                if tool_node and tool_node.__class__.__name__ == "ToolCall":
                    tool_name = tool_node.data.get("name", "")
                    
                    if tool_name == "function":
                        # Handle function calls specially
                        fn_args = tool_node.data.get("args", {})
                        fn_name = fn_args.get("function", "")
                        fn_args = fn_args.get("args", {})
                        tool_info = f"Function: {fn_name}, Args: {fn_args}"
                    elif tool_name == "subplan":
                        # Handle subplan calls specially
                        subplan_args = tool_node.data.get("args", {})
                        plan_id = subplan_args.get("plan_id", "")
                        plan_args = subplan_args.get("args", {})
                        tool_info = f"Subplan: {plan_id}, Args: {plan_args}"
                    else:
                        # Regular tool
                        tool_args = tool_node.data.get("args", {})
                        tool_info = f"Tool: {tool_name}, Args: {tool_args}"
                    
                    # Look for result variable in custom edges
                    for result_edge in plan._graph.get_edges(src=node.id, kind=EdgeKind.CUSTOM):
                        if result_edge.data.get("type") == "result_variable":
                            tool_info += f", Result → ${result_edge.data.get('variable', '')}"
            
            # Add step line
            lines.append(f"  {index:<6} {title:<35} {tool_info}")
    
    return "\n".join(lines)


def main():
    """Main function to create and display the plan"""
    plan = create_research_plan()
    
    print("\n🗂️  PLAN STRUCTURE\n")
    print(plan.outline())
    
    print("\n📋  DETAILED PLAN DESCRIPTION\n")
    print(describe_plan(plan))
    
    # Get plan details as dictionary for a cleaner view
    plan_dict = {
        "id": plan.id,
        "title": plan.title,
        "description": plan.description,
        "variables": plan.variables,
        "metadata": plan.metadata
    }
    
    print("\n📊  PLAN DATA\n")
    print(json.dumps(plan_dict, indent=2))
    
    plan_id = plan.save()
    print(f"\n✅  Plan saved with ID: {plan_id}\n")


if __name__ == "__main__":
    main()