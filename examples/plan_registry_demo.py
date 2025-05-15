#!/usr/bin/env python
# examples/plan_registry_demo.py
"""
plan_registry_demo.py
=====================

Demonstrates the PlanRegistry for storing and retrieving UniversalPlans.

This example:
1. Creates several research plans
2. Registers them with the PlanRegistry
3. Retrieves plans by ID
4. Searches for plans by tags and title
5. Shows persistence to disk
"""

import os
import json
import shutil
from typing import Dict, List, Any

from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.planner.plan_registry import PlanRegistry

# Create a temporary directory for the registry
REGISTRY_DIR = "temp_registry"
if os.path.exists(REGISTRY_DIR):
    shutil.rmtree(REGISTRY_DIR)
os.makedirs(REGISTRY_DIR)

def create_research_plan(
    title: str,
    description: str,
    tags: List[str],
    query: str
) -> UniversalPlan:
    """Create a research plan with the given parameters"""
    plan = UniversalPlan(
        title=title,
        description=description,
        tags=tags
    )
    
    # Add metadata
    plan.add_metadata("created_by", "registry_demo")
    plan.add_metadata("priority", "medium")
    
    # Add variables
    plan.set_variable("query", query)
    plan.set_variable("max_results", 5)
    
    # Add steps
    plan.step("Initial Research")
    
    # Add a search step
    search_step_id = plan.add_tool_step(
        title="Search for information",
        tool="search",
        args={"query": query},
        result_variable="search_results"
    )
    
    # Add an analysis step
    analysis_step_id = plan.add_function_step(
        title="Analyze results",
        function="analyze_content",
        args={"content": "${search_results}"},
        result_variable="analysis",
        depends_on=[search_step_id]
    )
    
    # Add a summarization step
    summary_step_id = plan.add_function_step(
        title="Generate summary",
        function="generate_summary",
        args={"analysis": "${analysis}"},
        result_variable="summary",
        depends_on=[analysis_step_id]
    )
    
    return plan

def main():
    # Create a registry
    registry = PlanRegistry(storage_dir=REGISTRY_DIR)
    print(f"Created PlanRegistry in {REGISTRY_DIR}")
    
    # Create several research plans
    plans = [
        create_research_plan(
            title="Climate Impact Research",
            description="Research on climate change impacts",
            tags=["climate", "research", "environment"],
            query="climate change impact data"
        ),
        create_research_plan(
            title="AI Ethics Study",
            description="Investigation into ethical AI development",
            tags=["ai", "ethics", "research"],
            query="artificial intelligence ethics guidelines"
        ),
        create_research_plan(
            title="Quantum Computing Applications",
            description="Research on practical quantum computing applications",
            tags=["quantum", "computing", "research"],
            query="quantum computing practical applications"
        ),
        create_research_plan(
            title="Climate Adaptation Strategies",
            description="Research on adapting to climate change",
            tags=["climate", "adaptation", "strategy"],
            query="climate change adaptation strategies"
        )
    ]
    
    # Register all plans
    print("\n\n=== Registering Plans ===")
    plan_ids = []
    for plan in plans:
        plan_id = registry.register_plan(plan)
        plan_ids.append(plan_id)
        print(f"Registered plan: {plan.title} (ID: {plan_id})")
    
    # Retrieve a plan by ID
    print("\n\n=== Retrieving Plan by ID ===")
    retrieved_plan = registry.get_plan(plan_ids[0])
    if retrieved_plan:
        print(f"Retrieved plan: {retrieved_plan.title}")
        print(f"Description: {retrieved_plan.description}")
        print(f"Tags: {', '.join(retrieved_plan.tags)}")
        print(f"Variables: {json.dumps(retrieved_plan.variables, indent=2)}")
        
        # Display plan structure
        print("\nPlan structure:")
        print(retrieved_plan.outline())
    
    # Find plans by tags
    print("\n\n=== Finding Plans by Tags ===")
    climate_plans = registry.find_plans(tags=["climate"])
    print(f"Found {len(climate_plans)} plans with 'climate' tag:")
    for plan in climate_plans:
        print(f"- {plan.title} (ID: {plan.id})")
    
    # Find plans by title
    print("\n\n=== Finding Plans by Title ===")
    ai_plans = registry.find_plans(title_contains="AI")
    print(f"Found {len(ai_plans)} plans with 'AI' in title:")
    for plan in ai_plans:
        print(f"- {plan.title} (ID: {plan.id})")
    
    # Find plans by both criteria
    print("\n\n=== Finding Plans by Tags and Title ===")
    climate_adaptation_plans = registry.find_plans(
        tags=["climate"], 
        title_contains="Adaptation"
    )
    print(f"Found {len(climate_adaptation_plans)} plans with 'climate' tag and 'Adaptation' in title:")
    for plan in climate_adaptation_plans:
        print(f"- {plan.title} (ID: {plan.id})")
    
    # Get all plans
    print("\n\n=== Getting All Plans ===")
    all_plans = registry.get_all_plans()
    print(f"Registry contains {len(all_plans)} plans:")
    for plan in all_plans:
        print(f"- {plan.title} (ID: {plan.id})")
    
    # Delete a plan
    print("\n\n=== Deleting a Plan ===")
    plan_to_delete = plan_ids[2]
    success = registry.delete_plan(plan_to_delete)
    print(f"Deleted plan {plan_to_delete}: {success}")
    
    # Verify deletion
    remaining_plans = registry.get_all_plans()
    print(f"Registry now contains {len(remaining_plans)} plans")
    
    # Demonstrate persistence by creating a new registry instance
    print("\n\n=== Testing Persistence ===")
    new_registry = PlanRegistry(storage_dir=REGISTRY_DIR)
    loaded_plans = new_registry.get_all_plans()
    print(f"New registry instance loaded {len(loaded_plans)} plans from disk:")
    for plan in loaded_plans:
        print(f"- {plan.title} (ID: {plan.id})")

if __name__ == "__main__":
    main()
    
    # Clean up
    print(f"\nCleaning up {REGISTRY_DIR}")
    shutil.rmtree(REGISTRY_DIR)