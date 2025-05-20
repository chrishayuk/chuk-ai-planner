#!/usr/bin/env python
# cli/demo_deep_researcher.py
"""
demo_deep_researcher.py
=======================

A demonstration of recursive research capability using the planner system.
This script takes a research query and performs iterative, multi-round research.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
import warnings
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse

# Import planner components
from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.models import ToolCall
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind
from chuk_ai_planner.processor import GraphAwareToolProcessor
from chuk_ai_planner.utils.visualization import print_session_events, print_graph_structure

# Import session management
from chuk_session_manager.storage import InMemorySessionStore, SessionStoreProvider
from chuk_session_manager.models.session import Session

# Import the actual tools from sample_tools
from sample_tools import SearchTool, VisitURL

# Import for LLM interactions
from openai import AsyncOpenAI

# Import modularized components
from cli.research_tracker import ResearchTracker
from cli.research_tools import register_enhanced_tools, resolve_url_placeholders, format_tool_result
from cli.research_summarizer import generate_research_summary, format_research_statistics

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Default configuration
DEFAULT_CONFIG = {
    "max_rounds": 3,
    "max_searches_per_round": 5,
    "verbose": True,
}

# LLM system prompt - Modified to strongly emphasize URL visiting
SYSTEM_PROMPT = """
You are an expert research assistant. Given a research query, create a JSON plan
that outlines steps to gather information about the topic. Return ONLY valid JSON!

Schema:
{
  "title": str,
  "steps": [
    {"title": str, "tool": str, "args": {}, "depends_on": [indices]}
  ]
}

Available tools: 
1. search - Find information on the web
   Args: 
   - query: The search query
   - max_results: (optional) Number of results to return (default: 10)

2. visit_url - Visit and extract content from a URL
   Args:
   - url: The URL to visit

For each step, specify:
- A clear title describing the step
- The appropriate tool (search or visit_url)
- The necessary arguments for the tool
- Any dependencies (steps that must complete first)

Important notes:
- Create 3-5 diverse search queries to gather comprehensive information
- MANDATORY: For EACH search query, ALWAYS add at least one visit_url step to visit a specific URL
- For visit_url steps, ALWAYS use actual, complete URLs with http:// or https:// prefix
- After a search step, use the visit_url tool to access at least one of the most relevant URLs discovered
- Use specific, focused search queries that include key terms
- Try different query formulations to find varied information
- Include searches for different content types (articles, videos, etc.)
- Indices start at 1
- "depends_on" should be an array of step indices that must complete first
- Todays date is 20th of May 2025

Your goal is to create a thorough research plan that explores multiple angles of the query.
Remember, a good plan ALWAYS includes both searching AND visiting specific URLs.
"""

async def call_llm(prompt: str) -> Dict[str, Any]:
    """Call OpenAI API to generate a research plan"""
    print(f"🤖 Calling OpenAI API with prompt: {prompt[:50]}...")
    
    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    
    # Get the content
    content = resp.choices[0].message.content
    print(f"📄 Response content: {content[:100]}...")
    
    # Handle JSON extraction if needed
    if "```json" in content:
        # Extract JSON from markdown code blocks
        json_content = content.split("```json")[1].split("```")[0].strip()
        return json.loads(json_content)
    elif "```" in content:
        # Extract from generic code blocks
        json_content = content.split("```")[1].split("```")[0].strip()
        return json.loads(json_content)
    else:
        # Try direct parsing
        return json.loads(content)

def convert_to_universal_plan(plan_json: Dict[str, Any]) -> UniversalPlan:
    """Convert JSON plan to UniversalPlan object"""
    plan = UniversalPlan(
        title=plan_json.get("title", "Research Plan"),
        description="Generated research plan",
        tags=["research", "auto-generated"]
    )
    
    # Add metadata
    plan.add_metadata("created_at", str(time.time()))
    plan.add_metadata("source", "llm")
    
    # Create steps mapping
    step_ids = {}
    
    # First pass: Create all steps without dependencies
    for i, step_data in enumerate(plan_json.get("steps", []), 1):
        title = step_data.get("title", f"Step {i}")
        step_index = plan.add_step(title, parent=None)
        
        # Find the step ID
        step_id = None
        for node in plan._graph.nodes.values():
            if node.__class__.__name__ == "PlanStep" and node.data.get("index") == step_index:
                step_id = node.id
                break
        
        if step_id:
            step_ids[i] = step_id
    
    # Second pass: Add tools and dependencies
    for i, step_data in enumerate(plan_json.get("steps", []), 1):
        step_id = step_ids.get(i)
        if not step_id:
            continue
        
        # Add tool call
        tool = step_data.get("tool")
        args = step_data.get("args", {})
        
        if tool:
            # Create and link tool call
            tool_call = ToolCall(data={"name": tool, "args": args})
            plan._graph.add_node(tool_call)
            plan._graph.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step_id, dst=tool_call.id))
            
            # Store result in variable
            plan._graph.add_edge(GraphEdge(
                kind=EdgeKind.CUSTOM,
                src=step_id,
                dst=tool_call.id,
                data={"type": "result_variable", "variable": f"result_{i}"}
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

async def prepare_session():
    """Create and initialize a session properly"""
    # Create a store and set it as the provider
    store = InMemorySessionStore()
    SessionStoreProvider.set_store(store)
    
    # Create a session
    session = Session()
    
    # Save the session (this may need to be awaited in your environment)
    try:
        # Try to await the save
        await store.save(session)
    except Exception:
        # If that fails, use the non-await version as in deep_research_cli_full.py
        SessionStoreProvider.get_store().save(session)
    
    return session

async def execute_plan(plan_json, session_id, tracker):
    """Execute a single research plan with direct execution"""
    # Convert JSON to UniversalPlan
    plan = convert_to_universal_plan(plan_json)
    plan_id = plan.save()
    
    # Setup processor with tools
    proc = GraphAwareToolProcessor(session_id, plan.graph)
    tools = {}  # Dictionary to store our registered tools
    
    # Create base tools
    search_tool = SearchTool()
    visit_url_tool = VisitURL()
    
    # Register enhanced tools
    await register_enhanced_tools(proc, tools, search_tool, visit_url_tool)
    
    # Extract steps for direct execution
    steps = []
    for i, step_data in enumerate(plan_json.get("steps", []), 1):
        if "tool" in step_data and "args" in step_data:
            steps.append({
                "index": i,
                "title": step_data.get("title", f"Step {i}"),
                "tool": step_data.get("tool"),
                "args": step_data.get("args", {}),
                "depends_on": step_data.get("depends_on", [])
            })
    
    # Execute steps directly
    results = []
    extracted_urls = {}  # Store URLs extracted from search results by step index
    
    for step in steps:
        print(f"🔄 Executing step {step['index']}: {step['title']}")
        tool_name = step["tool"]
        tool_args = step["args"].copy()  # Make a copy to avoid modifying the original
        
        # Handle URL placeholders and URL_FROM_STEP references
        if tool_name == "visit_url" and "url" in tool_args:
            original_url = tool_args["url"]
            resolved_url = resolve_url_placeholders(
                original_url, step["index"], extracted_urls, tracker
            )
            
            if resolved_url != original_url:
                tool_args["url"] = resolved_url
        
        # Get the tool function from our tools dictionary
        tool_fn = tools.get(tool_name)
        
        if not tool_fn:
            print(f"⚠️ Tool {tool_name} not found in tools dictionary")
            continue
        
        try:
            # Execute the tool
            result = await tool_fn(tool_args)
            
            # Store extracted URLs from search results
            if tool_name == "search" and "extracted_urls" in result:
                print(f"📋 Storing {len(result['extracted_urls'])} URLs from step {step['index']}")
                extracted_urls[step['index']] = result["extracted_urls"]
                
                # Remove extracted_urls from the result to avoid cluttering the output
                if "extracted_urls" in result:
                    del result["extracted_urls"]
            
            # Create a result object
            result_obj = type('ToolResult', (), {
                'tool': tool_name,
                'args': tool_args,
                'result': result
            })
            
            results.append(result_obj)
            
            # Track the result
            tracker.add_result(tool_name, tool_args, result)
            
        except Exception as e:
            print(f"❌ Error executing {tool_name}: {e}")
    
    # If we somehow didn't visit any URLs in this plan, add automatic URL visits
    if not any(r.tool == "visit_url" for r in results):
        print("⚠️ No URLs were visited in this plan. Adding automatic URL visits...")
        
        # Get all URLs we've found
        all_urls = []
        for step_idx, url_list in extracted_urls.items():
            for url_info in url_list:
                all_urls.append({
                    "step": step_idx,
                    "url_info": url_info
                })
        
        # Add up to 3 automatic URL visits
        for idx, url_data in enumerate(all_urls[:3]):
            step_idx = url_data["step"]
            url_info = url_data["url_info"]
            url = url_info.get("url")
            
            if url:
                print(f"🔄 Automatically visiting URL from step {step_idx}: {url}")
                
                try:
                    # Visit the URL
                    result = await tools["visit_url"]({"url": url})
                    
                    # Create a result object
                    result_obj = type('ToolResult', (), {
                        'tool': "visit_url",
                        'args': {"url": url},
                        'result': result
                    })
                    
                    results.append(result_obj)
                    
                    # Track the result
                    tracker.add_result("visit_url", {"url": url}, result)
                    
                except Exception as e:
                    print(f"❌ Error during automatic URL visit: {e}")
    
    return results

def ensure_url_visits_in_plan(plan_json):
    """
    Ensure the plan has URL visits for each search.
    
    Args:
        plan_json: The plan JSON to modify
        
    Returns:
        Modified plan with URL visits
    """
    steps = plan_json.get("steps", [])
    has_visit_url = any(step.get("tool") == "visit_url" for step in steps)
    search_steps = [i+1 for i, step in enumerate(steps) if step.get("tool") == "search"]
    
    if not has_visit_url and search_steps:
        print("⚠️ Plan does not include any URL visits. Adding them manually...")
        
        # Add a visit_url step for the first search step
        new_steps = []
        for step in steps:
            new_steps.append(step)
            
            # After each search step, add a URL visit
            if step.get("tool") == "search":
                search_idx = new_steps.index(step) + 1  # 1-based index
                visit_step = {
                    "title": f"Visit top result from search '{step['args']['query']}'",
                    "tool": "visit_url",
                    "args": {
                        "url": f"URL_FROM_STEP_{search_idx}_RESULT_1"
                    },
                    "depends_on": [search_idx]
                }
                new_steps.append(visit_step)
        
        # Update the plan
        plan_json["steps"] = new_steps
        print(f"✅ Added {len(new_steps) - len(steps)} URL visit steps to the plan")
    
    return plan_json

async def run(query, cfg):
    """Run the deep researcher with multiple rounds"""
    print("\n🔍  STARTING DEEPER RESEARCH...\n")
    
    # Initialize session
    session = await prepare_session()
    print(f"✅ Created session with ID: {session.id}")
    
    # Initialize tracker
    tracker = ResearchTracker()
    
    # Research loop
    for round_num in range(1, cfg.get("max_rounds", 3) + 1):
        # Start a new round
        tracker.start_round()
        
        # Generate plan
        prompt = f"Research query: {query}"
        if round_num > 1:
            # Add context from previous rounds
            prompt += "\nBased on previous findings, focus on new information."
            
            # Add a summary of what we've found so far
            prompt += tracker.format_summary_for_prompt(max_results=7)
            
            # Add suggestions for research gaps
            prompt += tracker.get_research_gaps(round_num)
        
        # Get plan from LLM
        plan_json = await call_llm(prompt)
        
        # Ensure the plan has URL visits
        plan_json = ensure_url_visits_in_plan(plan_json)
        
        # Log the plan
        print(f"\n📋  EXECUTING PLAN (round {round_num})")
        print(json.dumps(plan_json, indent=2))
        
        # Execute the plan
        results = await execute_plan(plan_json, session.id, tracker)
        
        # Print results
        print(f"\n📊  ROUND {round_num} RESULTS")
        for r in results:
            print(f"\n• {r.tool} - {json.dumps(r.args)}")
            
            # Format and display the result
            formatted_result = format_tool_result(r)
            for line in formatted_result.split("\n"):
                print(f"  → {line}")
        
        # Decide whether to continue
        if round_num < cfg.get("max_rounds", 3):
            print(f"\n⏳  Planning next research round...\n")
            await asyncio.sleep(1)  # Brief pause
        else:
            print(f"\n✅  Research complete after {round_num} rounds\n")
    
    # Generate a comprehensive summary
    print("\n📑  GENERATING COMPREHENSIVE RESEARCH SUMMARY...\n")
    summary = await generate_research_summary(query, tracker)
    print(summary)
    
    # Print statistics
    stats_str = format_research_statistics(tracker)
    print(stats_str)
    
    # Print session events if available
    try:
        print_session_events(session)
    except Exception as e:
        print(f"\n==== SESSION EVENTS ERROR: {e} ====")
    
    return tracker

def cli():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Deep Researcher Demo")
    parser.add_argument("query", help="Research query to investigate")
    parser.add_argument("--rounds", type=int, default=DEFAULT_CONFIG["max_rounds"],
                        help=f"Maximum research rounds (default: {DEFAULT_CONFIG['max_rounds']})")
    parser.add_argument("--steps", type=int, default=DEFAULT_CONFIG["max_searches_per_round"],
                        help=f"Max searches per round (default: {DEFAULT_CONFIG['max_searches_per_round']})")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    
    a = parser.parse_args()
    
    # Setup config
    cfg = DEFAULT_CONFIG.copy()
    cfg["max_rounds"] = a.rounds
    cfg["max_searches_per_round"] = a.steps
    cfg["verbose"] = not a.quiet
    
    # Check for OpenAI key
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY environment variable is required")
    
    # Run the researcher
    asyncio.run(run(a.query, cfg))

if __name__ == "__main__":
    cli()