#!/usr/bin/env python
"""
cli/demo_cli.py – deep research CLI
==================================
$ uv run cli/demo_cli.py \
    "Write an overview of Chris Hay of IBM"
"""

from __future__ import annotations

import argparse, asyncio, json, os, uuid
import warnings
from typing import Any, Dict, Tuple, List, Set, Optional

from dotenv import load_dotenv
load_dotenv()                        # ← OPENAI_API_KEY

# ── demo tools (auto-register on import) ────────────────────────────
from sample_tools import WeatherTool, SearchTool, VisitURL  # noqa: F401

# ── A2A plumbing ────────────────────────────────────────────────────
from a2a_session_manager.storage import InMemorySessionStore, SessionStoreProvider
from a2a_session_manager.models.session import Session
from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.planner import Plan
from chuk_ai_planner.models import ToolCall
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind
from chuk_ai_planner.processor import GraphAwareToolProcessor
from chuk_ai_planner.utils.visualization import print_session_events, print_graph_structure
from chuk_ai_planner.utils.registry_helpers import execute_tool

# ── planning agent ---------------------------------------------------
from chuk_ai_planner.agents.plan_agent import PlanAgent

# ╭──────────────────────────────────────────────────────────────────╮
# │ 1.  TOOL ALLOW-LIST & VALIDATION                                │
# ╰──────────────────────────────────────────────────────────────────╯
ALLOWED_TOOLS: set[str] = {"weather", "search", "visit_url"}

TOOL_SCHEMA: dict[str, dict[str, Any]] = {
    "weather":   {"location": lambda v: isinstance(v, str) and v.strip()},
    "search":    {"query":    lambda v: isinstance(v, str) and v.strip()},
    "visit_url": {"url":      lambda v: isinstance(v, str) and v.strip()},
}

def _tool_sig(name: str, spec: dict[str, Any]) -> str:
    inner = ", ".join(f"{k}:str" for k in spec)
    return f"  – {name}  {{{inner}}}"

SYS_MSG = (
    "You are an assistant that writes a JSON *plan* using only these tools:\n"
    + "\n".join(_tool_sig(n, TOOL_SCHEMA[n]) for n in ALLOWED_TOOLS)
    + "\nReturn ONLY a JSON object of the form\n"
    "{\n"
    '  "title": str,\n'
    '  "steps": [ { "title": str, "tool": str, "args": object, "depends_on": [] } ]\n'
    "}"
)

def validate_step(step: Dict[str, Any]) -> Tuple[bool, str]:
    tool = step.get("tool")
    if tool not in ALLOWED_TOOLS:
        return False, f"{tool!r} not allowed"
    spec, args = TOOL_SCHEMA[tool], step.get("args", {})
    miss  = [k for k in spec if k not in args]
    extra = [k for k in args if k not in spec]
    bad   = [k for k, fn in spec.items() if k in args and not fn(args[k])]
    if miss:  return False, f"{tool}: missing {miss}"
    if extra: return False, f"{tool}: unknown {extra}"
    if bad:   return False, f"{tool}: invalid {bad}"
    return True, ""

# ╭──────────────────────────────────────────────────────────────────╮
# │ 2.  TOOL-EXECUTION ADAPTER                                     │
# ╰──────────────────────────────────────────────────────────────────╯
async def _adapter(name: str, args: Dict[str, Any]) -> Any:
    """
    Enhanced adapter with improved error handling and URL preprocessing.
    """
    from urllib.parse import urlparse, parse_qs, unquote
    
    # Special handling for visit_url to preprocess DuckDuckGo URLs
    if name == "visit_url" and "url" in args:
        url = args["url"]
        
        # Extract the target URL from DuckDuckGo redirects before passing to tool
        if "duckduckgo.com/l/" in url:
            try:
                parsed = urlparse(url)
                params = parse_qs(parsed.query)
                if "uddg" in params and params["uddg"]:
                    args["url"] = unquote(params["uddg"][0])
            except Exception:
                pass
    
    # Create the tool call object with updated arguments
    tc = {
        "id": uuid.uuid4().hex,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)}
    }
    
    # Execute the tool with error handling
    try:
        result = await execute_tool(tc, None, None)
        return result
    except Exception as e:
        # Return a structured error response instead of raising
        return {"error": str(e)}

# ── replace current register_tools -----------------------------------
def register_tools(proc: GraphAwareToolProcessor) -> None:
    """
    Register tools with improved error handling.
    """
    from chuk_tool_processor.registry import default_registry
    
    # Suppress warnings that might interfere with tool execution
    warnings.filterwarnings("ignore", category=UserWarning)

    for t in ALLOWED_TOOLS:
        try:
            tool = default_registry.get_tool(t)
            # Bind *t* at definition-time with a default arg
            proc.register_tool(t, lambda a, _n=t: _adapter(_n, a))
        except KeyError:
            raise RuntimeError(f"Tool {t!r} not found in chuk registry")
        except Exception as e:
            raise RuntimeError(f"Error registering tool {t!r}: {e}")

# ╭──────────────────────────────────────────────────────────────────╮
# │ 3.  RESEARCH PLANNING                                          │
# ╰──────────────────────────────────────────────────────────────────╯
# Research status tracker to avoid duplication
class ResearchTracker:
    def __init__(self):
        self.search_queries = set()
        self.visited_urls = set()
        self.search_results = []
        self.website_results = []
        self.all_results = []
        self.all_graphs = []  # Store all graph objects for visualization
        
    def add_search_query(self, query):
        self.search_queries.add(query.lower())
        
    def add_url(self, url):
        self.visited_urls.add(url)
        
    def has_queried(self, query):
        return query.lower() in self.search_queries
        
    def has_visited(self, url):
        return url in self.visited_urls
        
    def add_result(self, result, is_search=False):
        self.all_results.append(result)
        if is_search:
            self.search_results.append(result)
        else:
            self.website_results.append(result)
    
    def add_graph(self, graph):
        self.all_graphs.append(graph)

async def create_plan_for_entities(agent: PlanAgent, topic: str, 
                                entities: List[str]) -> Optional[Dict]:
    """Create a plan to search for specific entities."""
    
    if not entities:
        return None
    
    steps = []
    for entity in entities:
        query = f"{entity} {topic}"
        steps.append({
            "title": f"Research: {entity}",
            "tool": "search",
            "args": {"query": query},
            "depends_on": []
        })
    
    return {
        "title": f"Research on Entities Related to {topic}",
        "steps": steps[:3]  # Limit to 3 steps
    }

async def create_follow_up_plan(agent: PlanAgent, goal: str, 
                               search_results: List[Dict]) -> Optional[Dict]:
    """Create a plan to visit websites based on search results."""
    
    if not search_results:
        return None
    
    # Extract URLs from search results
    url_steps = []
    
    # Flatten search results to get individual items
    items = []
    for result in search_results:
        if "results" in result:
            items.extend(result["results"])
    
    # Select up to 3 URLs to visit
    count = 0
    for item in items:
        if "url" in item and "title" in item and count < 3:
            url_steps.append({
                "title": f"Visit: {item['title']}",
                "tool": "visit_url",
                "args": {"url": item["url"]},
                "depends_on": []
            })
            count += 1
    
    if not url_steps:
        return None
        
    return {
        "title": f"Explore Key Sources for {goal}",
        "steps": url_steps
    }

async def extract_new_directions(agent: PlanAgent, goal: str, 
                                all_results: List[Dict]) -> List[str]:
    """Extract new research directions based on accumulated knowledge."""
    
    # Prepare a content summary for analysis
    content = "SEARCH AND WEBSITE RESULTS:\n\n"
    
    for i, result in enumerate(all_results[:10]):  # Limit to first 10 results
        if "title" in result:
            content += f"RESULT {i+1} TITLE: {result['title']}\n"
        
        if "results" in result:  # Search result
            for item in result["results"][:2]:
                content += f"- {item.get('title', 'Untitled')}: {item.get('snippet', '')}\n"
        elif "first_200_chars" in result:  # Website result
            content += f"CONTENT: {result.get('first_200_chars', '')}\n"
        
        content += "\n"
    
    prompt = (
        f"Based on the research we've done on '{goal}', identify 2-3 additional specific "
        f"aspects or topics that would provide valuable new information.\n\n"
        f"{content}\n\n"
        "Return ONLY a JSON array of additional search queries that would provide new "
        "information not already covered in our research. Focus on specifics that would "
        "deepen our understanding.\n\n"
        "Format: [\"specific query 1\", \"specific query 2\", \"specific query 3\"]"
    )
    
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        try:
            queries = json.loads(content)
            if isinstance(queries, list):
                return queries
        except:
            # If JSON parsing fails, try to extract queries from text
            pass
            
        # Fallback: extract lines that look like queries
        import re
        queries = re.findall(r'"([^"]+)"', content)
        return queries[:3]  # Limit to 3 queries
    
    except Exception as e:
        print(f"Error extracting new directions: {e}")
        return []

# ╭──────────────────────────────────────────────────────────────────╮
# │ 4.  PLAN EXECUTION                                             │
# ╰──────────────────────────────────────────────────────────────────╯
async def execute_plan(plan_json: Dict, session_id: str, 
                      tracker: ResearchTracker) -> List[Dict]:
    """Execute a research plan and collect results."""
    
    # Create plan structure
    plan = Plan(plan_json["title"])
    for s in plan_json["steps"]:
        plan.step(s["title"]).up()
    plan_id = plan.save()
    
    # Link steps to tools
    idx2step = {n.data["index"]: n.id
                for n in plan.graph.nodes.values()
                if n.__class__.__name__ == "PlanStep"}
    
    for i, s in enumerate(plan_json["steps"], 1):
        # Track searches and URLs
        if s["tool"] == "search" and "query" in s["args"]:
            tracker.add_search_query(s["args"]["query"])
        elif s["tool"] == "visit_url" and "url" in s["args"]:
            tracker.add_url(s["args"]["url"])
            
        tc = ToolCall(data={"name": s["tool"], "args": s["args"]})
        plan.graph.add_node(tc)
        plan.graph.add_edge(
            GraphEdge(kind=EdgeKind.PLAN_LINK, src=idx2step[str(i)], dst=tc.id)
        )
    
    # Store the graph for later visualization
    tracker.add_graph(plan.graph)
    
    # Set up processor
    proc = GraphAwareToolProcessor(session_id, plan.graph)
    register_tools(proc)
    
    # Execute plan and collect results
    results_list = []
    
    try:
        await proc.process_plan(
            plan_id, "assistant", lambda _: None, 
            on_step=lambda step_id, results: results_list.extend(results) or True
        )
    except Exception as e:
        print(f"Error executing plan: {e}")
    
    # Process and store results
    new_results = []
    
    for tr in results_list:
        result = tr.result
        if result:
            is_search = (tr.tool == "search")
            tracker.add_result(result, is_search)
            new_results.append(result)
            
    return new_results

# ╭──────────────────────────────────────────────────────────────────╮
# │ 5.  SUMMARY GENERATION                                         │
# ╰──────────────────────────────────────────────────────────────────╯
async def create_comprehensive_summary(results: list[dict[str, Any]], goal: str) -> str:
    """Generate a comprehensive summary of all research findings."""
    
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    
    # Prepare content for summarization
    content = "RESEARCH FINDINGS:\n\n"
    
    search_content = ""
    website_content = ""
    
    for result in results:
        if "results" in result:  # Search result
            search_content += "SEARCH RESULTS:\n"
            for item in result["results"]:
                search_content += f"- {item.get('title', 'No title')}: {item.get('snippet', 'No snippet')}\n"
            search_content += "\n"
        elif "first_200_chars" in result:  # Website content
            website_content += f"WEBSITE: {result.get('title', 'No title')}\n"
            website_content += f"CONTENT: {result.get('first_200_chars', 'No content')}\n\n"
    
    content += search_content + website_content
    
    prompt = (
        f"Task: Create a comprehensive, well-structured overview of {goal}\n\n"
        f"{content}\n\n"
        "Instructions:\n"
        "1. Organize the information into clear sections\n"
        "2. Include specific details and facts\n"
        "3. Highlight key achievements and contributions\n"
        "4. Provide context and background where relevant\n"
        "5. Create a coherent narrative that connects all the information\n\n"
        "Your summary should be thorough, well-organized, and professional."
    )
    
    try:
        rsp = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Error generating summary. Please check the research results manually."

# ╭──────────────────────────────────────────────────────────────────╮
# │ 6.  MAIN LOOP                                                  │
# ╰──────────────────────────────────────────────────────────────────╯
async def run(user_prompt: str) -> None:
    # Set up environment
    SessionStoreProvider.set_store(InMemorySessionStore())
    session = Session(); SessionStoreProvider.get_store().save(session)
    
    # Initialize research tracker
    tracker = ResearchTracker()
    
    # Initialize agent
    agent = PlanAgent(system_prompt=SYS_MSG, validate_step=validate_step)
    
    print("\n🔍  STARTING DEEP RESEARCH PROCESS...\n")
    
    # ---- PHASE 1: Initial Search ----
    print("\nPHASE 1: Initial Research")
    
    # Create initial research plan
    initial_plan_json = await agent.plan(user_prompt)
    print("\n📋  INITIAL RESEARCH PLAN\n")
    print(json.dumps(initial_plan_json, indent=2), "\n")
    
    # Execute initial plan
    print("Executing initial research...")
    initial_results = await execute_plan(initial_plan_json, session.id, tracker)
    
    print("\n✅  INITIAL RESULTS\n")
    for r in initial_results:
        print(json.dumps(r, indent=2), "\n")
    
    # ---- PHASE 2: Visit Key Websites ----
    print("\nPHASE 2: Exploring Key Sources")
    
    # Create plan to visit websites from search results
    website_plan = await create_follow_up_plan(agent, user_prompt, tracker.search_results)
    
    if website_plan and website_plan.get("steps"):
        print("\n📋  WEBSITE EXPLORATION PLAN\n")
        print(json.dumps(website_plan, indent=2), "\n")
        
        # Execute website exploration plan
        print("Visiting key websites...")
        website_results = await execute_plan(website_plan, session.id, tracker)
        
        print("\n✅  WEBSITE EXPLORATION RESULTS\n")
        for r in website_results:
            print(json.dumps(r, indent=2), "\n")
    
    # ---- PHASE 3: Deep Dive Research ----
    print("\nPHASE 3: Deep Dive Research")
    
    # Extract new research directions
    new_directions = await extract_new_directions(agent, user_prompt, tracker.all_results)
    
    if new_directions:
        # Create plan for deeper research
        deep_research_plan = await create_plan_for_entities(
            agent, user_prompt, new_directions
        )
        
        if deep_research_plan and deep_research_plan.get("steps"):
            print("\n📋  DEEP DIVE RESEARCH PLAN\n")
            print(json.dumps(deep_research_plan, indent=2), "\n")
            
            # Execute deep research plan
            print("Conducting deep research...")
            deep_results = await execute_plan(deep_research_plan, session.id, tracker)
            
            print("\n✅  DEEP RESEARCH RESULTS\n")
            for r in deep_results:
                print(json.dumps(r, indent=2), "\n")
    
            # ---- PHASE 4: Follow-up on New Findings ----
            print("\nPHASE 4: Follow-up Research")
            
            # Visit websites from new search results
            follow_up_plan = await create_follow_up_plan(
                agent, user_prompt, deep_results
            )
            
            if follow_up_plan and follow_up_plan.get("steps"):
                print("\n📋  FOLLOW-UP EXPLORATION PLAN\n")
                print(json.dumps(follow_up_plan, indent=2), "\n")
                
                # Execute follow-up plan
                print("Following up on new findings...")
                follow_up_results = await execute_plan(follow_up_plan, session.id, tracker)
                
                print("\n✅  FOLLOW-UP RESULTS\n")
                for r in follow_up_results:
                    print(json.dumps(r, indent=2), "\n")
    
    # ---- Final Summary ----
    print("\n🔍  COMPREHENSIVE RESEARCH SUMMARY\n")
    summary = await create_comprehensive_summary(tracker.all_results, user_prompt)
    print(summary, "\n")
    
    # Print research statistics
    print(f"\n📊  RESEARCH STATISTICS")
    print(f"- Search queries performed: {len(tracker.search_queries)}")
    print(f"- Websites visited: {len(tracker.visited_urls)}")
    print(f"- Total results: {len(tracker.all_results)}")
    
    # Print session events
    print_session_events(session)
    
    # Print graph structure of the first graph (if available)
    if tracker.all_graphs:
        print_graph_structure(tracker.all_graphs[0])

# ╭──────────────────────────────────────────────────────────────────╮
# │ 7.  CLI ENTRY-POINT                                            │
# ╰──────────────────────────────────────────────────────────────────╯
def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="user question / task")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY missing")

    asyncio.run(run(args.query))

if __name__ == "__main__":
    cli()