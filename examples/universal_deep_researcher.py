#!/usr/bin/env python
# examples/universal_deep_researcher.py
"""
Deep-Researcher with UniversalPlan – zero .Arguments issues.
Every capability is a *function* registered via `executor.register_function`,
so UniversalExecutor never tries to inspect a tool class.
"""

from __future__ import annotations
import argparse, asyncio, json, logging, os, re
from typing import Any, Dict, List
from urllib.parse import urlparse, parse_qs, unquote
from dotenv import load_dotenv; load_dotenv()

from chuk_ai_planner.planner.universal_plan          import UniversalPlan
from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor

# ───────────────────────────  search tool  ─────────────────────────────────
async def search_fn(**kw):
    """Mock search function - replace with real search API"""
    query = kw.get("query", "")
    logger.info(f"🔍 Searching for: {query}")
    
    # Mock search results
    results = [
        {
            "title": f"Top result for {query}",
            "url": f"https://example.com/1?q={query.replace(' ', '+')}",
            "snippet": f"This is comprehensive information about {query}...",
        },
        {
            "title": f"Research paper on {query}",
            "url": f"https://research.com/papers/{query.replace(' ', '-')}",
            "snippet": f"Academic study of {query} with detailed analysis...",
        },
        {
            "title": f"Wikipedia: {query}",
            "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            "snippet": f"Encyclopedia entry about {query} covering basics...",
        },
    ]
    
    return {"results": results, "query": query, "count": len(results)}

# ───────────────────────────  visit url tool  ──────────────────────────────
async def visit_fn(**kw):
    """Mock URL visit function"""
    url = kw.get("url", "")
    logger.info(f"🌐 Visiting: {url}")
    
    # Mock page content
    return {
        "url": url,
        "title": f"Page title for {url}",
        "content": f"This is the content from {url}. It contains detailed information about the topic.",
        "status": "success"
    }

# ───────────────────────────  weather tool  ─────────────────────────────────
async def weather_fn(**kw):
    """Mock weather function"""
    location = kw.get("location", "Unknown")
    logger.info(f"🌤️ Getting weather for: {location}")
    
    return {
        "location": location,
        "temperature": 72,
        "conditions": "Partly cloudy",
        "humidity": 65
    }

# ───────────────────────────  scrape_url  ────────────────────────────────────
async def scrape_fn(**kw):
    """Mock scraping function - in real implementation would use httpx + readability"""
    url = kw.get("url", "")
    
    # Handle DuckDuckGo redirect URLs
    if "duckduckgo.com/l/" in url:
        try:
            url = unquote(parse_qs(urlparse(url).query)["uddg"][0])
        except Exception:
            pass
    
    logger.info(f"📄 Scraping: {url}")
    
    # Mock scraping result
    try:
        return {
            "title": f"Scraped content from {urlparse(url).netloc}",
            "content": f"This is the scraped content from {url}. It contains detailed information extracted from the web page.",
            "url": url,
            "status": "success"
        }
    except Exception as e:
        return {"url": url, "error": str(e), "status": "failed"}

# ───────────────────────────  summarize  ─────────────────────────────────────
async def summarize_fn(**kw):
    """Summarize documents and research findings"""
    topic = kw.get("topic", "")
    documents = kw.get("documents", [])
    
    # Flatten and collect all documents from the resolved variables
    all_docs = []
    
    if isinstance(documents, list):
        for item in documents:
            if isinstance(item, dict):
                # Check if it's a search result with nested results
                if "results" in item and isinstance(item["results"], list):
                    all_docs.extend(item["results"])
                else:
                    # It's a direct document (scraped content, etc.)
                    all_docs.append(item)
            else:
                # String or other format
                all_docs.append({"content": str(item)})
    
    logger.info(f"📝 Summarizing {len(all_docs)} documents about: {topic}")
    
    # Create bullet points from documents
    bullets = []
    for i, doc in enumerate(all_docs[:8], 1):  # Limit to first 8 for summary
        if isinstance(doc, dict):
            title = doc.get('title', doc.get('url', f'Source {i}'))
            content = doc.get('content', doc.get('snippet', ''))
            if content:
                content_preview = content[:150] + "..." if len(content) > 150 else content
                bullets.append(f"• {title}: {content_preview}")
            else:
                bullets.append(f"• {title}")
        else:
            bullets.append(f"• Source {i}: {str(doc)[:100]}...")
    
    summary_text = f"""Research Summary: {topic}

Key Findings:
{chr(10).join(bullets) if bullets else "No detailed findings available."}

Conclusion: Based on {len(all_docs)} sources, this research provides {'comprehensive' if len(all_docs) > 2 else 'initial'} coverage of {topic}. The sources include search results, scraped web content, and additional research materials."""
    
    return {
        "summary": summary_text,
        "count": len(all_docs),
        "topic": topic,
        "source_types": {
            "search_results": len([d for d in documents if isinstance(d, dict) and "results" in d]),
            "web_content": len([d for d in documents if isinstance(d, dict) and "content" in d and "results" not in d]),
            "other": len([d for d in documents if not isinstance(d, dict)])
        }
    }

# ───────────────────────────  logging  ───────────────────────────────────────
logger = logging.getLogger("deepresearch")
logger.setLevel(logging.INFO)
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("%(message)s"))
logger.handlers.clear()
logger.addHandler(h)

# ───────────────────────────  LLM helper  ────────────────────────────────────
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

SYS = ("Return ONLY JSON {title, steps}. Each step {title, tool, args, depends_on}. "
       "Allowed tools: search, visit_url, scrape_url, weather, summarize. "
       "depends_on should be step numbers (1, 2, 3, etc.)")

async def llm_json(topic: str, live: bool) -> Dict[str, Any]:
    if live and AsyncOpenAI and os.getenv("OPENAI_API_KEY"):
        c = AsyncOpenAI()
        logger.info("📡 OpenAI plan request")
        r = await c.chat.completions.create(
            model="gpt-4o-mini", 
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYS},
                {"role": "user", "content": f"Create a research plan for: {topic}"}
            ]
        )
        txt = r.choices[0].message.content.strip()
    else:
        logger.info("🤖 Simulated plan generation")
        txt = json.dumps({
            "title": f"Deep Research on {topic}",
            "steps": [
                {
                    "title": f"Initial search for {topic}",
                    "tool": "search",
                    "args": {"query": topic},
                    "depends_on": []
                },
                {
                    "title": "Scrape top result",
                    "tool": "scrape_url", 
                    "args": {"url": "placeholder"},
                    "depends_on": [1]
                },
                {
                    "title": "Visit additional source",
                    "tool": "visit_url",
                    "args": {"url": "placeholder"},
                    "depends_on": [1]
                },
                {
                    "title": "Summarize all findings",
                    "tool": "summarize",
                    "args": {"topic": topic},
                    "depends_on": [1, 2, 3]
                }
            ]
        })
    
    # Extract JSON from markdown code blocks if present
    m = re.search(r"```(?:json)?(.*?)```", txt, re.S)
    return json.loads(m.group(1).strip() if m else txt)

# ───────────────────────────  sanitise  ──────────────────────────────────────
ALLOWED = {"search", "visit_url", "scrape_url", "weather", "summarize"}

def sanitise(steps: List[Dict[str, Any]], topic: str) -> List[Dict[str, Any]]:
    """Clean up and validate plan steps"""
    out = []
    
    for st in steps:
        tool = (st.get("tool") or "").lower()
        
        # Fix unknown tools
        if tool not in ALLOWED:
            title_lower = st.get("title", "").lower()
            if "scrape" in title_lower or "extract" in title_lower:
                tool = "scrape_url"
            elif "visit" in title_lower or "browse" in title_lower:
                tool = "visit_url"
            elif "weather" in title_lower:
                tool = "weather"
            elif "summary" in title_lower or "summarize" in title_lower:
                tool = "summarize"
            else:
                tool = "search"
        
        st["tool"] = tool
        
        # Ensure args is a dict
        args = st.get("args") if isinstance(st.get("args"), dict) else {}
        
        # Add default arguments
        if tool == "search" and not args.get("query"):
            args["query"] = topic
        
        if tool in {"visit_url", "scrape_url"} and not args.get("url"):
            args["url"] = "https://example.com"  # Placeholder - will be updated with real URLs
        
        st["args"] = args
        out.append(st)
    
    return out

# ───────────────────────────  plan builder  ─────────────────────────────────
def build_plan(pj: Dict[str, Any], topic: str) -> UniversalPlan:
    """Build a UniversalPlan from LLM JSON"""
    plan = UniversalPlan(
        pj.get("title", "Untitled Research"),
        description="LLM-generated research plan",
        tags=["llm", "research", "deep-research"]
    )
    
    # Track step IDs for dependencies
    idmap = {}
    
    for i, st in enumerate(sanitise(pj.get("steps", []), topic), 1):
        # Convert step number dependencies to actual step IDs
        deps = [idmap[d] for d in st.get("depends_on", []) if d in idmap]
        
        # For summarize tool, we'll pass variable references that the executor can resolve
        if st["tool"] == "summarize":
            # Create variable references for previous results
            prev_results = []
            for j in range(1, i):  # All previous steps
                prev_results.append(f"${{result_{j}}}")
            
            st["args"]["documents"] = prev_results
            st["args"]["topic"] = topic
        
        # Add the step
        sid = plan.add_function_step(
            title=st["title"],
            function=st["tool"],
            args=st["args"],
            depends_on=deps,
            result_variable=f"result_{i}"
        )
        
        # Track the step ID
        idmap[i] = sid
    
    plan.save()
    return plan

# ───────────────────────────  executor registration  ────────────────────────
def register_functions(executor: UniversalExecutor):
    """Register all research functions with the executor"""
    executor.register_function("search", search_fn)
    executor.register_function("visit_url", visit_fn)
    executor.register_function("weather", weather_fn)
    executor.register_function("scrape_url", scrape_fn)
    executor.register_function("summarize", summarize_fn)

# ───────────────────────────  main async  ───────────────────────────────────
async def main(topic: str, live: bool):
    """Main research workflow"""
    logger.info(f"\n🔬 DEEP RESEARCH: {topic}\n")
    
    # 1. Generate plan via LLM
    logger.info("📋 Generating research plan...")
    plan_json = await llm_json(topic, live)
    
    # 2. Build executable plan
    logger.info("🏗️ Building execution plan...")
    plan = build_plan(plan_json, topic)
    
    logger.info(f"\n📊 PLAN OVERVIEW:")
    logger.info(plan.outline())
    
    # 3. Set up executor
    logger.info("\n⚙️ Setting up executor...")
    executor = UniversalExecutor(graph_store=plan._graph)
    register_functions(executor)
    
    # 4. Execute plan
    logger.info("\n🚀 Executing research plan...")
    result = await executor.execute_plan(plan)
    
    # 5. Handle results
    if not result["success"]:
        logger.error(f"❌ Research failed: {result['error']}")
        return
    
    logger.info("\n" + "="*60)
    logger.info("📊 RESEARCH RESULTS")
    logger.info("="*60)
    
    # Display all variables
    for key, value in result["variables"].items():
        logger.info(f"\n🔹 {key}:")
        if isinstance(value, (dict, list)):
            logger.info(json.dumps(value, indent=2))
        else:
            logger.info(str(value))
    
    logger.info("\n✅ Deep research complete!")
    
    # Extract and display final summary if available
    summary_keys = [k for k in result["variables"].keys() if "summary" in k.lower()]
    if summary_keys:
        final_summary = result["variables"][summary_keys[-1]]
        if isinstance(final_summary, dict) and "summary" in final_summary:
            logger.info("\n" + "="*60)
            logger.info("📝 FINAL SUMMARY")
            logger.info("="*60)
            logger.info(final_summary["summary"])

# ───────────────────────────  CLI  ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Research Tool using UniversalPlan")
    parser.add_argument("topic", help="Research topic")
    parser.add_argument("--live", action="store_true", 
                       help="Use real OpenAI API for plan generation")
    args = parser.parse_args()
    
    if args.live and (not AsyncOpenAI or not os.getenv("OPENAI_API_KEY")):
        parser.error("--live requires OPENAI_API_KEY environment variable and openai package")
    
    asyncio.run(main(args.topic, args.live))