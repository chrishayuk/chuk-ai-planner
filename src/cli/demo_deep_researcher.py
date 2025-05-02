#!/usr/bin/env python
"""
cli/deep_research_cli_full.py
Deep-crawl research CLI (compact GPT context + smarter query generation)
"""

from __future__ import annotations
import argparse, asyncio, json, os, uuid, warnings, logging, re
from dataclasses import dataclass
from typing import Any, Dict, List, Set
from urllib.parse import urlparse, parse_qs, unquote
from dotenv import load_dotenv

load_dotenv()

# ── external deps ────────────────────────────────────────────────
import httpx, lxml.html
try:
    from readability import Document
except ImportError:
    try:
        from readability.readability import Document
    except ImportError as e:
        raise RuntimeError("pip install readability-lxml") from e

from sample_tools import WeatherTool, SearchTool, VisitURL  # auto-register

# A2A plumbing
from a2a_session_manager.storage import InMemorySessionStore, SessionStoreProvider
from a2a_session_manager.models.session import Session
from chuk_ai_planner.planner import Plan
from chuk_ai_planner.models import ToolCall
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind
from chuk_ai_planner.processor import GraphAwareToolProcessor
from chuk_ai_planner.utils.visualization import print_session_events, print_graph_structure
from chuk_ai_planner.utils.registry_helpers import execute_tool
from chuk_ai_planner.agents.plan_agent import PlanAgent


# ── configuration & logger ───────────────────────────────────────
@dataclass
class CrawlConfig:
    max_search_pages: int = 2
    links_per_page:   int = 5
    max_follow_links: int = 10
    max_rounds:       int = 3
    verbose:          bool = True

logger = logging.getLogger("deepresearch")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)


# ── tool schema & system prompt ──────────────────────────────────
ALLOWED_TOOLS = {"weather", "search", "visit_url", "scrape_url"}
TOOL_SCHEMA = {
    "weather":   {"location": lambda v: isinstance(v, str) and v.strip()},
    "search":    {"query":    lambda v: isinstance(v, str) and v.strip()},
    "visit_url": {"url":      lambda v: isinstance(v, str) and v.strip()},
    "scrape_url": {"url":     lambda v: isinstance(v, str) and v.strip()},
}
def _tool_sig(n,s): return f"  – {n} "+"{"+", ".join(f'{k}:str' for k in s)+"}"
SYS_MSG = ("You are an assistant that writes a JSON *plan* using only these tools:\n"
           + "\n".join(_tool_sig(n, TOOL_SCHEMA[n]) for n in ALLOWED_TOOLS) +
           "\nReturn ONLY a JSON object of the form\n"
           "{\n  \"title\": str,\n  \"steps\": [ {\"title\": str, \"tool\": str, "
           "\"args\": object, \"depends_on\": []} ]\n}")

def validate_step(step: Dict[str, Any]):
    tool = step.get("tool")
    if tool not in ALLOWED_TOOLS: return False, f"{tool!r} not allowed"
    spec, args = TOOL_SCHEMA[tool], step.get("args", {})
    miss  = [k for k in spec if k not in args]
    extra = [k for k in args if k not in spec]
    bad   = [k for k,fn in spec.items() if k in args and not fn(args[k])]
    if miss:  return False, f"{tool}: missing {miss}"
    if extra: return False, f"{tool}: unknown {extra}"
    if bad:   return False, f"{tool}: invalid {bad}"
    return True, ""


# ── helper: compact docs for GPT context ─────────────────────────
def _compact_docs(docs: List[Dict[str, Any]], max_chars: int = 300):
    out=[]
    for d in docs:
        nd={}
        for k,v in d.items():
            if isinstance(v,str) and len(v)>max_chars:
                nd[k]=v[:max_chars]+" …"
            else:
                nd[k]=v
        out.append(nd)
    return out

def _dedup_queries(existing: Set[str], new_queries: List[str],
                   min_len: int = 6) -> List[str]:
    """Remove duplicates/echoes and very short strings."""
    lower_existing = {q.lower() for q in existing}
    cleaned = []
    for q in new_queries:
        q_clean = q.strip()
        lc      = q_clean.lower()
        if len(lc) < min_len:
            continue
        if any(lc in ex or ex in lc for ex in lower_existing):
            continue
        cleaned.append(q_clean)
    return cleaned


# ── adapter & registry ───────────────────────────────────────────
async def fetch_article(url: str, timeout: float = 30):
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
            html = (await c.get(url)).text
        doc = Document(html)
        text = lxml.html.fromstring(doc.summary()).text_content()
        return {"title": doc.short_title() or "untitled",
                "content": text[:10_000], "url": url, "status": 200}
    except Exception as e:
        return {"url": url, "status": "error", "error": str(e)}

async def _adapter(name:str,args:Dict[str,Any],cfg:CrawlConfig|None=None):
    if name in {"visit_url","scrape_url"} and "url" in args:
        if "duckduckgo.com/l/" in args["url"]:
            try: args["url"]=unquote(parse_qs(urlparse(args["url"]).query)["uddg"][0])
            except Exception: pass
    if name=="scrape_url": return await fetch_article(args["url"])
    tc={"id":uuid.uuid4().hex,"type":"function",
        "function":{"name":name,"arguments":json.dumps(args)}}
    try: return await execute_tool(tc,None,None)
    except Exception as e: return {"error":str(e)}

def register_tools(proc: GraphAwareToolProcessor,cfg:CrawlConfig):
    from chuk_tool_processor.registry import default_registry
    warnings.filterwarnings("ignore",category=UserWarning)
    for t in ("weather","search","visit_url"):
        tool=default_registry.get_tool(t)
        proc.register_tool(t,lambda a,_n=t:_adapter(_n,a,cfg))
    proc.register_tool("scrape_url",lambda a:_adapter("scrape_url",a,cfg))


# ── paging & URL selection ───────────────────────────────────────
async def paged_search(q:str,cfg:CrawlConfig):
    out=[]
    for page in range(cfg.max_search_pages):
        r=await _adapter("search",{"query":q,"page":page,"num":cfg.links_per_page},cfg)
        if r and "results" in r: out.extend(r["results"])
    return out

def choose_urls(items,cfg):
    chosen,doms=[],{}
    for it in items:
        url=it.get("url"); dom=urlparse(url).netloc.split(":")[0] if url else ""
        if not url or doms.get(dom,0)>=2: continue
        doms[dom]=doms.get(dom,0)+1; chosen.append(url)
        if len(chosen)>=cfg.max_follow_links: break
    return chosen


# ── tracker ───────────────────────────────────────────────────────
class ResearchTracker:
    def __init__(self):
        self.search_queries:set[str]=set()
        self.visited_urls:set[str]=set()
        self.all_results:List[Dict]=[]
        self.all_graphs=[]
    def add_search_query(self,q): self.search_queries.add(q.lower())
    def add_url(self,u):          self.visited_urls.add(u)


# ── plan helpers ─────────────────────────────────────────────────
async def plan_for_entities(agent,topic,ents):
    if not ents: return None
    steps=[{"title":f"Research: {e}","tool":"search",
            "args":{"query":f"{e} {topic}"},"depends_on":[]} for e in ents[:3]]
    return {"title":f"Research on {topic} – extra angles","steps":steps}
def plan_visit_urls(title,urls):
    if not urls: return None
    return {"title":title,
            "steps":[{"title":f"Visit: {urlparse(u).netloc}","tool":"scrape_url",
                      "args":{"url":u},"depends_on":[]} for u in urls]}


# ── execute plan ────────────────────────────────────────────────
async def execute_plan(plan_json,session_id,tracker,cfg):
    plan=Plan(plan_json["title"]); [plan.step(s["title"]).up() for s in plan_json["steps"]]
    pid=plan.save()
    idx2step={n.data["index"]:n.id for n in plan.graph.nodes.values()
              if n.__class__.__name__=="PlanStep"}
    for i,s in enumerate(plan_json["steps"],1):
        (tracker.add_search_query if s["tool"]=="search"
         else tracker.add_url)(s["args"][list(s["args"])[0]])
        tc=ToolCall(data={"name":s["tool"],"args":s["args"]})
        plan.graph.add_node(tc)
        plan.graph.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK,
                                      src=idx2step[str(i)],dst=tc.id))
    tracker.all_graphs.append(plan.graph)
    proc=GraphAwareToolProcessor(session_id,plan.graph); register_tools(proc,cfg)
    raw=[]
    def _cb(_,tool_results): raw.extend(tool_results); return True
    await proc.process_plan(pid,"assistant",lambda _:None,on_step=_cb)
    serial=[tr.result if hasattr(tr,"result") else tr for tr in raw]
    tracker.all_results.extend(serial)
    return serial


# ── GPT helpers ──────────────────────────────────────────────────
async def find_new_queries(goal:str,tracker:ResearchTracker)->List[str]:
    from openai import AsyncOpenAI
    client=AsyncOpenAI()
    payload={"docs":_compact_docs(tracker.all_results[-15:])}
    prompt=(f"You are helping research '{goal}'. JSON of reviewed sources:\n"
            f"```json\n{json.dumps(payload,indent=2)}```\n\n"
            "Suggest up to three *new* web search queries **as a JSON list**.\n"
            "Do **not** repeat the phrase of the original request or close variants.")
    try:
        rsp=await client.chat.completions.create(model="gpt-4o-mini",temperature=0.3,
                messages=[{"role":"user","content":prompt}])
        raw=rsp.choices[0].message.content.strip()
        if raw.startswith("["):
            candidate=json.loads(raw)
        else:
            candidate=re.findall(r'\"([^\"]+)\"',raw) or re.findall(r'^[*-]\\s*(.+)$',raw,re.M)
        cleaned=_dedup_queries(tracker.search_queries,candidate)
        if not cleaned:
            # fallback generic angles
            base=goal.split()[0]
            cleaned=[f"{base} patents",f"{base} keynote talks",f"{base} media interviews"]
        return cleaned[:3]
    except Exception as e:
        logger.warning(f"GPT error: {e}")
        return []

async def build_summary(goal:str,tracker:ResearchTracker):
    from openai import AsyncOpenAI
    client=AsyncOpenAI()
    prompt=(f"Write a focused overview of **{goal}**.\n"
            f"Sources (JSON):\n```json\n{json.dumps(_compact_docs(tracker.all_results[-40:]),indent=2)}```")
    try:
        rsp=await client.chat.completions.create(model="gpt-4o-mini",temperature=0,
                messages=[{"role":"user","content":prompt}])
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        return f"[summary error: {e}]"


# ── main loop ────────────────────────────────────────────────────
async def run(user_prompt:str,cfg:CrawlConfig):
    SessionStoreProvider.set_store(InMemorySessionStore())
    session=Session(); SessionStoreProvider.get_store().save(session)
    tracker=ResearchTracker(); agent=PlanAgent(system_prompt=SYS_MSG,validate_step=validate_step)
    logger.info("\n🔍  STARTING DEEPER RESEARCH...\n")
    round_no=0; queue=[await agent.plan(user_prompt)]

    while queue and round_no<=cfg.max_rounds:
        plan_json=queue.pop(0)
        logger.info(f"\n📋  EXECUTING PLAN (round {round_no+1})\n{json.dumps(plan_json,indent=2)}\n")
        results=await execute_plan(plan_json,session.id,tracker,cfg)
        logger.info(f"✅  Results fetched: {len(results)}")

        if any(s["tool"]=="search" for s in plan_json["steps"]):
            serp=[itm for r in results if "results" in r for itm in r["results"]]
            vp=plan_visit_urls(f"Explore URLs – round {round_no+1}",choose_urls(serp,cfg))
            if vp: queue.append(vp); continue

        new_qs=await find_new_queries(user_prompt,tracker)
        if new_qs:
            extra=await plan_for_entities(agent,user_prompt,new_qs)
            if extra: queue.append(extra)
        else: break
        round_no+=1

    logger.info("\n🔎  FINAL SUMMARY\n"+await build_summary(user_prompt,tracker))
    logger.info("\n📊  STATS"
                f"\n• Queries: {len(tracker.search_queries)}"
                f"\n• URLs visited: {len(tracker.visited_urls)}"
                f"\n• Docs: {len(tracker.all_results)}")
    print_session_events(session)
    if tracker.all_graphs:
        print_graph_structure(tracker.all_graphs[0])


# ── CLI ──────────────────────────────────────────────────────────
def cli():
    p=argparse.ArgumentParser(description="Deep research CLI")
    p.add_argument("query"); p.add_argument("--pages",type=int,default=2)
    p.add_argument("--links",type=int,default=5); p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--quiet",action="store_true")
    a=p.parse_args()
    if not os.getenv("OPENAI_API_KEY"): p.error("OPENAI_API_KEY missing")
    cfg=CrawlConfig(a.pages,a.links,a.links*a.pages,a.rounds,not a.quiet)
    if a.quiet: logger.setLevel(logging.WARNING)
    asyncio.run(run(a.query,cfg))

if __name__=="__main__":
    cli()
