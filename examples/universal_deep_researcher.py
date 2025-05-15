# #!/usr/bin/env python
# # examples/universal_deep_researcher.py
# """
# Deep-Researcher with UniversalPlan – zero .Arguments issues.
# Every capability is a *function* registered via `executor.register_function`,
# so UniversalExecutor never tries to inspect a tool class.
# """

# from __future__ import annotations
# import argparse, asyncio, json, logging, os, re
# from typing import Any, Dict, List
# from urllib.parse import urlparse, parse_qs, unquote
# from dotenv import load_dotenv; load_dotenv()

# from chuk_ai_planner.planner.universal_plan          import UniversalPlan
# from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor

# # ───────────────────────────  sample_tools wrappers  ─────────────────────────
# from sample_tools import SearchTool, VisitURL, WeatherTool

# async def search_fn(**kw):   return await SearchTool.run(kw)
# async def visit_fn(**kw):    return await VisitURL.run(kw)
# async def weather_fn(**kw):  return await WeatherTool.run(kw)

# # ───────────────────────────  scrape_url  ────────────────────────────────────
# import httpx, lxml.html
# from readability import Document
# async def scrape_fn(**kw):
#     url = kw.get("url", "")
#     if "duckduckgo.com/l/" in url:
#         try: url = unquote(parse_qs(urlparse(url).query)["uddg"][0])
#         except Exception: pass
#     try:
#         async with httpx.AsyncClient(timeout=30,follow_redirects=True) as c:
#             html = (await c.get(url)).text
#         doc  = Document(html)
#         text = lxml.html.fromstring(doc.summary()).text_content()
#         return {"title":doc.short_title() or "untitled",
#                 "content":text[:10_000],"url":url}
#     except Exception as e:
#         return {"url":url,"error":str(e)}

# # ───────────────────────────  summarize  ─────────────────────────────────────
# async def summarize_fn(**kw):
#     topic     = kw.get("topic","")
#     documents = kw.get("documents",[])
#     bullets="\n".join(
#         f"- {d.get('title',d.get('url','?'))}"
#         if isinstance(d,dict) else f"- {str(d)[:60]}…" for d in documents[:8])
#     return {"summary":f"Summary for {topic}\n{bullets}",
#             "count":len(documents)}

# # attach dummy attribute so introspection (if any) is satisfied
# summarize_fn.Arguments = dict      # type: ignore[attr-defined]

# # ───────────────────────────  logging  ───────────────────────────────────────
# logger = logging.getLogger("deepresearch")
# logger.setLevel(logging.INFO)
# h=logging.StreamHandler(); h.setFormatter(logging.Formatter("%(message)s"))
# logger.handlers.clear(); logger.addHandler(h)

# # ───────────────────────────  LLM helper  ────────────────────────────────────
# try:
#     from openai import AsyncOpenAI
# except ImportError:
#     AsyncOpenAI = None

# SYS=("Return ONLY JSON {title, steps}. Each step {title, tool, args, depends_on}. "
#      "Allowed: search, visit_url, scrape_url, weather, summarize.")

# async def llm_json(topic:str, live:bool)->Dict[str,Any]:
#     if live and AsyncOpenAI and os.getenv("OPENAI_API_KEY"):
#         c=AsyncOpenAI(); logger.info("📡  OpenAI plan request")
#         r=await c.chat.completions.create(
#             model="gpt-4o-mini",temperature=0.3,
#             messages=[{"role":"system","content":SYS},
#                       {"role":"user","content":f"Create a research plan for: {topic}"}])
#         txt=r.choices[0].message.content.strip()
#     else:
#         logger.info("🤖  simulated plan")
#         txt=json.dumps({
#             "title":f"Research on {topic}",
#             "steps":[
#                 {"title":"Initial search","tool":"search","args":{},"depends_on":[]},
#                 {"title":"Scrape first","tool":"scrape_url","args":{},"depends_on":[1]},
#                 {"title":"Summarize","tool":"summarize","args":{},"depends_on":[1,2]}
#             ]})
#     m=re.search(r"```(?:json)?(.*?)```",txt,re.S)
#     return json.loads(m.group(1).strip() if m else txt)

# # ───────────────────────────  sanitise  ──────────────────────────────────────
# ALLOWED={"search","visit_url","scrape_url","weather","summarize"}
# def sanitise(steps:List[Dict[str,Any]],topic:str)->List[Dict[str,Any]]:
#     out=[]
#     for st in steps:
#         tool=(st.get("tool") or "").lower()
#         if tool not in ALLOWED:
#             t=st.get("title","").lower()
#             if "scrape" in t or "url" in t: tool="scrape_url"
#             elif "visit"  in t:             tool="visit_url"
#             elif "weather"in t:             tool="weather"
#             elif "summary"in t:             tool="summarize"
#             else:                           tool="search"
#         st["tool"]=tool
#         args=st.get("args") if isinstance(st.get("args"),dict) else {}
#         if tool=="search" and not args.get("query"): args["query"]=topic
#         if tool in {"visit_url","scrape_url"} and not args.get("url"):
#             args["url"]="https://example.com"
#         st["args"]=args
#         out.append(st)
#     return out

# # ───────────────────────────  plan builder  ─────────────────────────────────
# def build_plan(pj:Dict[str,Any],topic:str)->UniversalPlan:
#     plan=UniversalPlan(pj.get("title","Untitled"),
#                        description="LLM research",tags=["llm","research"])
#     idmap={}; prev=[]
#     for i,st in enumerate(sanitise(pj.get("steps",[]),topic),1):
#         deps=[idmap[d] for d in st.get("depends_on",[]) if d in idmap]
#         if st["tool"]=="summarize":
#             st["args"]={"topic":topic,"documents":prev.copy()}
#         sid=plan.add_function_step(st["title"],st["tool"],st["args"],
#                                    depends_on=deps,result_variable=f"result_{i}")
#         if st["tool"]!="summarize": prev.append(f"$result_{i}")
#         idmap[i]=sid
#     plan.save(); return plan

# # ───────────────────────────  executor registration  ────────────────────────
# def register(ex:UniversalExecutor):
#     ex.register_function("search",     search_fn)
#     ex.register_function("visit_url",  visit_fn)
#     ex.register_function("weather",    weather_fn)
#     ex.register_function("scrape_url", scrape_fn)
#     ex.register_function("summarize",  summarize_fn)

# # ───────────────────────────  main async  ───────────────────────────────────
# async def main(topic:str, live:bool):
#     pj   = await llm_json(topic, live)
#     plan = build_plan(pj, topic)
#     logger.info("\n"+plan.outline())

#     ex=UniversalExecutor(graph_store=plan._graph)
#     register(ex)

#     res=await ex.execute_plan(plan)
#     if not res["success"]:
#         logger.error(res["error"]); return
#     for k,v in res["variables"].items():
#         logger.info(f"\n{k} → {json.dumps(v,indent=2) if isinstance(v,(dict,list)) else v}")
#     logger.info("\n✅ Research complete")

# # ───────────────────────────  CLI  ──────────────────────────────────────────
# if __name__=="__main__":
#     p=argparse.ArgumentParser()
#     p.add_argument("--topic", required=True)
#     p.add_argument("--live",  action="store_true")
#     a=p.parse_args()
#     if a.live and (not AsyncOpenAI or not os.getenv("OPENAI_API_KEY")):
#         p.error("--live requires OPENAI_API_KEY and openai package")
#     asyncio.run(main(a.topic,a.live))
