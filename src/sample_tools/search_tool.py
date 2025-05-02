"""
sample_tools/search_tool.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple web-search tool that scrapes DuckDuckGo’s HTML results page
(so it still works without an API key).  Returns up to *max_results*
organic links.

Install deps once:

    pip install requests beautifulsoup4

"""

from __future__ import annotations
import time, requests, re
from typing import Dict, List
from bs4 import BeautifulSoup
from chuk_tool_processor.registry.decorators import register_tool
from chuk_tool_processor.models.validated_tool import ValidatedTool


@register_tool(name="search")
class SearchTool(ValidatedTool):
    """Return the first *n* DuckDuckGo results for a query."""

    # ── validated arguments & result ─────────────────────────────────
    class Arguments(ValidatedTool.Arguments):
        query: str
        max_results: int = 3

    class Result(ValidatedTool.Result):
        results: List[Dict]

    # ----------------------------------------------------------------
    @staticmethod
    def _search_ddg_html(query: str, max_results: int) -> List[Dict]:
        url = "https://duckduckgo.com/html/?q=" + requests.utils.quote(query)
        html = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (a2a-demo/1.0)"},
            timeout=10,
        ).text

        soup = BeautifulSoup(html, "html.parser")
        hits: List[Dict] = []

        for res in soup.select("div.result")[:max_results]:
            a_title = res.select_one("a.result__a")
            url_tag = res.select_one("a.result__url")
            snippet = res.select_one("a.result__snippet")

            if not (a_title and url_tag and snippet):
                continue

            # duckduckgo rewrites URLs; clean them up a bit
            href = url_tag["href"]
            href = re.sub(r"^/l/\\?uddg=", "", href)
            href = requests.utils.unquote(href)

            hits.append(
                {
                    "title": a_title.get_text(" ", strip=True),
                    "url":   "https://" + href.lstrip("/"),
                    "snippet": snippet.get_text(" ", strip=True),
                }
            )
        return hits or [
            {
                "title": "No results",
                "url": "",
                "snippet": f"No results found for '{query}'",
            }
        ]

    # ── synchronous execution (required) ─────────────────────────────
    def run(self, **kwargs) -> Dict:
        args = self.Arguments(**kwargs)
        # tiny sleep so demos don’t hammer DDG
        time.sleep(0.4)
        hits = self._search_ddg_html(args.query, args.max_results)
        return self.Result(results=hits).model_dump()

    # ── optional async wrapper (so callers can await arun) ───────────
    async def arun(self, **kwargs) -> Dict:
        from asyncio import get_running_loop

        loop = get_running_loop()
        return await loop.run_in_executor(None, lambda: self.run(**kwargs))
