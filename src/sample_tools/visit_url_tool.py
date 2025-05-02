"""
sample_tools/visit_url_tool.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple web-page fetcher that retrieves the title and a snippet of content
from a URL. Handles DuckDuckGo redirect URLs and provides robust error handling.

"""

from __future__ import annotations
import re
import urllib.parse
from typing import Dict, Any

import requests
from chuk_tool_processor.registry.decorators import register_tool
from chuk_tool_processor.models.validated_tool import ValidatedTool

@register_tool(name="visit_url")
class VisitURL(ValidatedTool):
    """Fetch a web-page and return title + basic content snippet."""

    # ── validated arguments & result ─────────────────────────────────
    class Arguments(ValidatedTool.Arguments):
        url: str

    class Result(ValidatedTool.Result):
        title: str
        first_200_chars: str
        url: str
        status: int

    # ── helper methods ────────────────────────────────────────────────
    def _extract_real_url(self, url: str) -> str:
        """Extract the actual URL from DuckDuckGo redirect URLs."""
        try:
            if "duckduckgo.com/l/" in url:
                parsed = urllib.parse.urlparse(url)
                params = urllib.parse.parse_qs(parsed.query)
                
                if "uddg" in params and params["uddg"]:
                    return urllib.parse.unquote(params["uddg"][0])
        except Exception:
            pass
        
        return url
    
    def _extract_title(self, html: str) -> str:
        """Extract title from HTML content."""
        try:
            match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
            if match:
                # Clean up the title
                title = match.group(1).strip()
                # Remove excess whitespace and normalize
                title = re.sub(r'\s+', ' ', title)
                return title
        except Exception:
            pass
        
        return ""
    
    def _extract_clean_text(self, html: str) -> str:
        """Extract readable text from HTML content."""
        try:
            # Remove scripts and style elements
            html = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style[^>]*>.*?</style>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove HTML comments
            html = re.sub(r'<!--.*?-->', ' ', html, flags=re.DOTALL)
            
            # Remove remaining HTML tags
            text = re.sub(r'<[^>]+>', ' ', html)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception:
            return ""

    # ── synchronous execution (required) ─────────────────────────────
    def run(self, **kwargs) -> Dict:
        """Fetch a webpage and extract useful information."""
        args = self.Arguments(**kwargs)
        url = args.url
        
        # Extract real URL from DuckDuckGo
        real_url = self._extract_real_url(url)
        
        # Ensure URL has protocol
        if not real_url.startswith(("http://", "https://")):
            real_url = "https://" + real_url
        
        # Initialize default values
        title = real_url
        content_preview = ""
        status = 0
        
        # Attempt to fetch content
        try:
            # Suppress SSL warnings
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # Fetch the URL
            response = requests.get(
                real_url,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36"
                },
                verify=False,
                allow_redirects=True
            )
            
            status = response.status_code
            
            if status == 200:
                # Extract title and content
                title = self._extract_title(response.text) or real_url
                content_preview = self._extract_clean_text(response.text)[:200]
            else:
                content_preview = f"Failed to fetch content (HTTP {status})"
                
        except Exception as e:
            content_preview = f"Error accessing page: {str(e)[:100]}"
        
        # Return the result
        result = self.Result(
            title=title,
            first_200_chars=content_preview,
            url=real_url,
            status=status
        ).model_dump()
        
        return result

    # ── optional async wrapper (so callers can await arun) ───────────
    async def arun(self, **kwargs) -> Dict:
        """Async wrapper around the synchronous run method."""
        from asyncio import get_running_loop
        
        loop = get_running_loop()
        return await loop.run_in_executor(None, lambda: self.run(**kwargs))