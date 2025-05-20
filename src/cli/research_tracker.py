#!/usr/bin/env python
# cli/research_tracker.py
"""
Research Tracker Module
======================

This module provides a class to track the progress and results of
multi-round research processes.
"""

import time
from typing import Dict, List, Set, Any, Optional


class ResearchTracker:
    """
    Track and manage research progress and results across multiple rounds.
    
    The ResearchTracker keeps track of searches, visited URLs, and all research
    results organized by rounds. It provides functionality to analyze research
    coverage and extract insights from the collected data.
    """
    
    def __init__(self):
        """Initialize a new research tracker."""
        self.rounds: List[Dict[str, Any]] = []
        self.current_round: int = 0
        self.results: List[Dict[str, Any]] = []
        self.search_queries: Set[str] = set()
        self.visited_urls: Set[str] = set()
        self.url_cache: Dict[str, List[Dict[str, Any]]] = {}  # Cache for extracted URLs from searches
    
    def start_round(self) -> int:
        """
        Start a new research round and return the round number.
        
        Returns:
            int: The current round number
        """
        self.current_round += 1
        self.rounds.append({
            "round": self.current_round,
            "steps": [],
            "start_time": time.time()
        })
        return self.current_round
    
    def add_search_query(self, query: str) -> None:
        """
        Add a search query to the tracked queries.
        
        Args:
            query: The search query to track
        """
        self.search_queries.add(query.lower())
    
    def add_url(self, url: str) -> None:
        """
        Add a URL to the visited URLs list.
        
        Args:
            url: The URL to track
        """
        self.visited_urls.add(url)
    
    def add_result(self, tool: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Add a research result with its metadata.
        
        Args:
            tool: The tool used (e.g., "search", "visit_url")
            args: The arguments passed to the tool
            result: The result returned by the tool
        """
        # Track the query if it's a search
        if tool == "search" and "query" in args:
            self.add_search_query(args["query"])
            
            # Cache extracted URLs from this search for later use
            if "extracted_urls" in result:
                query = args["query"]
                self.url_cache[query] = result["extracted_urls"]
        
        # Track URL if appropriate
        if tool == "visit_url" and "url" in args:
            self.add_url(args["url"])
        
        self.results.append({
            "round": self.current_round,
            "tool": tool,
            "args": args,
            "result": result,
            "timestamp": time.time()
        })
    
    def get_cached_urls(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get cached URLs either for a specific query or all queries.
        
        Args:
            query: Optional specific query to get URLs for
            
        Returns:
            List of URL information dictionaries
        """
        if query:
            return self.url_cache.get(query, [])
        else:
            # Flatten all URLs from all queries
            all_urls = []
            for urls in self.url_cache.values():
                all_urls.extend(urls)
            return all_urls
    
    def get_round_results(self, round_num: int) -> List[Dict[str, Any]]:
        """
        Get all results from a specific round.
        
        Args:
            round_num: The round number to get results for
            
        Returns:
            List of result dictionaries from the specified round
        """
        return [r for r in self.results if r["round"] == round_num]
    
    def get_last_n_results(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the last N results across all rounds.
        
        Args:
            n: Number of recent results to return
            
        Returns:
            List of the most recent result dictionaries
        """
        return self.results[-n:] if len(self.results) >= n else self.results
    
    def get_results_by_tool(self, tool_name: str) -> List[Dict[str, Any]]:
        """
        Get all results for a specific tool.
        
        Args:
            tool_name: The name of the tool (e.g., "search", "visit_url")
            
        Returns:
            List of result dictionaries for the specified tool
        """
        return [r for r in self.results if r.get("tool") == tool_name]
    
    def get_summary(self) -> Dict[str, int]:
        """
        Get a summary of research statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            "rounds": len(self.rounds),
            "total_steps": len(self.results),
            "queries": len(self.search_queries),
            "urls": len(self.visited_urls),
            "search_results": len([r for r in self.results if r.get("tool") == "search"]),
            "url_visits": len([r for r in self.results if r.get("tool") == "visit_url"])
        }
    
    def format_summary_for_prompt(self, max_results: int = 7) -> str:
        """
        Format recent findings for inclusion in an LLM prompt.
        
        Args:
            max_results: Maximum number of results to include
            
        Returns:
            Formatted string with recent findings
        """
        summary = "\n\nPrevious findings:"
        
        # Group results by category for better organization
        search_findings = []
        url_findings = []
        
        for result in self.results[-max_results:]:  # Last N results
            tool = result.get("tool", "")
            if tool == "search":
                query_text = result.get('args', {}).get('query', '')
                search_findings.append(f"- Searched for '{query_text}'")
                
                # Add top results
                search_results = result.get('result', {}).get('results', [])
                for i, sr in enumerate(search_results[:3], 1):  # First 3 search results
                    title = sr.get('title', '')
                    snippet = sr.get('snippet', '')[:100]
                    search_findings.append(f"  * {title}: {snippet}")
            
            elif tool == "visit_url":
                url = result.get('args', {}).get('url', '')
                title = result.get('result', {}).get('title', '')
                
                # Use the content field if available
                content = result.get('result', {}).get('content', '')
                if not content:  # Fall back if needed
                    content = result.get('result', {}).get('first_200_chars', '')
                
                if content and not content.startswith("Error") and not content.startswith("URL must be"):
                    url_findings.append(f"- Visited: {url}")
                    url_findings.append(f"  * Title: {title}")
                    url_findings.append(f"  * Content preview: {content[:150]}...")
        
        # Add findings to prompt
        if search_findings:
            summary += "\n\nSearch findings:"
            for finding in search_findings:
                summary += f"\n{finding}"
        
        if url_findings:
            summary += "\n\nURL findings:"
            for finding in url_findings:
                summary += f"\n{finding}"
        
        return summary
    
    def get_research_gaps(self, round_num: int) -> str:
        """
        Generate suggestions for research gaps to explore.
        
        Args:
            round_num: Current round number
            
        Returns:
            String with research gap suggestions
        """
        gaps = "\n\nResearch gaps to explore next:"
        
        # Analyze existing findings to identify gaps
        if round_num == 2:
            # In round 2, encourage exploration of different content types
            gaps += "\n- Look for different types of content (videos, academic publications, social media)"
            gaps += "\n- Explore more recent information and updates"
            gaps += "\n- Consider searching for specific aspects not yet covered"
        elif round_num == 3:
            # In round 3, encourage deepening and verifying information
            gaps += "\n- Focus on verifying key facts from multiple sources"
            gaps += "\n- Look for more detailed or technical information on important aspects"
            gaps += "\n- Try to find the most authoritative sources on this topic"
        
        return gaps