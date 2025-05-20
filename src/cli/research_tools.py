#!/usr/bin/env python
# research_tools.py
"""
Research Tools Module
====================

This module provides enhanced tool wrappers for search and URL visiting
functionality to support in-depth research.
"""

import re
import requests
from typing import Dict, Any, List, Optional, Callable
from urllib.parse import urlparse


async def register_enhanced_tools(processor, tools_dict, search_tool, visit_url_tool):
    """
    Register enhanced research tools with the processor.
    
    Args:
        processor: The tool processor to register with
        tools_dict: Dictionary to store tool references
        search_tool: Base search tool to enhance
        visit_url_tool: Base URL visitor tool to enhance
        
    Returns:
        The processor with enhanced tools registered
    """
    
    # Enhanced search wrapper with multi-page support and URL extraction
    async def enhanced_search_wrapper(args):
        query = args.get("query", "")
        # Use requested max_results or default to 10
        max_results = args.get("max_results", 10)
        
        print(f"🔍 Enhanced search for: {query} (max results: {max_results})")
        
        # First search with original query
        results = await search_tool.arun(query=query, max_results=max_results)
        
        # Try a second page if needed and we don't have enough results
        if len(results.get("results", [])) < max_results:
            print(f"🔍 Searching second page for: {query}")
            try:
                # Use page=1 (second page) parameter
                page2_args = {"query": query, "max_results": max_results, "page": 1}
                page2_results = await search_tool.arun(**page2_args)
                
                # Combine results while avoiding duplicates
                existing_urls = {r.get("url", "") for r in results.get("results", [])}
                for r in page2_results.get("results", []):
                    if r.get("url") not in existing_urls:
                        results.get("results", []).append(r)
                        existing_urls.add(r.get("url", ""))
            except Exception as e:
                print(f"⚠️ Error fetching second page: {e}")
        
        # Extract and save URLs from search results for use in visit_url steps
        extracted_urls = []
        for idx, result in enumerate(results.get("results", [])):
            url = result.get("url", "")
            if url and not url.startswith("data:"):  # Skip data URLs
                extracted_urls.append({
                    "index": idx + 1,
                    "title": result.get("title", ""),
                    "url": url
                })
        
        # Add extracted URLs to the results for later use
        results["extracted_urls"] = extracted_urls
        
        # Print extracted URLs for debugging
        if extracted_urls:
            print(f"📋 Extracted {len(extracted_urls)} URLs from search results")
            for idx, url_info in enumerate(extracted_urls[:3]):  # Show first 3
                print(f"   {idx+1}. {url_info['url']}")
            if len(extracted_urls) > 3:
                print(f"   ... and {len(extracted_urls)-3} more")
        
        return results
    
    # Enhanced URL wrapper with extended content and fallback to search
    async def enhanced_visit_url_wrapper(args):
        url = args.get("url", "")
        
        # Create a modified version that returns more content
        original_url = url
        print(f"🌐 Visiting URL: {url}")
        
        # Check if URL is usable
        if url.startswith("URL_FROM_STEP_") or not url or (not url.startswith("http") and not url.startswith("https")):
            print(f"⚠️ Invalid URL format: {url}")
            return {
                "title": f"Invalid URL: {url}",
                "content": "URL must be a valid web address starting with http:// or https://",
                "url": url,
                "status": "error"
            }
        
        try:
            # First, add protocol if missing
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
                print(f"🔄 Added https:// prefix, now using: {url}")
            
            # Call the original tool but modify the result to include more content
            original_result = await visit_url_tool.arun(url=url)
            
            # Check if we got a valid result
            if original_result.get("status") == 200:
                # Get the extended content by manually patching the VisitURL tool method
                try:
                    # Use same logic as the VisitURL tool but get more content
                    response = requests.get(
                        url,
                        timeout=15,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36"
                        },
                        verify=False,
                        allow_redirects=True
                    )
                    
                    if response.status_code == 200:
                        # Extract extended content (up to 2000 chars instead of 200)
                        html = response.text
                        
                        # Remove scripts and style elements
                        html = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
                        html = re.sub(r'<style[^>]*>.*?</style>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
                        
                        # Remove HTML comments
                        html = re.sub(r'<!--.*?-->', ' ', html, flags=re.DOTALL)
                        
                        # Remove remaining HTML tags
                        text = re.sub(r'<[^>]+>', ' ', html)
                        
                        # Clean up whitespace
                        text = re.sub(r'\s+', ' ', text).strip()
                        
                        # Get a larger chunk of content
                        extended_content = text[:2000]
                        
                        # Return enhanced result with more content
                        return {
                            "title": original_result.get("title", url),
                            "content": extended_content,
                            "url": url,
                            "status": 200
                        }
                except Exception as e:
                    print(f"⚠️ Error extracting extended content: {e}")
            
            # If we couldn't get extended content, return the original result with renamed field
            return {
                "title": original_result.get("title", url),
                "content": original_result.get("first_200_chars", ""),
                "url": original_result.get("url", url),
                "status": original_result.get("status", 0)
            }
            
        except Exception as e:
            print(f"❌ Error visiting URL: {e}")
            
            # If it's a social media site that's blocking access, try to get info via search
            if any(domain in url.lower() for domain in ["linkedin.com", "twitter.com", "facebook.com"]):
                print(f"🔍 Attempting to get information about {url} via search...")
                
                # Extract domain and potential username from URL
                domain = urlparse(url).netloc
                path = urlparse(url).path.rstrip("/")
                username = path.split("/")[-1] if path else ""
                
                if username and len(username) > 2:
                    # Perform a search for this profile
                    search_query = f"{username} {domain} profile"
                    search_results = await enhanced_search_wrapper({"query": search_query, "max_results": 3})
                    
                    return {
                        "title": f"Information about {url} (via search)",
                        "content": f"Could not directly access the URL. Here's information found via search:\n\n" + 
                                  "\n\n".join([f"- {r.get('title', '')}: {r.get('snippet', '')}" 
                                               for r in search_results.get("results", [])]),
                        "url": url,
                        "status": "search_fallback"
                    }
            
            # Default error response
            return {
                "title": f"Error accessing {url}",
                "content": f"Error: {str(e)}",
                "url": url,
                "status": "error"
            }
    
    # Register the enhanced tools with both the processor and our local dictionary
    processor.register_tool("search", enhanced_search_wrapper)
    processor.register_tool("visit_url", enhanced_visit_url_wrapper)
    
    # Also store in our dictionary for direct access
    tools_dict["search"] = enhanced_search_wrapper
    tools_dict["visit_url"] = enhanced_visit_url_wrapper
    
    print("✅ Enhanced tools registered: search, visit_url")
    return processor


def resolve_url_placeholders(url, step_idx, extracted_urls, tracker):
    """
    Resolve URL placeholders like URL_FROM_STEP_X to actual URLs.
    
    Args:
        url: The URL or placeholder
        step_idx: The current step index
        extracted_urls: Dictionary mapping step indices to extracted URLs
        tracker: The ResearchTracker instance
        
    Returns:
        The resolved URL
    """
    if not isinstance(url, str) or not url.startswith("URL_FROM_STEP_"):
        return url
    
    try:
        # Extract the step number
        step_num = int(url.replace("URL_FROM_STEP_", "").split("_")[0])
        
        # Look up URLs from that step
        if step_num in extracted_urls and extracted_urls[step_num]:
            # Default to first URL from that step
            result_idx = 0
            
            # Check if there's a specific result index specified (URL_FROM_STEP_1_RESULT_2)
            if "_RESULT_" in url:
                try:
                    result_idx = int(url.split("_RESULT_")[1]) - 1
                except:
                    result_idx = 0
            
            # Get the URL
            if result_idx < len(extracted_urls[step_num]):
                real_url = extracted_urls[step_num][result_idx].get("url")
                if real_url:
                    print(f"🔄 Replacing placeholder {url} with actual URL: {real_url}")
                    return real_url
                else:
                    print(f"⚠️ No URL found in result {result_idx+1} from step {step_num}")
            else:
                print(f"⚠️ Result index {result_idx+1} out of range for step {step_num}")
        else:
            # If we can't find a specific reference, check if we have any URLs from any step
            all_urls = []
            for urls in extracted_urls.values():
                all_urls.extend(urls)
            
            if all_urls:
                # Use the first URL we found
                real_url = all_urls[0].get("url")
                print(f"🔄 No URLs from step {step_num}, using URL from other step: {real_url}")
                return real_url
            else:
                # Still no URLs, try to get one from tracker's cache
                cached_urls = tracker.get_cached_urls()
                if cached_urls:
                    real_url = cached_urls[0].get("url")
                    print(f"🔄 Using URL from cache: {real_url}")
                    return real_url
                else:
                    print(f"⚠️ No URLs found from any step to replace {url}")
    except Exception as e:
        print(f"⚠️ Error replacing URL placeholder: {e}")
    
    # Return the original URL if we couldn't resolve it
    return url


def format_tool_result(result_obj):
    """
    Format a tool result for display.
    
    Args:
        result_obj: The result object to format
        
    Returns:
        String representation of the result
    """
    if not result_obj:
        return "No result"
    
    tool = result_obj.tool
    args = result_obj.args
    result = result_obj.result
    
    if tool == "search":
        # Format search results
        output = [f"Search query: {args.get('query', '')}"]
        
        for i, sr in enumerate(result.get('results', [])[:5], 1):
            output.append(f"{i}. {sr.get('title', '')}")
            output.append(f"   {sr.get('snippet', '')[:200]}")
            output.append(f"   URL: {sr.get('url', '')}")
        
        return "\n".join(output)
    
    elif tool == "visit_url":
        # Format URL visit results
        output = [
            f"Visited URL: {args.get('url', '')}",
            f"Title: {result.get('title', '')}"
        ]
        
        # Format content
        content = result.get('content', '')
        if not content:
            content = result.get('first_200_chars', '')
        
        if content:
            if len(content) > 300:
                output.append(f"Content preview: {content[:300]}...")
            else:
                output.append(f"Content: {content}")
        
        # Add metadata if available
        if result.get('publication_date'):
            output.append(f"Publication date: {result.get('publication_date')}")
        if result.get('author'):
            output.append(f"Author: {result.get('author')}")
        
        return "\n".join(output)
    
    else:
        # Generic formatting for other tools
        return f"{tool} - {args}\n{result}"