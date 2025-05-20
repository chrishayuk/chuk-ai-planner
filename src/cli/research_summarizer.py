#!/usr/bin/env python
# research_summarizer.py
"""
Research Summarizer Module
=========================

This module provides functionality to create comprehensive summaries
of research findings.
"""

from typing import Dict, Any, List
from openai import AsyncOpenAI


async def generate_research_summary(query, tracker):
    """
    Generate a comprehensive summary of research findings.
    
    Args:
        query: The original research query
        tracker: The ResearchTracker containing results
        
    Returns:
        String containing the formatted research summary
    """
    try:
        # Create a prompt for the summary
        prompt = f"Create a comprehensive summary about '{query}' based on these research findings. The summary should be well-structured, organized chronologically where appropriate, and should cover all major aspects of the topic:\n\n"
        
        # Organize findings by content type and date for better presentation
        webpages = []
        social_media = []
        videos = []
        search_results = []
        
        # Process and organize results
        for result in tracker.results:
            tool = result.get("tool", "")
            if tool == "search":
                # Add to search_results
                search_content = {
                    "query": result.get("args", {}).get("query", ""),
                    "results": []
                }
                
                for sr in result.get("result", {}).get("results", []):
                    search_content["results"].append({
                        "title": sr.get("title", ""),
                        "snippet": sr.get("snippet", ""),
                        "url": sr.get("url", "")
                    })
                
                search_results.append(search_content)
            elif tool == "visit_url":
                # Categorize by content type
                url = result.get("args", {}).get("url", "")
                title = result.get("result", {}).get("title", "")
                content = result.get("result", {}).get("content", "")
                if not content:  # Fall back to first_200_chars if needed
                    content = result.get("result", {}).get("first_200_chars", "")
                
                # Skip if no meaningful content
                if not content or content.startswith("Error") or content.startswith("URL must be"):
                    continue
                
                # Get other metadata if available
                content_type = result.get("result", {}).get("content_type", "webpage")
                publication_date = result.get("result", {}).get("publication_date", "")
                author = result.get("result", {}).get("author", "")
                
                # Create a structured entry
                entry = {
                    "url": url,
                    "title": title,
                    "content": content[:500],  # Limit to 500 chars for prompt size
                    "date": publication_date,
                    "author": author
                }
                
                # Add to appropriate category
                if "video" in url.lower() or "youtube" in url.lower() or content_type == "video":
                    videos.append(entry)
                elif any(sm in url.lower() for sm in ["twitter", "facebook", "linkedin", "instagram"]) or content_type == "social_media":
                    social_media.append(entry)
                else:
                    webpages.append(entry)
        
        # Add structured content to the prompt
        if webpages:
            prompt += "## WEBPAGES/ARTICLES:\n"
            for i, page in enumerate(webpages, 1):
                prompt += f"### {i}. {page['title']}\n"
                if page['date']:
                    prompt += f"Date: {page['date']}\n"
                if page['author']:
                    prompt += f"Author: {page['author']}\n"
                prompt += f"URL: {page['url']}\n"
                prompt += f"Content: {page['content']}\n\n"
        
        if videos:
            prompt += "## VIDEOS:\n"
            for i, video in enumerate(videos, 1):
                prompt += f"### {i}. {video['title']}\n"
                prompt += f"URL: {video['url']}\n"
                prompt += f"Content: {video['content']}\n\n"
        
        if social_media:
            prompt += "## SOCIAL MEDIA CONTENT:\n"
            for i, post in enumerate(social_media, 1):
                prompt += f"### {i}. {post['title']}\n"
                prompt += f"URL: {post['url']}\n"
                prompt += f"Content: {post['content']}\n\n"
        
        if search_results:
            prompt += "## SEARCH RESULTS:\n"
            for i, search in enumerate(search_results, 1):
                prompt += f"### Search Query: {search['query']}\n"
                for j, result in enumerate(search['results'][:3], 1):  # Limit to top 3 results per query
                    prompt += f"{j}. {result['title']}: {result['snippet']}\n"
                prompt += "\n"
        
        # Add guidelines for the summary
        prompt += """
## SUMMARY GUIDELINES:
1. Structure the summary with clear headings and sections
2. Organize information chronologically where appropriate
3. Include all key facts, positions, contributions, and roles
4. Note any contradictions or uncertainties in the information
5. Distinguish between facts and potential speculation
6. Format the summary in a readable, professional style with appropriate headings
7. For people, include a timeline of their career if possible
8. For technical topics, explain key concepts and their relationships
"""
        
        # Call the LLM for the summary
        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "You are an expert researcher who synthesizes information into clear, comprehensive summaries. Include key facts and insights, note any contradictions, and organize the information in a logical structure with appropriate headings. Present information chronologically where appropriate and clearly distinguish between facts and uncertainties."},
                {"role": "user", "content": prompt}
            ],
        )
        
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Unable to generate comprehensive summary due to an error."


def format_research_statistics(tracker):
    """
    Format research statistics for display.
    
    Args:
        tracker: The ResearchTracker containing results
        
    Returns:
        String with formatted statistics
    """
    stats = tracker.get_summary()
    
    output = [
        "\n📊  STATISTICS:",
        f"Completed {stats['rounds']} research rounds with {stats['total_steps']} total steps",
        f"Searched {stats['queries']} unique queries and visited {stats['urls']} URLs"
    ]
    
    # Add additional statistics if available
    if 'search_results' in stats:
        output.append(f"Total search operations: {stats['search_results']}")
    if 'url_visits' in stats:
        output.append(f"Total URL visits: {stats['url_visits']}")
    
    return "\n".join(output)