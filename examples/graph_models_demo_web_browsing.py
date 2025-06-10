#!/usr/bin/env python3
# examples/graph_models_demo_web_browsing.py
"""
Techmeme AI News Browsing Demo: Graph Models for Web Automation
===============================================================

This demo showcases how chuk_ai_planner graph models can represent
a complex web browsing workflow to extract AI news from Techmeme.com.

The demo creates a realistic graph structure showing:
1. Session management for web browsing
2. Multi-step plan for news extraction
3. Tool calls for web scraping operations
4. Error handling and retry logic
5. Data aggregation and analysis
6. Content summarization

Run with: python demo_techmeme_browsing.py
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

# Core imports
from chuk_ai_planner.models import (
    NodeKind, SessionNode, PlanNode, PlanStep,
    UserMessage, AssistantMessage, ToolCall, TaskRun, Summary
)

from chuk_ai_planner.models.edges import (
    EdgeKind, ParentChildEdge, NextEdge, PlanEdge, StepEdge, GraphEdge
)

from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.utils.visualization import print_graph_structure

def create_web_browsing_session():
    """Create a comprehensive web browsing session for AI news extraction."""
    print("🌐 Creating Techmeme AI News Browsing Session")
    print("=" * 60)
    
    graph = InMemoryGraphStore()
    
    # Create the main session
    session = SessionNode(data={
        "session_type": "web_browsing_automation",
        "target_site": "techmeme.com",
        "objective": "extract_latest_ai_news",
        "user_agent": "AI News Assistant Bot v1.0",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "timeout_seconds": 300
    })
    
    # User's request
    user_request = UserMessage(data={
        "content": "Please browse Techmeme and get me the latest AI news. I'm particularly interested in stories about LLMs, AI safety, and new AI product launches.",
        "priority": "high",
        "categories_of_interest": "LLMs, AI safety, AI products, machine learning, generative AI"
    })
    
    # Assistant's response with browsing plan
    assistant_response = AssistantMessage(data={
        "content": "I'll browse Techmeme to collect the latest AI news for you. Let me create a systematic plan to extract and analyze the most relevant stories.",
        "plan_created": True,
        "estimated_duration": "2-3 minutes"
    })
    
    # Main browsing plan
    browsing_plan = PlanNode(data={
        "title": "Techmeme AI News Extraction Plan",
        "description": "Systematic extraction and analysis of AI news from Techmeme",
        "target_url": "https://techmeme.com",
        "extraction_strategy": "comprehensive_scraping",
        "content_filters": "artificial intelligence, AI, machine learning, LLM, generative",
        "max_articles": "20"
    })
    
    # Detailed browsing steps
    browsing_steps = [
        PlanStep(data={
            "index": "1",
            "description": "Navigate to Techmeme homepage and verify page load",
            "step_type": "navigation",
            "url": "https://techmeme.com",
            "success_criteria": "page_title_contains:Techmeme, content_loaded, no_errors"
        }),
        
        PlanStep(data={
            "index": "2", 
            "description": "Extract all article headlines and metadata",
            "step_type": "content_extraction",
            "selectors": "headlines:.storylink, timestamps:.time, sources:.source, discussion_links:.discussion",
            "data_fields": "title, url, timestamp, source, discussion_count"
        }),
        
        PlanStep(data={
            "index": "3",
            "description": "Filter articles for AI-related content",
            "step_type": "content_filtering", 
            "filter_keywords": "AI, artificial intelligence, machine learning, LLM, ChatGPT, GPT, Claude, Gemini, neural, deep learning",
            "filter_method": "keyword_matching_with_fuzzy",
            "minimum_relevance_score": "0.7"
        }),
        
        PlanStep(data={
            "index": "4",
            "description": "Extract detailed content from top AI articles",
            "step_type": "deep_content_extraction",
            "max_articles_to_process": "10",
            "content_elements": "full_text, images, author, publication_date, tags",
            "follow_source_links": "true"
        }),
        
        PlanStep(data={
            "index": "5",
            "description": "Analyze discussion threads for trending topics",
            "step_type": "discussion_analysis",
            "analysis_depth": "moderate",
            "extract_sentiment": "true",
            "identify_trends": "true",
            "max_comments_per_thread": "50"
        }),
        
        PlanStep(data={
            "index": "6",
            "description": "Categorize and rank articles by relevance",
            "step_type": "content_analysis",
            "categorization_scheme": "product_launches, research_breakthroughs, industry_news, policy_regulation, funding",
            "ranking_factors": "recency, discussion_activity, source_authority, keyword_relevance"
        }),
        
        PlanStep(data={
            "index": "7",
            "description": "Generate comprehensive news summary",
            "step_type": "content_synthesis",
            "summary_format": "structured_report",
            "include_sections": "top_stories, trending_topics, key_quotes, future_implications",
            "max_summary_length": "2000"
        })
    ]
    
    # Tool calls for each step
    browsing_tools = [
        # Step 1: Navigation
        ToolCall(data={
            "name": "web_navigate",
            "args": {
                "url": "https://techmeme.com",
                "wait_for": "dom_content_loaded",
                "timeout": 30,
                "headers": {"User-Agent": "AI News Assistant Bot v1.0"}
            },
            "result": {
                "status_code": 200,
                "page_title": "Techmeme",
                "load_time": 1.2,
                "page_size": "2.1MB",
                "cookies_set": 3,
                "javascript_errors": 0
            },
            "execution_time": 1.8,
            "cached": False
        }),
        
        # Step 2: Content extraction
        ToolCall(data={
            "name": "extract_page_content",
            "args": {
                "selectors": {
                    "headlines": ".storylink",
                    "timestamps": ".time", 
                    "sources": ".source",
                    "discussion_links": ".discussion"
                },
                "extract_attributes": ["href", "text", "data-timestamp"],
                "output_format": "structured_json"
            },
            "result": {
                "articles_found": 34,
                "extraction_success_rate": 0.97,
                "data_quality_score": 0.91,
                "headlines": [
                    {
                        "title": "OpenAI announces GPT-5 with enhanced reasoning capabilities",
                        "url": "https://example-source.com/gpt5-announcement",
                        "timestamp": "2025-06-10T20:30:00Z",
                        "source": "TechCrunch",
                        "discussion_count": 47
                    },
                    {
                        "title": "Anthropic releases Claude 4 with improved safety features",
                        "url": "https://anthropic.com/claude-4-release",
                        "timestamp": "2025-06-10T19:15:00Z", 
                        "source": "Anthropic Blog",
                        "discussion_count": 23
                    },
                    {
                        "title": "Google's Gemini Ultra shows breakthrough in multimodal AI",
                        "url": "https://blog.google/gemini-ultra-update",
                        "timestamp": "2025-06-10T18:45:00Z",
                        "source": "Google AI Blog",
                        "discussion_count": 31
                    }
                ]
            },
            "execution_time": 2.4,
            "cached": False
        }),
        
        # Step 3: AI content filtering
        ToolCall(data={
            "name": "filter_ai_content",
            "args": {
                "articles": "extracted_articles_from_step2",
                "filter_keywords": ["AI", "artificial intelligence", "machine learning", "LLM"],
                "relevance_threshold": 0.7,
                "use_semantic_matching": True
            },
            "result": {
                "total_articles_processed": 34,
                "ai_relevant_articles": 12,
                "filter_accuracy": 0.89,
                "top_matching_articles": [
                    {"title": "OpenAI announces GPT-5", "relevance_score": 0.95},
                    {"title": "Anthropic releases Claude 4", "relevance_score": 0.93},
                    {"title": "Google's Gemini Ultra breakthrough", "relevance_score": 0.91}
                ],
                "filtered_keywords_found": ["AI", "GPT", "Claude", "Gemini", "neural", "LLM"]
            },
            "execution_time": 1.1,
            "cached": False
        }),
        
        # Step 4: Deep content extraction
        ToolCall(data={
            "name": "extract_article_details",
            "args": {
                "article_urls": ["url1", "url2", "url3"],
                "extract_elements": ["full_text", "author", "publication_date", "images"],
                "follow_redirects": True,
                "max_content_length": 10000
            },
            "result": {
                "articles_processed": 10,
                "successful_extractions": 9,
                "average_content_length": 1247,
                "total_images_found": 15,
                "article_details": [
                    {
                        "url": "https://example-source.com/gpt5-announcement",
                        "author": "Sarah Chen",
                        "publication_date": "2025-06-10T20:30:00Z",
                        "word_count": 1456,
                        "key_points": [
                            "GPT-5 shows 40% improvement in reasoning tasks",
                            "New safety alignment techniques implemented",
                            "Beta testing with select partners begins next month"
                        ],
                        "images": 3,
                        "content_quality": "high"
                    }
                ]
            },
            "execution_time": 8.7,
            "cached": False
        }),
        
        # Step 5: Discussion analysis
        ToolCall(data={
            "name": "analyze_discussions",
            "args": {
                "discussion_urls": ["url1/discuss", "url2/discuss"],
                "max_comments": 50,
                "sentiment_analysis": True,
                "trend_detection": True
            },
            "result": {
                "total_comments_analyzed": 234,
                "sentiment_distribution": {"positive": 0.62, "neutral": 0.24, "negative": 0.14},
                "trending_topics": [
                    {"topic": "AI safety concerns", "mentions": 47, "sentiment": "mixed"},
                    {"topic": "competitive landscape", "mentions": 31, "sentiment": "positive"},
                    {"topic": "technical capabilities", "mentions": 89, "sentiment": "positive"}
                ],
                "key_discussion_themes": [
                    "Performance improvements over previous models",
                    "Implications for AI industry competition",
                    "Safety and alignment considerations"
                ]
            },
            "execution_time": 4.2,
            "cached": False
        }),
        
        # Step 6: Content categorization
        ToolCall(data={
            "name": "categorize_articles",
            "args": {
                "articles": "filtered_ai_articles",
                "categories": ["product_launches", "research_breakthroughs", "industry_news"],
                "ranking_factors": ["recency", "discussion_activity", "source_authority"]
            },
            "result": {
                "categorization_results": {
                    "product_launches": 5,
                    "research_breakthroughs": 4,
                    "industry_news": 3
                },
                "top_ranked_articles": [
                    {
                        "title": "OpenAI announces GPT-5",
                        "category": "product_launches",
                        "rank_score": 0.94,
                        "factors": {"recency": 0.98, "discussion": 0.91, "authority": 0.92}
                    },
                    {
                        "title": "Anthropic releases Claude 4", 
                        "category": "product_launches",
                        "rank_score": 0.89,
                        "factors": {"recency": 0.95, "discussion": 0.84, "authority": 0.88}
                    }
                ]
            },
            "execution_time": 1.8,
            "cached": False
        }),
        
        # Step 7: Summary generation
        ToolCall(data={
            "name": "generate_news_summary",
            "args": {
                "articles": "top_ranked_articles",
                "summary_sections": ["top_stories", "trending_topics", "key_quotes"],
                "max_length": 2000,
                "tone": "professional"
            },
            "result": {
                "summary_generated": True,
                "word_count": 1847,
                "sections_included": 4,
                "key_statistics": {
                    "total_articles_summarized": 12,
                    "companies_mentioned": 8,
                    "technologies_covered": 6
                },
                "summary_text": "Today's AI news highlights significant developments across major tech companies. OpenAI announced GPT-5 with enhanced reasoning capabilities, showing 40% improvement in complex problem-solving tasks. Anthropic followed with Claude 4, emphasizing new safety features and alignment techniques. Google's Gemini Ultra demonstrated breakthrough multimodal capabilities...",
                "trending_themes": [
                    "Enhanced reasoning in large language models",
                    "Increased focus on AI safety and alignment", 
                    "Competitive dynamics between major AI labs"
                ]
            },
            "execution_time": 3.1,
            "cached": False
        })
    ]
    
    # Task runs for each tool execution
    task_runs = [
        TaskRun(data={
            "success": True,
            "step_name": "navigation",
            "execution_time": 1.8,
            "memory_used": "45MB",
            "network_requests": 1,
            "errors": [],
            "performance_metrics": {"dom_load_time": 1.2, "first_paint": 0.8}
        }),
        
        TaskRun(data={
            "success": True,
            "step_name": "content_extraction", 
            "execution_time": 2.4,
            "memory_used": "78MB",
            "elements_extracted": 34,
            "data_quality": 0.97,
            "extraction_errors": 1
        }),
        
        TaskRun(data={
            "success": True,
            "step_name": "ai_filtering",
            "execution_time": 1.1,
            "memory_used": "23MB", 
            "articles_filtered": 12,
            "filter_precision": 0.89,
            "false_positives": 2
        }),
        
        TaskRun(data={
            "success": True,
            "step_name": "deep_extraction",
            "execution_time": 8.7,
            "memory_used": "156MB",
            "articles_processed": 10,
            "success_rate": 0.9,
            "network_requests": 15,
            "content_bytes_processed": 2456789
        }),
        
        TaskRun(data={
            "success": True, 
            "step_name": "discussion_analysis",
            "execution_time": 4.2,
            "memory_used": "89MB",
            "comments_processed": 234,
            "sentiment_accuracy": 0.87,
            "trends_identified": 8
        }),
        
        TaskRun(data={
            "success": True,
            "step_name": "categorization",
            "execution_time": 1.8,
            "memory_used": "34MB",
            "articles_categorized": 12,
            "categorization_confidence": 0.91,
            "ranking_computed": True
        }),
        
        TaskRun(data={
            "success": True,
            "step_name": "summary_generation",
            "execution_time": 3.1,
            "memory_used": "67MB",
            "summary_length": 1847,
            "readability_score": 0.84,
            "fact_check_passed": True
        })
    ]
    
    # Step summaries
    step_summaries = [
        Summary(data={
            "content": "Successfully navigated to Techmeme homepage. Page loaded in 1.2s with no JavaScript errors.",
            "load_time": "1.2",
            "status_code": "200"
        }),
        
        Summary(data={
            "content": "Extracted 34 article headlines with 97% success rate. Found strong variety of tech news topics.",
            "articles_found": "34",
            "success_rate": "0.97"
        }),
        
        Summary(data={
            "content": "Filtered to 12 AI-relevant articles using keyword and semantic matching. High precision achieved.",
            "ai_articles": "12",
            "precision": "0.89"
        }),
        
        Summary(data={
            "content": "Deep extraction completed on 10 articles. Gathered full content, metadata, and key insights.",
            "articles_processed": "10",
            "avg_length": "1247"
        }),
        
        Summary(data={
            "content": "Analyzed 234 discussion comments. Positive sentiment dominates with trending focus on AI safety.",
            "comments": "234",
            "positive_sentiment": "0.62"
        }),
        
        Summary(data={
            "content": "Categorized articles into product launches (5), research (4), and industry news (3). Clear ranking established.",
            "categories": "3",
            "confidence": "0.91"
        }),
        
        Summary(data={
            "content": "Generated comprehensive 1,847-word summary covering top stories and trending themes in AI.",
            "word_count": "1847",
            "readability": "0.84"
        })
    ]
    
    # Add all nodes to graph
    all_nodes = ([session, user_request, assistant_response, browsing_plan] + 
                browsing_steps + browsing_tools + task_runs + step_summaries)
    
    for node in all_nodes:
        graph.add_node(node)
    
    # Create comprehensive edge structure
    edges = [
        # Session structure
        ParentChildEdge(src=session.id, dst=user_request.id),
        ParentChildEdge(src=session.id, dst=assistant_response.id),
        ParentChildEdge(src=session.id, dst=browsing_plan.id),
        NextEdge(src=user_request.id, dst=assistant_response.id),
        NextEdge(src=assistant_response.id, dst=browsing_plan.id),
        
        # Plan structure
        *[ParentChildEdge(src=browsing_plan.id, dst=step.id) for step in browsing_steps],
        
        # Step dependencies (sequential workflow)
        *[StepEdge(src=browsing_steps[i].id, dst=browsing_steps[i+1].id) 
          for i in range(len(browsing_steps)-1)],
        
        # Tool execution chains
        *[PlanEdge(src=browsing_steps[i].id, dst=browsing_tools[i].id) 
          for i in range(len(browsing_steps))],
        *[ParentChildEdge(src=browsing_tools[i].id, dst=task_runs[i].id) 
          for i in range(len(browsing_tools))],
        *[ParentChildEdge(src=task_runs[i].id, dst=step_summaries[i].id) 
          for i in range(len(step_summaries))],
        
        # Data flow edges (custom)
        GraphEdge(kind=EdgeKind.CUSTOM, src=browsing_tools[1].id, dst=browsing_tools[2].id,
                 data={"data_type": "extracted_articles", "flow_direction": "output_to_input"}),
        GraphEdge(kind=EdgeKind.CUSTOM, src=browsing_tools[2].id, dst=browsing_tools[3].id,
                 data={"data_type": "filtered_articles", "flow_direction": "output_to_input"}),
        GraphEdge(kind=EdgeKind.CUSTOM, src=browsing_tools[5].id, dst=browsing_tools[6].id,
                 data={"data_type": "ranked_articles", "flow_direction": "output_to_input"}),
    ]
    
    for edge in edges:
        graph.add_edge(edge)
    
    return graph, session, browsing_plan, browsing_steps, browsing_tools, task_runs

def analyze_browsing_workflow(graph, session, browsing_plan, steps, tools, task_runs):
    """Analyze the web browsing workflow performance."""
    print("\n🔍 Browsing Workflow Analysis")
    print("=" * 60)
    
    # Overall statistics
    total_execution_time = sum(tr.data.get('execution_time', 0) for tr in task_runs)
    total_memory_used = sum(
        float(tr.data.get('memory_used', '0MB').replace('MB', '')) 
        for tr in task_runs
    )
    successful_steps = sum(1 for tr in task_runs if tr.data.get('success', False))
    
    print(f"📊 Workflow Statistics:")
    print(f"   Total execution time: {total_execution_time:.1f} seconds")
    print(f"   Peak memory usage: {total_memory_used:.0f} MB")
    print(f"   Successful steps: {successful_steps}/{len(task_runs)}")
    print(f"   Success rate: {successful_steps/len(task_runs)*100:.1f}%")
    
    # Content analysis
    extraction_tool = tools[1]  # Content extraction tool
    filtering_tool = tools[2]   # AI filtering tool
    summary_tool = tools[6]     # Summary generation tool
    
    articles_found = extraction_tool.data['result']['articles_found']
    ai_articles = filtering_tool.data['result']['ai_relevant_articles'] 
    summary_length = summary_tool.data['result']['word_count']
    
    print(f"\n📰 Content Analysis:")
    print(f"   Total articles found: {articles_found}")
    print(f"   AI-relevant articles: {ai_articles}")
    print(f"   Content reduction: {(1 - ai_articles/articles_found)*100:.1f}%")
    print(f"   Final summary length: {summary_length} words")
    
    # Performance bottlenecks
    slowest_step = max(task_runs, key=lambda tr: tr.data.get('execution_time', 0))
    fastest_step = min(task_runs, key=lambda tr: tr.data.get('execution_time', 0))
    
    print(f"\n⚡ Performance Analysis:")
    print(f"   Slowest step: {slowest_step.data.get('step_name')} ({slowest_step.data.get('execution_time', 0):.1f}s)")
    print(f"   Fastest step: {fastest_step.data.get('step_name')} ({fastest_step.data.get('execution_time', 0):.1f}s)")
    
    # Data quality metrics
    extraction_quality = extraction_tool.data['result']['data_quality_score']
    filter_precision = filtering_tool.data['result']['filter_accuracy']
    categorization_confidence = tools[5].data['result']['categorization_results']
    
    print(f"\n🎯 Data Quality Metrics:")
    print(f"   Extraction quality: {extraction_quality:.1%}")
    print(f"   Filter precision: {filter_precision:.1%}")
    print(f"   Categorization confidence: {tools[5].data['result']['top_ranked_articles'][0]['rank_score']:.1%}")

def demo_error_handling_scenario():
    """Demonstrate error handling in web browsing workflow."""
    print("\n🚨 Error Handling Scenario")
    print("=" * 60)
    
    graph = InMemoryGraphStore()
    
    # Create a scenario with some failures
    session = SessionNode(data={
        "session_type": "web_browsing_with_errors",
        "target_site": "techmeme.com",
        "retry_policy": "exponential_backoff"
    })
    
    # Failed navigation attempt
    failed_navigation = ToolCall(data={
        "name": "web_navigate",
        "args": {"url": "https://techmeme.com", "timeout": 30},
        "result": None,
        "error": "Connection timeout after 30 seconds",
        "retry_count": 3,
        "cached": False
    })
    
    failed_task = TaskRun(data={
        "success": False,
        "step_name": "navigation",
        "execution_time": 30.0,
        "error_type": "NetworkTimeoutError",
        "error_message": "Connection timeout after 30 seconds",
        "retry_attempts": 3,
        "recovery_action": "switch_to_backup_proxy"
    })
    
    # Successful retry with backup proxy
    retry_navigation = ToolCall(data={
        "name": "web_navigate",
        "args": {
            "url": "https://techmeme.com",
            "timeout": 30,
            "proxy": "backup_proxy_server",
            "retry_attempt": 4
        },
        "result": {
            "status_code": 200,
            "page_title": "Techmeme",
            "load_time": 2.1,
            "via_proxy": True
        },
        "cached": False
    })
    
    retry_task = TaskRun(data={
        "success": True,
        "step_name": "navigation_retry",
        "execution_time": 2.5,
        "recovery_method": "backup_proxy",
        "performance_impact": "minimal"
    })
    
    # Add nodes and edges
    nodes = [session, failed_navigation, failed_task, retry_navigation, retry_task]
    for node in nodes:
        graph.add_node(node)
    
    edges = [
        ParentChildEdge(src=session.id, dst=failed_navigation.id),
        ParentChildEdge(src=failed_navigation.id, dst=failed_task.id),
        NextEdge(src=failed_task.id, dst=retry_navigation.id),
        ParentChildEdge(src=retry_navigation.id, dst=retry_task.id),
    ]
    
    for edge in edges:
        graph.add_edge(edge)
    
    print("🔄 Error Recovery Workflow:")
    print(f"   Initial attempt: FAILED (timeout)")
    print(f"   Retry attempts: 3")
    print(f"   Recovery strategy: backup proxy")
    print(f"   Final result: SUCCESS")
    print(f"   Total time: {failed_task.data['execution_time'] + retry_task.data['execution_time']:.1f}s")
    
    return graph

def demo_real_time_monitoring():
    """Demonstrate real-time monitoring capabilities."""
    print("\n📡 Real-time Monitoring Dashboard")
    print("=" * 60)
    
    # Simulate real-time status updates
    current_time = datetime.now(timezone.utc)
    
    monitoring_data = {
        "session_status": "active",
        "current_step": "Step 4: Deep content extraction",
        "progress": "57%",
        "elapsed_time": "42.3 seconds",
        "estimated_remaining": "31.7 seconds",
        "active_threads": 3,
        "memory_usage": "156 MB",
        "network_status": "stable",
        "error_count": 1,
        "success_rate": "94.2%"
    }
    
    print("🎯 Current Status:")
    for key, value in monitoring_data.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # Performance timeline
    timeline = [
        {"step": "Navigation", "duration": 1.8, "status": "✅"},
        {"step": "Content Extraction", "duration": 2.4, "status": "✅"}, 
        {"step": "AI Filtering", "duration": 1.1, "status": "✅"},
        {"step": "Deep Extraction", "duration": 8.7, "status": "🔄"},
        {"step": "Discussion Analysis", "duration": 0, "status": "⏳"},
        {"step": "Categorization", "duration": 0, "status": "⏳"},
        {"step": "Summary Generation", "duration": 0, "status": "⏳"}
    ]
    
    print(f"\n⏱️ Execution Timeline:")
    for item in timeline:
        status_icon = item["status"]
        duration_str = f"{item['duration']:.1f}s" if item["duration"] > 0 else "pending"
        print(f"   {status_icon} {item['step']:<20} {duration_str}")

def main():
    """Run the comprehensive Techmeme browsing demo."""
    print("🌐 Techmeme AI News Browsing - Graph Models Demo")
    print("=" * 80)
    
    try:
        # Create the main browsing workflow
        graph, session, plan, steps, tools, task_runs = create_web_browsing_session()
        
        # Show graph structure
        print("\n🏗️ Graph Structure Visualization")
        print("=" * 60)
        print_graph_structure(graph)
        
        # Analyze the workflow
        analyze_browsing_workflow(graph, session, plan, steps, tools, task_runs)
        
        # Demonstrate error handling
        demo_error_handling_scenario()
        
        # Show real-time monitoring
        demo_real_time_monitoring()
        
        # Final summary
        print("\n" + "=" * 80)
        print("✅ Techmeme Browsing Demo Complete!")
        print("\n🎯 Key Demonstrations:")
        print("   • Complex multi-step web automation workflow")
        print("   • Rich graph structure with 25+ nodes and relationships")
        print("   • Real-world tool calls with realistic results")
        print("   • Error handling and recovery mechanisms")
        print("   • Performance monitoring and analytics")
        print("   • Content extraction, filtering, and analysis pipeline")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()