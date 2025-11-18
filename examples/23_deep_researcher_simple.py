#!/usr/bin/env python
"""
examples/universal_deep_researcher_simple.py
A simplified version of the deep-research tool using UniversalPlan.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
#  Imports from the planner package                                           #
# --------------------------------------------------------------------------- #

from chuk_ai_planner.core.planner.universal_plan import UniversalPlan
from chuk_ai_planner.core.planner.universal_plan_executor import UniversalExecutor

load_dotenv()

# --------------------------------------------------------------------------- #
#  Logging setup                                                              #
# --------------------------------------------------------------------------- #

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Dataclasses                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class ResearchConfig:
    max_results: int = 5
    max_rounds: int = 1


# --------------------------------------------------------------------------- #
#  In-memory store for docs gathered during a run                             #
# --------------------------------------------------------------------------- #

research_documents: List[Dict[str, Any]] = []


# --------------------------------------------------------------------------- #
#  Tool: search                                                               #
# --------------------------------------------------------------------------- #


async def search_tool(query: str = "unknown") -> dict:
    """
    Mock search tool—replace with real search API if desired.
    """
    logger.info(f"🔍 Searching for: {query}")

    results = [
        {
            "title": f"Result 1 for {query}",
            "snippet": f"This is information about {query}…",
            "url": f"https://example.com/1?q={query}",
        },
        {
            "title": f"Result 2 for {query}",
            "snippet": f"More information about {query}…",
            "url": f"https://example.com/2?q={query}",
        },
        {
            "title": f"Result 3 for {query}",
            "snippet": f"Additional details about {query}…",
            "url": f"https://example.com/3?q={query}",
        },
    ]

    # Update the global list in place (no reassignment)
    research_documents.clear()
    research_documents.extend(results)
    logger.info(f"Updated global research_documents with {len(results)} items")

    return {"results": results}


# --------------------------------------------------------------------------- #
#  Tool: summarize                                                            #
# --------------------------------------------------------------------------- #


def summarize_tool(*_: Any, **__: Any) -> Dict[str, Any]:
    """
    Create a summary of `research_documents`.
    Ignores incoming args; relies solely on the global list.
    """
    logger.info("📝 Creating summary")

    if not research_documents:
        return {"summary": "No information was found.", "source_count": 0}

    snippets = [doc["snippet"] for doc in research_documents if "snippet" in doc]
    summary = (
        f"Based on {len(research_documents)} sources, here's what we found: "
        + " ".join(snippets[:3])
    )

    return {"summary": summary, "source_count": len(research_documents)}


# --------------------------------------------------------------------------- #
#  Main research routine                                                      #
# --------------------------------------------------------------------------- #


async def research_topic(topic: str, config: ResearchConfig) -> Dict[str, Any]:
    """
    Conduct research on a topic using UniversalPlan/UniversalExecutor.
    """
    logger.info(f"\n🔍 RESEARCHING: {topic}\n")

    # Clear docs from any previous run
    research_documents.clear()

    # Register tools with the executor
    executor = UniversalExecutor()
    await executor.register_tool("search", search_tool)
    await executor.register_function("summarize", summarize_tool)

    # Build the plan
    plan = UniversalPlan(
        title=f"Research on {topic}",
        description=f"Simple research plan for {topic}",
        tags=["research"],
        graph=executor.graph_store,
    )

    # Step 1: search
    s1 = await plan.add_tool_step(
        title=f"Search for {topic}",
        tool="search",
        args={"query": topic},
        result_variable="search_results",
    )

    # Step 2: summarize
    await plan.add_function_step(
        title=f"Summarize findings about {topic}",
        function="summarize",
        args={},  # summary uses global docs
        depends_on=[s1],
        result_variable="summary",
    )

    # Persist the plan (optional)
    await plan.save()
    logger.info(f"Plan created:\n{plan.outline()}")

    # Execute the plan
    try:
        result = await executor.execute_plan(plan)

        if not result.get("success", False):
            err = result.get("error", "Unknown error")
            logger.error(f"Error executing plan: {err}")
            return {"error": err}

        # Extract variables from the run
        search_results = (
            result["variables"].get("search_results", {}).get("results", [])
        )
        summary_text = (
            result["variables"]
            .get("summary", {})
            .get("summary", "No summary generated.")
        )

        # Ensure the global list holds the final results
        research_documents.clear()
        research_documents.extend(search_results)

        logger.info("\n📊 RESEARCH COMPLETE")
        logger.info(f"• Documents found: {len(search_results)}")

        logger.info("\n📝 SUMMARY")
        logger.info(summary_text)

        return {
            "topic": topic,
            "summary": summary_text,
            "documents": search_results,
        }

    except Exception as exc:  # pragma: no cover
        logger.error(f"Error in research: {exc}")
        import traceback

        traceback.print_exc()
        return {"error": str(exc)}


# --------------------------------------------------------------------------- #
#  CLI wrapper
# --------------------------------------------------------------------------- #


async def main() -> None:
    """Run the simple deep research demo."""
    topic = "artificial intelligence"
    config = ResearchConfig(max_results=5, max_rounds=1)
    await research_topic(topic, config)


if __name__ == "__main__":
    asyncio.run(main())
