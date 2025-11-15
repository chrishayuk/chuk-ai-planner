#!/usr/bin/env python3
"""
LLM Extension Example
=====================

This example demonstrates using LLM-specific nodes:
- UserMessage, AssistantMessage, SystemMessage
- These are EXTENSIONS of the core domain-agnostic graph
- Other projects (chuk-motion, etc.) can create their own extensions

Key Takeaway: Core graph is domain-agnostic, extensions add domain-specific nodes.
"""

# Core graph imports (domain-agnostic)
from chuk_ai_planner.graph import (
    SessionNode,
    PlanNode,
    PlanStep,
    ToolCall,
    ParentChildEdge,
    NextEdge,
)

# LLM-specific extension imports
from chuk_ai_planner.graph.nodes.llm import (
    UserMessage,
    AssistantMessage,
    SystemMessage,
)

from chuk_ai_planner.store.memory import InMemoryGraphStore


def main():
    print("=" * 70)
    print("LLM Extension Example")
    print("=" * 70)
    print("\nDemonstrating domain-specific extensions of the core graph")
    print("=" * 70)

    graph = InMemoryGraphStore()

    # 1. Create a Session (core node)
    session = SessionNode(
        name="Weather Query Conversation",
        description="User asking about weather"
    )
    graph.add_node(session)
    print(f"\n✅ Created Session: {session.name}")

    # 2. System Message (LLM extension!)
    system_msg = SystemMessage(
        content="You are a helpful weather assistant.",
        role="system"
    )
    graph.add_node(system_msg)
    graph.add_edge(ParentChildEdge(src=session.id, dst=system_msg.id))
    print(f"\n🤖 System Message: {system_msg.content}")
    print(f"   Type: {system_msg.kind}")  # "system_message"
    print("   This is an LLM extension node!")

    # 3. User Message (LLM extension!)
    user_msg = UserMessage(
        content="What's the weather in New York?",
        role="user",
        user_id="user123"
    )
    graph.add_node(user_msg)
    graph.add_edge(ParentChildEdge(src=session.id, dst=user_msg.id))
    graph.add_edge(NextEdge(src=system_msg.id, dst=user_msg.id))
    print(f"\n👤 User Message: {user_msg.content}")
    print(f"   User ID: {user_msg.user_id}")
    print("   This is an LLM extension node!")

    # 4. Create a Plan (core node) in response
    plan = PlanNode(
        title="Weather Research Plan",
        description="Fetch and summarize NYC weather"
    )
    graph.add_node(plan)
    graph.add_edge(ParentChildEdge(src=session.id, dst=plan.id))
    print(f"\n📋 Created Plan: {plan.title}")
    print("   This is a core graph node!")

    # 5. Plan Steps (core nodes)
    step1 = PlanStep(description="Fetch weather data", index="1")
    step2 = PlanStep(description="Summarize findings", index="2")

    graph.add_node(step1)
    graph.add_node(step2)
    graph.add_edge(ParentChildEdge(src=plan.id, dst=step1.id))
    graph.add_edge(ParentChildEdge(src=plan.id, dst=step2.id))

    # 6. Tool Call for step 1 (core node)
    weather_tool = ToolCall(
        name="get_weather",
        args={"city": "New York", "units": "fahrenheit"}
    )
    graph.add_node(weather_tool)

    # 7. Assistant Message with tool call (LLM extension!)
    assistant_msg = AssistantMessage(
        content="I'll check the weather for you.",
        role="assistant",
        tool_calls=[{
            "id": weather_tool.id,
            "name": "get_weather",
            "arguments": {"city": "New York"}
        }],
        model="gpt-4"
    )
    graph.add_node(assistant_msg)
    graph.add_edge(ParentChildEdge(src=session.id, dst=assistant_msg.id))
    graph.add_edge(NextEdge(src=user_msg.id, dst=assistant_msg.id))

    print(f"\n🤖 Assistant Message: {assistant_msg.content}")
    print(f"   Model: {assistant_msg.model}")
    print(f"   Tool Calls: {len(assistant_msg.tool_calls)}")
    print("   This is an LLM extension node!")

    # 8. Query the Graph
    print("\n" + "=" * 70)
    print("Graph Structure")
    print("=" * 70)

    # Core nodes
    print("\n📊 Core Graph Nodes:")
    print(f"   Sessions: {len(graph.get_nodes_by_kind('session'))}")
    print(f"   Plans: {len(graph.get_nodes_by_kind('plan'))}")
    print(f"   Steps: {len(graph.get_nodes_by_kind('plan_step'))}")
    print(f"   Tools: {len(graph.get_nodes_by_kind('tool_call'))}")

    # LLM extension nodes
    print("\n📊 LLM Extension Nodes:")
    print(f"   System Messages: {len([n for n in graph.nodes.values() if n.kind == 'system_message'])}")
    print(f"   User Messages: {len([n for n in graph.nodes.values() if n.kind == 'user_message'])}")
    print(f"   Assistant Messages: {len([n for n in graph.nodes.values() if n.kind == 'assistant_message'])}")

    # 9. Demonstrate Extension Pattern
    print("\n" + "=" * 70)
    print("Extension Pattern")
    print("=" * 70)

    print("\n✨ Core Graph (domain-agnostic):")
    print("   from chuk_ai_planner.graph import PlanNode, PlanStep")
    print("   → Works for ANY planning domain")

    print("\n✨ LLM Extension:")
    print("   from chuk_ai_planner.graph.nodes.llm import UserMessage")
    print("   → Adds chat/LLM-specific nodes")

    print("\n✨ Future Extensions:")
    print("   from chuk_motion.graph.nodes import VideoNode, AudioNode")
    print("   from chuk_video.graph.nodes import SceneNode, TransitionNode")
    print("   → Projects can extend with their own domain nodes!")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n✅ Core graph is domain-agnostic (PlanNode, PlanStep, etc.)")
    print("✅ LLM extension adds UserMessage, AssistantMessage, SystemMessage")
    print("✅ Extensions keep the core clean and reusable")
    print("✅ Other projects can create their own extensions")
    print("\n")


if __name__ == "__main__":
    main()
