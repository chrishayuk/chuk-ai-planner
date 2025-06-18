# examples/graph_model_example.py
from uuid import uuid4
from datetime import datetime, timezone
from chuk_ai_planner.models import (
    SessionNode, 
    UserMessage, 
    AssistantMessage,
    PlanNode, 
    PlanStep,
    ToolCall, 
    TaskRun, 
    Summary
)
from chuk_ai_planner.models.edges import (
    ParentChildEdge,
    NextEdge,
    PlanEdge,
    StepEdge
)

# Create a graph representing a conversation with planning and tool execution

# 1. Create a session
session = SessionNode(
    data={"title": "Weather Research Session", "user_id": "user123"}
)

# 2. User asks a question
user_msg = UserMessage(
    data={"content": "Can you find the weather forecast for New York and summarize it?"}
)

# 3. Create parent-child relationship between session and message
session_to_msg = ParentChildEdge(
    src=session.id,
    dst=user_msg.id
)

# 4. Assistant creates a plan with steps
assistant_plan = PlanNode(
    data={"description": "Find and summarize weather forecast for New York"}
)

plan_to_session = ParentChildEdge(
    src=session.id,
    dst=assistant_plan.id
)

# 5. Create plan steps
step1 = PlanStep(
    data={"description": "Search for current weather in New York", "index": 1}
)

step2 = PlanStep(
    data={"description": "Search for weather forecast for next 3 days", "index": 2}
)

step3 = PlanStep(
    data={"description": "Summarize findings", "index": 3}
)

# 6. Link steps to plan
plan_to_step1 = ParentChildEdge(src=assistant_plan.id, dst=step1.id)
plan_to_step2 = ParentChildEdge(src=assistant_plan.id, dst=step2.id)
plan_to_step3 = ParentChildEdge(src=assistant_plan.id, dst=step3.id)

# 7. Create step ordering edges
step1_to_step2 = StepEdge(src=step1.id, dst=step2.id)
step2_to_step3 = StepEdge(src=step2.id, dst=step3.id)

# 8. Assistant responds to user
assistant_msg = AssistantMessage(
    data={
        "content": "I'll help you find the weather forecast for New York and provide a summary. Let me break this down into steps.",
        "plan_id": assistant_plan.id
    }
)

# 9. Link messages
user_to_assistant = NextEdge(src=user_msg.id, dst=assistant_msg.id)
session_to_assistant = ParentChildEdge(src=session.id, dst=assistant_msg.id)

# 10. Execute tool calls for each step
tool_call1 = ToolCall(
    data={
        "name": "search_weather",
        "args": {"location": "New York", "type": "current"},
        "result": {
            "temperature": 72,
            "conditions": "Partly Cloudy",
            "humidity": 45,
            "wind": "10mph NE"
        }
    }
)

tool_call2 = ToolCall(
    data={
        "name": "search_weather",
        "args": {"location": "New York", "type": "forecast", "days": 3},
        "result": {
            "forecast": [
                {"day": "Monday", "high": 75, "low": 60, "conditions": "Sunny"},
                {"day": "Tuesday", "high": 78, "low": 62, "conditions": "Partly Cloudy"},
                {"day": "Wednesday", "high": 68, "low": 55, "conditions": "Rain"}
            ]
        }
    }
)

# 11. Create task runs that executed the tool calls
task_run1 = TaskRun(
    data={
        "executor": "weather_service",
        "cost": 0.005,
        "latency_ms": 250
    }
)

task_run2 = TaskRun(
    data={
        "executor": "weather_service",
        "cost": 0.008,
        "latency_ms": 320
    }
)

# 12. Link tool calls and task runs to the assistant message and plan steps
assistant_to_tool1 = ParentChildEdge(src=assistant_msg.id, dst=tool_call1.id)
assistant_to_tool2 = ParentChildEdge(src=assistant_msg.id, dst=tool_call2.id)

tool1_to_task1 = ParentChildEdge(src=tool_call1.id, dst=task_run1.id)
tool2_to_task2 = ParentChildEdge(src=tool_call2.id, dst=task_run2.id)

# 13. Link plan steps to tool calls
plan_to_tool1 = PlanEdge(src=step1.id, dst=tool_call1.id)
plan_to_tool2 = PlanEdge(src=step2.id, dst=tool_call2.id)

# 14. Create a summary node (for step 3)
summary = Summary(
    data={
        "content": """
        Weather Summary for New York:
        
        Currently: 72°F, Partly Cloudy with 45% humidity and 10mph winds from NE.
        
        3-Day Forecast:
        - Monday: High 75°F, Low 60°F, Sunny
        - Tuesday: High 78°F, Low 62°F, Partly Cloudy
        - Wednesday: High 68°F, Low 55°F, Rain
        
        The weather starts pleasant but brings rain by midweek. Tuesday will be the warmest day.
        """
    }
)

# 15. Link summary to plan step and task
plan_to_summary = PlanEdge(src=step3.id, dst=summary.id)
assistant_to_summary = ParentChildEdge(src=assistant_msg.id, dst=summary.id)

# 16. Final assistant message with the summary
final_msg = AssistantMessage(
    data={
        "content": """
        Here's the weather forecast for New York:
        
        Currently: 72°F, Partly Cloudy with 45% humidity and 10mph winds from NE.
        
        3-Day Forecast:
        - Monday: High 75°F, Low 60°F, Sunny
        - Tuesday: High 78°F, Low 62°F, Partly Cloudy
        - Wednesday: High 68°F, Low 55°F, Rain
        
        The weather starts pleasant but brings rain by midweek. Tuesday will be the warmest day.
        """,
        "summary_id": summary.id
    }
)

# 17. Link final message
assistant_to_final = NextEdge(src=assistant_msg.id, dst=final_msg.id)
session_to_final = ParentChildEdge(src=session.id, dst=final_msg.id)

# 18. Collect all nodes and edges
nodes = [
    session, 
    user_msg, 
    assistant_plan, 
    step1, step2, step3,
    assistant_msg, 
    tool_call1, tool_call2,
    task_run1, task_run2,
    summary,
    final_msg
]

edges = [
    session_to_msg,
    plan_to_session,
    plan_to_step1, plan_to_step2, plan_to_step3,
    step1_to_step2, step2_to_step3,
    user_to_assistant, session_to_assistant,
    assistant_to_tool1, assistant_to_tool2,
    tool1_to_task1, tool2_to_task2,
    plan_to_tool1, plan_to_tool2,
    plan_to_summary, assistant_to_summary,
    assistant_to_final, session_to_final
]

# Demo how to traverse the graph:

def print_graph_structure():
    print(f"Session: {session!r}")
    print("├── Messages:")
    
    # Find all direct children of session that are messages
    message_edges = [e for e in edges if e.src == session.id and (
        e.dst == user_msg.id or 
        e.dst == assistant_msg.id or
        e.dst == final_msg.id
    )]
    
    for i, edge in enumerate(message_edges):
        is_last = i == len(message_edges) - 1
        prefix = "└──" if is_last else "├──"
        
        # Find the message node
        message = next((n for n in nodes if n.id == edge.dst), None)
        if message:
            if message.kind.value == "user_message":
                print(f"{prefix} User: {message!r}")
                print(f"    └── Content: {message.data.get('content', '')[:50]}...")
            else:
                print(f"{prefix} Assistant: {message!r}")
                print(f"    └── Content: {message.data.get('content', '')[:50]}...")
                
                # Find tool calls for this message
                tool_edges = [e for e in edges if e.src == message.id and e.kind.value == "parent_child"]
                for j, tool_edge in enumerate(tool_edges):
                    is_last_tool = j == len(tool_edges) - 1
                    tool_prefix = "    └──" if is_last_tool else "    ├──"
                    
                    tool = next((n for n in nodes if n.id == tool_edge.dst), None)
                    if tool and tool.kind.value == "tool_call":
                        print(f"{tool_prefix} Tool Call: {tool!r}")
                        print(f"        └── {tool.data.get('name', 'unnamed')}({tool.data.get('args', {})})") 
    
    # Print the plan structure
    print("\nPlan Structure:")
    print(f"Plan: {assistant_plan!r}")
    
    # Find all steps
    step_edges = [e for e in edges if e.src == assistant_plan.id and e.kind.value == "parent_child"]
    
    for i, edge in enumerate(step_edges):
        is_last = i == len(step_edges) - 1
        prefix = "└──" if is_last else "├──"
        
        step = next((n for n in nodes if n.id == edge.dst), None)
        if step and step.kind.value == "plan_step":
            print(f"{prefix} Step {step.data.get('index')}: {step.data.get('description')}")
            
            # Find executions linked to this step
            exec_edges = [e for e in edges if e.src == step.id and e.kind.value == "plan_link"]
            for j, exec_edge in enumerate(exec_edges):
                is_last_exec = j == len(exec_edges) - 1
                exec_prefix = "    └──" if is_last_exec else "    ├──"
                
                execution = next((n for n in nodes if n.id == exec_edge.dst), None)
                if execution:
                    print(f"{exec_prefix} Execution: {execution!r}")

if __name__ == "__main__":
    print_graph_structure()
    
    # You could also visualize this graph using libraries like networkx and matplotlib
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        G = nx.DiGraph()
        
        # Add nodes
        for node in nodes:
            G.add_node(node.id, kind=node.kind.value, label=f"{node.kind.value}:{node.id[:6]}")
        
        # Add edges
        for edge in edges:
            G.add_edge(edge.src, edge.dst, kind=edge.kind.value)
        
        # Draw the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        
        node_colors = {
            'session': 'skyblue',
            'user_message': 'lightgreen',
            'assistant_message': 'lightcoral',
            'plan': 'yellow',
            'plan_step': 'orange',
            'tool_call': 'purple',
            'task_run': 'pink',
            'summary': 'cyan'
        }
        
        colors = [node_colors[G.nodes[n]['kind']] for n in G.nodes]
        
        nx.draw(G, pos, with_labels=False, node_color=colors, node_size=500, arrows=True)
        
        # Add labels with minimal text
        labels = {n: G.nodes[n]['label'] for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        plt.title("Conversation Graph Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("conversation_graph.png")
        print("\nGraph visualization saved as 'conversation_graph.png'")
    except ImportError:
        print("\nNote: Install networkx and matplotlib to visualize the graph")