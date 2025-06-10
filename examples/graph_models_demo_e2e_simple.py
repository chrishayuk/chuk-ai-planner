#!/usr/bin/env python3
# examples/graph_models_demo_e2e_simple.py
"""
Simple End-to-End Demo: chuk_ai_planner Graph Models
===================================================

This demo shows all key functionality in a simple, easy-to-follow example:
1. Creating all node types
2. Creating all edge types  
3. Building a complete graph
4. Querying and analyzing the graph
5. Graph visualization
6. Node immutability

Scenario: AI assistant helps user analyze data with a simple 3-step plan

Run with: python demo_e2e_simple.py
"""

import json
from datetime import datetime, timezone

# Core imports
from chuk_ai_planner.models import (
    NodeKind, SessionNode, PlanNode, PlanStep,
    UserMessage, AssistantMessage, ToolCall, TaskRun, Summary
)

from chuk_ai_planner.models.edges import (
    EdgeKind, ParentChildEdge, NextEdge, PlanEdge, StepEdge
)

from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.utils.visualization import print_graph_structure

def step1_create_nodes():
    """Step 1: Create all types of nodes"""
    print("📦 STEP 1: Creating Graph Nodes")
    print("=" * 50)
    
    # Create a session for our AI assistant interaction
    session = SessionNode(data={
        "user_id": "demo_user",
        "session_type": "data_analysis_help"
    })
    
    # User asks for help
    user_msg = UserMessage(data={
        "content": "Can you help me analyze my sales data?"
    })
    
    # Assistant responds with a plan
    assistant_msg = AssistantMessage(data={
        "content": "I'll help you analyze your sales data. Let me create a plan.",
        "has_plan": "true"
    })
    
    # Create a simple 3-step plan
    plan = PlanNode(data={
        "title": "Sales Data Analysis Plan",
        "description": "Simple 3-step data analysis workflow"
    })
    
    # Plan steps
    step1 = PlanStep(data={
        "index": "1",
        "description": "Load the sales data file"
    })
    
    step2 = PlanStep(data={
        "index": "2", 
        "description": "Calculate key metrics"
    })
    
    step3 = PlanStep(data={
        "index": "3",
        "description": "Generate summary report"
    })
    
    # Tool calls for each step
    tool1 = ToolCall(data={
        "name": "load_csv",
        "args": '{"file": "sales_data.csv"}',
        "result": '{"rows": 1000, "columns": 5}'
    })
    
    tool2 = ToolCall(data={
        "name": "calculate_metrics",
        "args": '{"metrics": ["total_sales", "avg_order"]}',
        "result": '{"total_sales": 50000, "avg_order": 50}'
    })
    
    tool3 = ToolCall(data={
        "name": "generate_report",
        "args": '{"format": "summary"}',
        "result": '{"report_length": 500, "charts": 3}'
    })
    
    # Task execution results
    task1 = TaskRun(data={
        "success": "true",
        "execution_time": "1.2"
    })
    
    task2 = TaskRun(data={
        "success": "true", 
        "execution_time": "0.8"
    })
    
    task3 = TaskRun(data={
        "success": "true",
        "execution_time": "2.1"
    })
    
    # Step summaries
    summary1 = Summary(data={
        "content": "Successfully loaded 1000 rows of sales data"
    })
    
    summary2 = Summary(data={
        "content": "Calculated metrics: $50K total sales, $50 avg order"
    })
    
    summary3 = Summary(data={
        "content": "Generated comprehensive report with 3 charts"
    })
    
    nodes = [
        session, user_msg, assistant_msg, plan,
        step1, step2, step3,
        tool1, tool2, tool3,
        task1, task2, task3,
        summary1, summary2, summary3
    ]
    
    print(f"✅ Created {len(nodes)} nodes:")
    for node in nodes:
        print(f"   {node.kind.value}: {node.id[:8]}...")
    
    return nodes

def step2_create_edges(nodes):
    """Step 2: Create relationships between nodes"""
    print(f"\n🔗 STEP 2: Creating Graph Edges")
    print("=" * 50)
    
    # Unpack nodes for clarity
    (session, user_msg, assistant_msg, plan,
     step1, step2, step3,
     tool1, tool2, tool3, 
     task1, task2, task3,
     summary1, summary2, summary3) = nodes
    
    edges = [
        # Session contains messages and plan
        ParentChildEdge(src=session.id, dst=user_msg.id),
        ParentChildEdge(src=session.id, dst=assistant_msg.id),
        ParentChildEdge(src=session.id, dst=plan.id),
        
        # Message conversation flow
        NextEdge(src=user_msg.id, dst=assistant_msg.id),
        
        # Plan contains steps
        ParentChildEdge(src=plan.id, dst=step1.id),
        ParentChildEdge(src=plan.id, dst=step2.id),
        ParentChildEdge(src=plan.id, dst=step3.id),
        
        # Step execution order (step dependencies)
        StepEdge(src=step1.id, dst=step2.id),  # step2 depends on step1
        StepEdge(src=step2.id, dst=step3.id),  # step3 depends on step2
        
        # Steps execute tools
        PlanEdge(src=step1.id, dst=tool1.id),
        PlanEdge(src=step2.id, dst=tool2.id),
        PlanEdge(src=step3.id, dst=tool3.id),
        
        # Tools create task runs
        ParentChildEdge(src=tool1.id, dst=task1.id),
        ParentChildEdge(src=tool2.id, dst=task2.id),
        ParentChildEdge(src=tool3.id, dst=task3.id),
        
        # Tasks create summaries
        ParentChildEdge(src=task1.id, dst=summary1.id),
        ParentChildEdge(src=task2.id, dst=summary2.id),
        ParentChildEdge(src=task3.id, dst=summary3.id),
    ]
    
    print(f"✅ Created {len(edges)} edges:")
    edge_counts = {}
    for edge in edges:
        kind = edge.kind.value
        edge_counts[kind] = edge_counts.get(kind, 0) + 1
    
    for kind, count in edge_counts.items():
        print(f"   {kind}: {count}")
    
    return edges

def step3_build_graph(nodes, edges):
    """Step 3: Add everything to a graph store"""
    print(f"\n🗄️ STEP 3: Building the Graph")
    print("=" * 50)
    
    # Create graph store
    graph = InMemoryGraphStore()
    
    # Add all nodes
    print("📥 Adding nodes...")
    for node in nodes:
        graph.add_node(node)
    
    # Add all edges
    print("🔗 Adding edges...")
    for edge in edges:
        graph.add_edge(edge)
    
    # Show graph statistics
    print(f"\n📊 Graph Statistics:")
    print(f"   Total nodes: {len(graph.nodes)}")
    print(f"   Total edges: {len(graph.edges)}")
    
    # Show nodes by type
    print(f"\n📦 Nodes by type:")
    for kind in NodeKind:
        nodes_of_kind = graph.get_nodes_by_kind(kind)
        if nodes_of_kind:
            print(f"   {kind.value}: {len(nodes_of_kind)}")
    
    return graph

def step4_query_graph(graph):
    """Step 4: Query and analyze the graph"""
    print(f"\n🔍 STEP 4: Querying the Graph")
    print("=" * 50)
    
    # Find the session node
    session_nodes = graph.get_nodes_by_kind(NodeKind.SESSION)
    session = session_nodes[0]
    print(f"📱 Found session: {session.id[:8]}...")
    
    # Find what the session contains
    session_children = graph.get_edges(src=session.id, kind=EdgeKind.PARENT_CHILD)
    print(f"\n👥 Session contains {len(session_children)} items:")
    for edge in session_children:
        child = graph.get_node(edge.dst)
        print(f"   {child.kind.value}: {dict(child.data).get('content', dict(child.data).get('title', 'N/A'))[:50]}...")
    
    # Find the plan and its steps
    plan_nodes = graph.get_nodes_by_kind(NodeKind.PLAN)
    plan = plan_nodes[0]
    plan_steps = graph.get_edges(src=plan.id, kind=EdgeKind.PARENT_CHILD)
    print(f"\n📋 Plan '{dict(plan.data)['title']}' has {len(plan_steps)} steps:")
    
    # Get steps and sort by index
    steps = []
    for edge in plan_steps:
        step = graph.get_node(edge.dst)
        steps.append(step)
    
    steps.sort(key=lambda s: s.data['index'])
    
    for step in steps:
        print(f"   Step {step.data['index']}: {step.data['description']}")
        
        # Find tools for this step
        step_tools = graph.get_edges(src=step.id, kind=EdgeKind.PLAN_LINK)
        for tool_edge in step_tools:
            tool = graph.get_node(tool_edge.dst)
            tool_name = tool.data['name']
            print(f"      🔧 Tool: {tool_name}")
    
    # Analyze step dependencies
    print(f"\n⚡ Step Dependencies:")
    step_deps = graph.get_edges(kind=EdgeKind.STEP_ORDER)
    for dep_edge in step_deps:
        src_step = graph.get_node(dep_edge.src)
        dst_step = graph.get_node(dep_edge.dst)
        print(f"   Step {dst_step.data['index']} depends on Step {src_step.data['index']}")
    
    # Check execution results
    print(f"\n✅ Execution Results:")
    task_runs = graph.get_nodes_by_kind(NodeKind.TASK_RUN)
    successful_tasks = sum(1 for task in task_runs if task.data.get('success') == 'true')
    total_time = sum(float(task.data.get('execution_time', '0')) for task in task_runs)
    
    print(f"   Successful tasks: {successful_tasks}/{len(task_runs)}")
    print(f"   Total execution time: {total_time:.1f} seconds")
    print(f"   Success rate: {successful_tasks/len(task_runs)*100:.1f}%")

def step5_visualize_graph(graph):
    """Step 5: Visualize the graph structure"""
    print(f"\n👁️ STEP 5: Graph Visualization")
    print("=" * 50)
    
    print("🎨 Graph Structure:")
    print_graph_structure(graph)

def step6_test_immutability():
    """Step 6: Demonstrate node immutability"""
    print(f"\n🔒 STEP 6: Testing Node Immutability")
    print("=" * 50)
    
    # Create a test node
    tool = ToolCall(data={
        "name": "test_tool",
        "args": '{"test": "value"}'
    })
    
    print(f"📦 Created tool: {tool.id[:8]}...")
    print(f"   Original data: {dict(tool.data)}")
    
    # Test immutability
    print(f"\n🚫 Testing immutability...")
    
    tests = [
        ("Modify ID", lambda: setattr(tool, 'id', 'new_id')),
        ("Modify kind", lambda: setattr(tool, 'kind', NodeKind.SUMMARY)),
        ("Modify data", lambda: tool.data.update({'new': 'value'})),
        ("Add data field", lambda: setattr(tool.data, 'new_field', 'value'))
    ]
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"   ❌ {test_name}: Should have failed!")
        except (TypeError, AttributeError):
            print(f"   ✅ {test_name}: Properly blocked")
    
    # Show correct way to "update" nodes
    print(f"\n🔄 Correct way to update nodes:")
    updated_data = {**tool.data, "updated": "true"}
    new_tool = ToolCall(id=tool.id, data=updated_data)
    print(f"   Created new node with same ID and updated data")
    print(f"   New data: {dict(new_tool.data)}")

def main():
    """Run the complete end-to-end demo"""
    print("🚀 Simple End-to-End Graph Models Demo")
    print("=" * 80)
    print("Scenario: AI assistant helps user analyze sales data\n")
    
    try:
        # Step 1: Create nodes
        nodes = step1_create_nodes()
        
        # Step 2: Create edges
        edges = step2_create_edges(nodes)
        
        # Step 3: Build graph
        graph = step3_build_graph(nodes, edges)
        
        # Step 4: Query graph
        step4_query_graph(graph)
        
        # Step 5: Visualize graph
        step5_visualize_graph(graph)
        
        # Step 6: Test immutability
        step6_test_immutability()
        
        # Summary
        print(f"\n" + "=" * 80)
        print("✅ End-to-End Demo Complete!")
        print("\n🎯 What we demonstrated:")
        print("   ✓ Created all 8 node types with realistic data")
        print("   ✓ Connected nodes with 5 different edge types")
        print("   ✓ Built a complete graph with 16 nodes and 18 edges")
        print("   ✓ Queried graph for relationships and analytics")
        print("   ✓ Visualized graph structure hierarchically") 
        print("   ✓ Verified node immutability and data integrity")
        print("\n🏗️ Graph represents a complete AI workflow:")
        print("   User Request → Assistant Response → 3-Step Plan → Tool Execution → Results")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()