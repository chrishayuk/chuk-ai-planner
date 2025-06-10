#!/usr/bin/env python3
# examples/graph_models_demo_e2e.py
"""
Graph Models Demo: chuk_ai_planner Core Models and Graph Operations
=================================================================

This demo showcases:
1. Creating different types of graph nodes
2. Working with graph edges and relationships
3. Graph store operations (add, get, query)
4. Node immutability and validation
5. Graph visualization utilities
6. Real-world graph structure examples

Run with: python demo_graph_models.py
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Core imports
from chuk_ai_planner.models import (
    NodeKind, GraphNode,
    SessionNode, PlanNode, PlanStep,
    UserMessage, AssistantMessage,
    ToolCall, TaskRun, Summary
)

from chuk_ai_planner.models.edges import (
    EdgeKind, GraphEdge,
    ParentChildEdge, NextEdge,
    PlanEdge, StepEdge
)

from chuk_ai_planner.store.memory import InMemoryGraphStore
from chuk_ai_planner.utils.visualization import print_graph_structure

def demo_node_creation():
    """Demonstrate creating different types of nodes."""
    print("\n" + "="*60)
    print("🔷 DEMO 1: Node Creation and Properties")
    print("="*60)
    
    # Create different types of nodes
    session = SessionNode(data={"user_id": "user123", "started_at": datetime.now(timezone.utc).isoformat()})
    
    user_msg = UserMessage(data={"content": "Hello, I need help with a data analysis task"})
    
    assistant_msg = AssistantMessage(data={
        "content": "I'll help you with data analysis. Let me break this down into steps.",
        "tool_calls": []
    })
    
    plan = PlanNode(data={
        "title": "Data Analysis Plan",
        "description": "Comprehensive data analysis workflow"
    })
    
    step1 = PlanStep(data={
        "index": "1",
        "description": "Load and examine the dataset"
    })
    
    step2 = PlanStep(data={
        "index": "2", 
        "description": "Clean and preprocess the data"
    })
    
    tool_call = ToolCall(data={
        "name": "pandas_read_csv",
        "args": {"filepath": "data.csv", "delimiter": ","},
        "result": {"shape": [1000, 15], "columns": ["col1", "col2"]},
        "cached": False
    })
    
    task_run = TaskRun(data={
        "success": True,
        "execution_time": 2.3,
        "memory_used": "45MB"
    })
    
    summary = Summary(data={
        "content": "Successfully loaded dataset with 1000 rows and 15 columns"
    })
    
    # Display node information
    nodes = [session, user_msg, assistant_msg, plan, step1, step2, tool_call, task_run, summary]
    
    for node in nodes:
        print(f"\n📦 {node.__class__.__name__}")
        print(f"   ID: {node.id[:8]}...")
        print(f"   Kind: {node.kind.value}")
        print(f"   Timestamp: {node.ts.strftime('%H:%M:%S')}")
        print(f"   Data: {dict(node.data)}")
        
        # Demonstrate immutability
        try:
            node.data["new_field"] = "should fail"
        except (TypeError, AttributeError) as e:
            print(f"   ✅ Immutability enforced: {type(e).__name__}")
    
    return nodes

def demo_edge_creation():
    """Demonstrate creating different types of edges."""
    print("\n" + "="*60)
    print("🔗 DEMO 2: Edge Creation and Relationships")
    print("="*60)
    
    # Create some nodes for edge examples
    session = SessionNode()
    user_msg = UserMessage()
    assistant_msg = AssistantMessage()
    tool_call = ToolCall()
    task_run = TaskRun()
    plan = PlanNode()
    step1 = PlanStep()
    step2 = PlanStep()
    
    # Create different types of edges
    edges = [
        ParentChildEdge(src=session.id, dst=user_msg.id, data={"relationship": "contains"}),
        ParentChildEdge(src=session.id, dst=assistant_msg.id),
        ParentChildEdge(src=assistant_msg.id, dst=tool_call.id),
        ParentChildEdge(src=tool_call.id, dst=task_run.id),
        
        NextEdge(src=user_msg.id, dst=assistant_msg.id, data={"sequence": 1}),
        
        ParentChildEdge(src=plan.id, dst=step1.id),
        ParentChildEdge(src=plan.id, dst=step2.id),
        StepEdge(src=step1.id, dst=step2.id, data={"dependency": "step1_must_complete_first"}),
        
        PlanEdge(src=step1.id, dst=tool_call.id, data={"execution_order": 1}),
        
        GraphEdge(kind=EdgeKind.CUSTOM, src=step1.id, dst=step2.id, data={
            "custom_type": "data_flow",
            "variables": ["dataset", "cleaned_data"]
        })
    ]
    
    # Display edge information
    for edge in edges:
        print(f"\n🔗 {edge.__class__.__name__}")
        print(f"   ID: {edge.id[:8]}...")
        print(f"   Kind: {edge.kind.value}")
        print(f"   Direction: {edge.src[:8]}... → {edge.dst[:8]}...")
        if edge.data:
            print(f"   Data: {dict(edge.data)}")
    
    return edges, [session, user_msg, assistant_msg, tool_call, task_run, plan, step1, step2]

def demo_graph_store_operations():
    """Demonstrate graph store operations."""
    print("\n" + "="*60)
    print("🗄️ DEMO 3: Graph Store Operations")
    print("="*60)
    
    # Create a graph store
    graph = InMemoryGraphStore()
    
    # Create a conversation scenario
    session = SessionNode(data={"conversation_id": "conv_001"})
    user_msg = UserMessage(data={"content": "Analyze sales data from Q3"})
    assistant_msg = AssistantMessage(data={"content": "I'll analyze the Q3 sales data for you."})
    
    # Create a plan with steps
    plan = PlanNode(data={
        "title": "Q3 Sales Analysis",
        "description": "Comprehensive analysis of Q3 sales performance"
    })
    
    steps = [
        PlanStep(data={"index": "1", "description": "Load Q3 sales data"}),
        PlanStep(data={"index": "2", "description": "Calculate key metrics"}),
        PlanStep(data={"index": "3", "description": "Generate visualizations"}),
        PlanStep(data={"index": "4", "description": "Create summary report"})
    ]
    
    # Create tool calls for each step
    tools = [
        ToolCall(data={"name": "load_csv", "args": {"file": "q3_sales.csv"}}),
        ToolCall(data={"name": "calculate_metrics", "args": {"columns": ["revenue", "units"]}}),
        ToolCall(data={"name": "create_chart", "args": {"type": "line", "x": "month", "y": "revenue"}}),
        ToolCall(data={"name": "generate_report", "args": {"template": "executive_summary"}})
    ]
    
    # Add all nodes to the graph
    all_nodes = [session, user_msg, assistant_msg, plan] + steps + tools
    
    print("📥 Adding nodes to graph store...")
    for node in all_nodes:
        graph.add_node(node)
        print(f"   Added {node.kind.value}: {node.id[:8]}...")
    
    # Create and add edges
    edges = [
        # Session contains messages and plan
        ParentChildEdge(src=session.id, dst=user_msg.id),
        ParentChildEdge(src=session.id, dst=assistant_msg.id),
        ParentChildEdge(src=session.id, dst=plan.id),
        
        # Message sequence
        NextEdge(src=user_msg.id, dst=assistant_msg.id),
        
        # Plan contains steps
        ParentChildEdge(src=plan.id, dst=steps[0].id),
        ParentChildEdge(src=plan.id, dst=steps[1].id),
        ParentChildEdge(src=plan.id, dst=steps[2].id),
        ParentChildEdge(src=plan.id, dst=steps[3].id),
        
        # Step dependencies (linear workflow)
        StepEdge(src=steps[0].id, dst=steps[1].id),
        StepEdge(src=steps[1].id, dst=steps[2].id),
        StepEdge(src=steps[2].id, dst=steps[3].id),
        
        # Steps link to tools
        PlanEdge(src=steps[0].id, dst=tools[0].id),
        PlanEdge(src=steps[1].id, dst=tools[1].id),
        PlanEdge(src=steps[2].id, dst=tools[2].id),
        PlanEdge(src=steps[3].id, dst=tools[3].id),
    ]
    
    print(f"\n🔗 Adding {len(edges)} edges to graph store...")
    for edge in edges:
        graph.add_edge(edge)
    
    # Demonstrate querying operations
    print(f"\n🔍 Graph Store Statistics:")
    print(f"   Total nodes: {len(graph.nodes)}")
    print(f"   Total edges: {len(graph.edges)}")
    
    # Query by node kind
    print(f"\n📊 Nodes by kind:")
    for kind in NodeKind:
        nodes_of_kind = graph.get_nodes_by_kind(kind)
        if nodes_of_kind:
            print(f"   {kind.value}: {len(nodes_of_kind)}")
    
    # Query edges by kind
    print(f"\n🔗 Edges by kind:")
    for kind in EdgeKind:
        edges_of_kind = graph.get_edges(kind=kind)
        if edges_of_kind:
            print(f"   {kind.value}: {len(edges_of_kind)}")
    
    # Find children of session
    session_children = graph.get_edges(src=session.id, kind=EdgeKind.PARENT_CHILD)
    print(f"\n👥 Session has {len(session_children)} direct children:")
    for edge in session_children:
        child_node = graph.get_node(edge.dst)
        print(f"   {child_node.kind.value}: {child_node.id[:8]}...")
    
    # Find plan steps
    plan_steps = graph.get_edges(src=plan.id, kind=EdgeKind.PARENT_CHILD)
    print(f"\n📋 Plan has {len(plan_steps)} steps:")
    for edge in plan_steps:
        step_node = graph.get_node(edge.dst)
        step_data = dict(step_node.data)
        print(f"   Step {step_data.get('index')}: {step_data.get('description')}")
    
    # Find step dependencies
    print(f"\n⚡ Step dependencies:")
    for step in steps:
        deps = graph.get_edges(dst=step.id, kind=EdgeKind.STEP_ORDER)
        if deps:
            for dep_edge in deps:
                dep_step = graph.get_node(dep_edge.src)
                print(f"   Step {dict(step.data).get('index')} depends on Step {dict(dep_step.data).get('index')}")
    
    return graph

def demo_node_immutability():
    """Demonstrate node immutability features."""
    print("\n" + "="*60)
    print("🔒 DEMO 4: Node Immutability and Validation")
    print("="*60)
    
    # Create a node
    tool_call = ToolCall(data={
        "name": "example_tool",
        "args": {"param1": "value1"},
        "result": {"output": "success"}
    })
    
    print(f"📦 Created ToolCall: {tool_call.id[:8]}...")
    print(f"   Original data: {dict(tool_call.data)}")
    
    # Try to modify the node (should fail)
    print(f"\n🚫 Testing immutability...")
    
    try:
        tool_call.id = "new_id"
        print("   ❌ ID modification should have failed!")
    except TypeError:
        print("   ✅ ID modification properly blocked")
    
    try:
        tool_call.kind = NodeKind.SUMMARY
        print("   ❌ Kind modification should have failed!")
    except TypeError:
        print("   ✅ Kind modification properly blocked")
    
    try:
        tool_call.data["new_field"] = "new_value"
        print("   ❌ Data modification should have failed!")
    except TypeError:
        print("   ✅ Data modification properly blocked")
    
    # Show that we can read the data but not modify it
    print(f"\n📖 Data access (read-only):")
    print(f"   tool_call.data['name'] = '{tool_call.data['name']}'")
    print(f"   tool_call.data.keys() = {list(tool_call.data.keys())}")
    
    # Demonstrate creating a new node with updated data
    print(f"\n🔄 Creating updated node (proper way to 'modify'):")
    updated_data = {**tool_call.data, "cached": True, "updated_at": datetime.now(timezone.utc).isoformat()}
    
    new_tool_call = ToolCall(
        id=tool_call.id,  # Keep same ID
        data=updated_data
    )
    
    print(f"   New data: {dict(new_tool_call.data)}")

def demo_complex_graph_scenario():
    """Demonstrate a complex real-world graph scenario."""
    print("\n" + "="*60)
    print("🏗️ DEMO 5: Complex Graph Scenario - AI Research Assistant")
    print("="*60)
    
    graph = InMemoryGraphStore()
    
    # Create a research session
    session = SessionNode(data={
        "session_type": "research_assistant",
        "user_id": "researcher_001",
        "topic": "machine_learning_optimization"
    })
    
    # User's research request
    user_msg = UserMessage(data={
        "content": "I need to research recent advances in neural network optimization techniques, particularly focusing on adaptive learning rates and gradient clipping methods."
    })
    
    # Assistant's response with plan
    assistant_msg = AssistantMessage(data={
        "content": "I'll help you research neural network optimization techniques. Let me create a comprehensive research plan.",
        "tool_calls": ["search_papers", "analyze_trends", "summarize_findings"]
    })
    
    # Research plan
    research_plan = PlanNode(data={
        "title": "Neural Network Optimization Research",
        "description": "Comprehensive research on adaptive learning rates and gradient clipping",
        "research_scope": ["adaptive_learning_rates", "gradient_clipping", "optimization_algorithms"],
        "time_frame": "recent_5_years"
    })
    
    # Research steps
    research_steps = [
        PlanStep(data={
            "index": "1",
            "description": "Search for recent papers on adaptive learning rates",
            "search_terms": ["adaptive learning rate", "learning rate scheduling", "AdaGrad", "Adam", "RMSprop"]
        }),
        PlanStep(data={
            "index": "2", 
            "description": "Search for papers on gradient clipping techniques",
            "search_terms": ["gradient clipping", "gradient explosion", "gradient norm clipping"]
        }),
        PlanStep(data={
            "index": "3",
            "description": "Analyze citation patterns and trending topics",
            "analysis_type": "citation_network"
        }),
        PlanStep(data={
            "index": "4",
            "description": "Synthesize findings and identify key innovations",
            "output_format": "structured_summary"
        })
    ]
    
    # Tool calls for each research step
    research_tools = [
        ToolCall(data={
            "name": "academic_search",
            "args": {
                "query": "adaptive learning rate neural networks",
                "databases": ["arxiv", "ieee", "acm"],
                "date_range": "2019-2024"
            },
            "result": {
                "papers_found": 156,
                "relevant_papers": 23,
                "top_venues": ["ICML", "NeurIPS", "ICLR"]
            }
        }),
        ToolCall(data={
            "name": "academic_search", 
            "args": {
                "query": "gradient clipping techniques deep learning",
                "databases": ["arxiv", "ieee", "acm"],
                "date_range": "2019-2024"
            },
            "result": {
                "papers_found": 89,
                "relevant_papers": 17,
                "top_methods": ["norm_clipping", "value_clipping", "adaptive_clipping"]
            }
        }),
        ToolCall(data={
            "name": "citation_analysis",
            "args": {
                "paper_ids": ["paper_001", "paper_002", "paper_003"],
                "analysis_depth": "2_hops"
            },
            "result": {
                "citation_clusters": 3,
                "trending_topics": ["attention_mechanisms", "transformer_optimization"],
                "influential_authors": ["Smith, J.", "Chen, L.", "Wang, M."]
            }
        }),
        ToolCall(data={
            "name": "research_synthesizer",
            "args": {
                "input_papers": 40,
                "synthesis_method": "thematic_analysis"
            },
            "result": {
                "key_findings": [
                    "Adaptive learning rates show 15-30% improvement in convergence",
                    "Gradient clipping prevents training instability in 85% of cases",
                    "Combined approaches yield best results"
                ],
                "future_directions": ["neuromorphic_optimization", "quantum_gradients"]
            }
        })
    ]
    
    # Task runs showing execution results
    task_runs = [
        TaskRun(data={
            "success": True,
            "execution_time": 45.2,
            "papers_processed": 156,
            "cache_hits": 12
        }),
        TaskRun(data={
            "success": True,
            "execution_time": 32.7,
            "papers_processed": 89,
            "cache_hits": 8
        }),
        TaskRun(data={
            "success": True,
            "execution_time": 18.5,
            "citations_analyzed": 1250,
            "clusters_identified": 3
        }),
        TaskRun(data={
            "success": True,
            "execution_time": 67.3,
            "synthesis_quality": 0.89,
            "findings_extracted": 15
        })
    ]
    
    # Summaries for each step
    summaries = [
        Summary(data={
            "content": "Found 23 highly relevant papers on adaptive learning rates. Key algorithms include Adam variants and novel scheduling techniques."
        }),
        Summary(data={
            "content": "Identified 17 papers on gradient clipping with focus on norm-based and adaptive methods showing significant stability improvements."
        }),
        Summary(data={
            "content": "Citation analysis reveals three main research clusters with emerging focus on attention mechanism optimization."
        }),
        Summary(data={
            "content": "Research synthesis complete: Adaptive learning rates + gradient clipping combination shows strongest empirical results across datasets."
        })
    ]
    
    # Add all nodes to graph
    all_nodes = ([session, user_msg, assistant_msg, research_plan] + 
                research_steps + research_tools + task_runs + summaries)
    
    for node in all_nodes:
        graph.add_node(node)
    
    # Create comprehensive edge structure
    edges = [
        # Session structure
        ParentChildEdge(src=session.id, dst=user_msg.id),
        ParentChildEdge(src=session.id, dst=assistant_msg.id),
        ParentChildEdge(src=session.id, dst=research_plan.id),
        NextEdge(src=user_msg.id, dst=assistant_msg.id),
        
        # Plan structure
        *[ParentChildEdge(src=research_plan.id, dst=step.id) for step in research_steps],
        
        # Step dependencies
        *[StepEdge(src=research_steps[i].id, dst=research_steps[i+1].id) 
          for i in range(len(research_steps)-1)],
        
        # Tool execution chains
        *[PlanEdge(src=research_steps[i].id, dst=research_tools[i].id) 
          for i in range(len(research_steps))],
        *[ParentChildEdge(src=research_tools[i].id, dst=task_runs[i].id) 
          for i in range(len(research_tools))],
        *[ParentChildEdge(src=task_runs[i].id, dst=summaries[i].id) 
          for i in range(len(task_runs))],
    ]
    
    for edge in edges:
        graph.add_edge(edge)
    
    # Display complex graph statistics
    print(f"📊 Complex Research Graph Created:")
    print(f"   Total nodes: {len(graph.nodes)}")
    print(f"   Total edges: {len(graph.edges)}")
    print(f"   Research steps: {len(research_steps)}")
    print(f"   Tools executed: {len(research_tools)}")
    print(f"   Successful task runs: {sum(1 for tr in task_runs if tr.data.get('success'))}")
    
    # Analyze the research workflow
    print(f"\n🔬 Research Workflow Analysis:")
    total_papers = sum(tool.data.get('result', {}).get('papers_found', 0) for tool in research_tools[:2])
    total_time = sum(tr.data.get('execution_time', 0) for tr in task_runs)
    
    print(f"   Papers analyzed: {total_papers}")
    print(f"   Total execution time: {total_time:.1f} seconds")
    print(f"   Research efficiency: {total_papers/total_time:.1f} papers/second")
    
    # Show research findings
    synthesis_result = research_tools[-1].data.get('result', {})
    if 'key_findings' in synthesis_result:
        print(f"\n🎯 Key Research Findings:")
        for i, finding in enumerate(synthesis_result['key_findings'], 1):
            print(f"   {i}. {finding}")
    
    return graph

def demo_visualization():
    """Demonstrate graph visualization capabilities."""
    print("\n" + "="*60)
    print("👁️ DEMO 6: Graph Visualization")
    print("="*60)
    
    # Create a simple but complete graph
    graph = InMemoryGraphStore()
    
    # Create nodes
    session = SessionNode(data={"name": "Demo Session"})
    plan = PlanNode(data={"title": "Simple Demo Plan", "description": "A plan for demonstration"})
    step1 = PlanStep(data={"index": "1", "description": "First step"})
    step2 = PlanStep(data={"index": "2", "description": "Second step"})
    tool1 = ToolCall(data={"name": "demo_tool", "args": {"param": "value"}})
    tool2 = ToolCall(data={"name": "another_tool", "args": {"setting": "test"}})
    
    # Add nodes
    for node in [session, plan, step1, step2, tool1, tool2]:
        graph.add_node(node)
    
    # Add edges
    edges = [
        ParentChildEdge(src=session.id, dst=plan.id),
        ParentChildEdge(src=plan.id, dst=step1.id),
        ParentChildEdge(src=plan.id, dst=step2.id),
        StepEdge(src=step1.id, dst=step2.id),
        PlanEdge(src=step1.id, dst=tool1.id),
        PlanEdge(src=step2.id, dst=tool2.id),
    ]
    
    for edge in edges:
        graph.add_edge(edge)
    
    print("🎨 Using built-in visualization:")
    print_graph_structure(graph)

def main():
    """Run all demos."""
    print("🚀 Starting chuk_ai_planner Graph Models Demo")
    print("=" * 80)
    
    try:
        # Run all demos
        demo_node_creation()
        demo_edge_creation()
        demo_graph_store_operations()
        demo_node_immutability()
        demo_complex_graph_scenario()
        demo_visualization()
        
        print("\n" + "="*80)
        print("✅ All demos completed successfully!")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()