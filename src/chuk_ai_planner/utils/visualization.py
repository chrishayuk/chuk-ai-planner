# chuk_ai_planner/utils/visualization.py
"""
Visualization utilities for sessions and graph structures.

These functions help with visualizing the structure of sessions and graphs
in a human-readable format for debugging and presentation.

Updated to use chuk-ai-session-manager and handle variations in EventType enums.
"""

from typing import Dict, List, Any, Optional, Union
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_planner.models import NodeKind
from chuk_ai_planner.models.edges import EdgeKind

from ..store.base import GraphStore


def print_session_events(session: 'Session') -> None:
    """
    Print session events in a hierarchical tree structure.
    
    This shows the parent-child relationships between events and highlights
    specific event types like tool calls.
    
    Parameters
    ----------
    session : Session
        The session whose events will be printed
    """    
    events = session.events
    
    print(f"\n==== SESSION EVENTS ({len(events)}) ====")
    
    if not events:
        print("No events found in session")
        return
    
    # Build parent-child relationships
    children = {}
    for event in events:
        parent_id = None
        if hasattr(event, 'metadata') and event.metadata:
            parent_id = event.metadata.get("parent_event_id")
        
        if parent_id:
            if parent_id not in children:
                children[parent_id] = []
            children[parent_id].append(event)
    
    # Print the tree
    def print_event(event, indent=0):
        prefix = "  " * indent
        
        # Get event type safely
        event_type = "unknown"
        if hasattr(event, 'type') and event.type:
            if hasattr(event.type, 'value'):
                event_type = event.type.value
            else:
                event_type = str(event.type)
        
        event_id = getattr(event, 'id', 'no-id')
        print(f"{prefix}• {event_type:10} id={event_id}")
        
        # Get message safely
        message = {}
        if hasattr(event, 'message'):
            if isinstance(event.message, dict):
                message = event.message
            elif hasattr(event.message, '__dict__'):
                message = event.message.__dict__
            else:
                message = {"content": str(event.message)}
        
        # Handle different event types
        if EventType and hasattr(EventType, 'TOOL_CALL') and event.type == EventType.TOOL_CALL:
            tool_name = message.get("tool", "unknown")
            has_error = message.get("error") is not None
            error_str = "error=Yes" if has_error else "error=None"
            print(f"{prefix}  ⇒ {tool_name:10} {error_str}")
        elif EventType and hasattr(EventType, 'SUMMARY') and event.type == EventType.SUMMARY:
            if "description" in message:
                print(f"{prefix}  ⇒ {message.get('description')}")
            elif "note" in message:
                print(f"{prefix}  ⇒ Note: {message.get('note')}")
            elif "step_id" in message:
                status = message.get("status", "unknown")
                print(f"{prefix}  ⇒ Step {message.get('step_id')}: {status}")
        # Handle error events - check by event type name or by message content
        elif hasattr(event.type, 'name') and event.type.name in ('ERROR', 'EXCEPTION', 'FAILURE'):
            print(f"{prefix}  ⇒ Error: {message.get('error', 'Unknown error')}")
        elif 'error' in message:
            # If the message contains an error field, show it regardless of event type
            print(f"{prefix}  ⇒ Error: {message.get('error', 'Unknown error')}")
        elif 'content' in message:
            # Show content for message events
            content = str(message.get('content', ''))
            if len(content) > 100:
                content = content[:97] + "..."
            print(f"{prefix}  ⇒ {content}")
        
        # Print children recursively
        for child in children.get(event.id, []):
            print_event(child, indent + 1)
    
    # Find all root events (those without a parent)
    roots = []
    for e in events:
        parent_id = None
        if hasattr(e, 'metadata') and e.metadata:
            parent_id = e.metadata.get("parent_event_id")
        if not parent_id:
            roots.append(e)
    
    if not roots:
        print("No root events found - all events have parents")
        # Show first few events anyway
        for i, event in enumerate(events[:5]):
            print_event(event)
            if i < len(events) - 1:
                print()
    else:
        for root in roots:
            print_event(root)


def print_graph_structure(graph_store: GraphStore) -> None:
    """
    Print the structure of the graph in a human-readable format.
    
    This shows node types, their connections, and important relationships
    like plan steps and tool executions.
    
    Parameters
    ----------
    graph_store : GraphStore
        The graph store containing the nodes and edges to visualize
    """
    # Get all nodes and edges
    nodes = []
    if hasattr(graph_store, "nodes"):
        # InMemoryGraphStore has a nodes dict
        nodes = list(graph_store.nodes.values())
    else:
        # Try to get all nodes through get_nodes_by_kind if available
        try:
            for kind in NodeKind:
                nodes.extend(graph_store.get_nodes_by_kind(kind))
        except (AttributeError, NotImplementedError):
            print("Warning: Unable to retrieve nodes from graph store")
    
    # Get edges
    edges = []
    if hasattr(graph_store, "edges"):
        # InMemoryGraphStore has an edges dict or list
        if isinstance(graph_store.edges, dict):
            edges = list(graph_store.edges.values())
        else:
            edges = graph_store.edges
    else:
        # If there's no direct access to edges, we can't easily list them all
        # This would require querying edges for all node combinations
        print("Warning: Unable to retrieve all edges from graph store")
    
    print("\n==== GRAPH STRUCTURE ====")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")
    
    if not nodes:
        print("No nodes found in graph")
        return
    
    # Group nodes by kind
    nodes_by_kind = {}
    for node in nodes:
        kind = node.kind.value if hasattr(node.kind, 'value') else str(node.kind)
        if kind not in nodes_by_kind:
            nodes_by_kind[kind] = []
        nodes_by_kind[kind].append(node)
    
    print("\nNodes by type:")
    for kind, kind_nodes in nodes_by_kind.items():
        print(f"  {kind}: {len(kind_nodes)}")
    
    # Find the session node(s)
    session_nodes = nodes_by_kind.get("session", [])
    if not session_nodes:
        print("\nNo session nodes found. Showing all root nodes:")
        # Show nodes that have no incoming edges
        incoming_nodes = set()
        for edge in edges:
            if hasattr(edge, 'dst'):
                incoming_nodes.add(edge.dst)
        
        root_nodes = [n for n in nodes if n.id not in incoming_nodes]
        for node in root_nodes[:5]:  # Show first 5 root nodes
            print(f"  {node.kind.value if hasattr(node.kind, 'value') else node.kind}: {node!r}")
        return
    
    # For each session, show its direct children
    for session_node in session_nodes:
        print(f"\nSession: {session_node!r}")
        
        # Find direct children
        child_edges = [e for e in edges if hasattr(e, 'src') and e.src == session_node.id]
        
        for i, edge in enumerate(child_edges):
            is_last = i == len(child_edges) - 1
            prefix = "└──" if is_last else "├──"
            
            child = None
            for node in nodes:
                if node.id == edge.dst:
                    child = node
                    break
                    
            if child:
                child_kind = child.kind.value if hasattr(child.kind, 'value') else str(child.kind)
                print(f"{prefix} {child_kind}: {child!r}")
                
                # If this is a plan node, show its steps
                if child.kind == NodeKind.PLAN:
                    _print_plan_structure(graph_store, child, nodes, edges, "    ")
                
                # If this is an assistant message, show its tool calls
                elif child.kind == NodeKind.ASSIST_MSG:
                    _print_assistant_structure(graph_store, child, nodes, edges, "    ")


def _print_plan_structure(
    graph_store: GraphStore, 
    plan_node: Any, 
    nodes: List[Any], 
    edges: List[Any],
    indent: str
) -> None:
    """
    Print the structure of a plan node.
    
    Parameters
    ----------
    graph_store : GraphStore
        The graph store
    plan_node : Any
        The plan node to print
    nodes : List[Any]
        All nodes in the graph
    edges : List[Any]
        All edges in the graph
    indent : str
        Indentation string for formatting
    """
    # Find all steps linked to this plan
    step_edges = [
        e for e in edges 
        if hasattr(e, 'src') and e.src == plan_node.id and 
        any(n.id == e.dst and n.kind == NodeKind.PLAN_STEP for n in nodes)
    ]
    
    for i, step_edge in enumerate(step_edges):
        is_last = i == len(step_edges) - 1
        prefix = f"{indent}└──" if is_last else f"{indent}├──"
        
        step = None
        for node in nodes:
            if node.id == step_edge.dst:
                step = node
                break
                
        if step:
            step_index = step.data.get("index", i+1) if hasattr(step, 'data') and step.data else i+1
            step_desc = step.data.get("description", "Unknown step") if hasattr(step, 'data') and step.data else "Unknown step"
            print(f"{prefix} Step {step_index}: {step_desc}")
            
            # Show tool executions for this step
            tool_edges = [
                e for e in edges 
                if hasattr(e, 'src') and hasattr(e, 'kind') and e.src == step.id and e.kind == EdgeKind.PLAN_LINK
            ]
            
            next_indent = indent + ("    " if is_last else "│   ")
            
            for j, tool_edge in enumerate(tool_edges):
                is_last_tool = j == len(tool_edges) - 1
                tool_prefix = f"{next_indent}└──" if is_last_tool else f"{next_indent}├──"
                
                tool = None
                for node in nodes:
                    if node.id == tool_edge.dst:
                        tool = node
                        break
                        
                if tool:
                    tool_name = tool.data.get("name", "unknown tool") if hasattr(tool, 'data') and tool.data else "unknown tool"
                    print(f"{tool_prefix} {tool_name}: {tool!r}")


def _print_assistant_structure(
    graph_store: GraphStore, 
    assistant_node: Any, 
    nodes: List[Any], 
    edges: List[Any],
    indent: str
) -> None:
    """
    Print the structure of an assistant message node.
    
    Parameters
    ----------
    graph_store : GraphStore
        The graph store
    assistant_node : Any
        The assistant message node to print
    nodes : List[Any]
        All nodes in the graph
    edges : List[Any]
        All edges in the graph
    indent : str
        Indentation string for formatting
    """
    # Find tool calls linked to this assistant message
    tool_edges = [
        e for e in edges 
        if hasattr(e, 'src') and e.src == assistant_node.id and 
        any(n.id == e.dst and n.kind == NodeKind.TOOL_CALL for n in nodes)
    ]
    
    for i, tool_edge in enumerate(tool_edges):
        is_last = i == len(tool_edges) - 1
        prefix = f"{indent}└──" if is_last else f"{indent}├──"
        
        tool = None
        for node in nodes:
            if node.id == tool_edge.dst:
                tool = node
                break
                
        if tool:
            tool_name = tool.data.get("name", "unknown") if hasattr(tool, 'data') and tool.data else "unknown"
            print(f"{prefix} Tool: {tool_name}")
            
            # Show task run for this tool
            task_edges = [
                e for e in edges 
                if hasattr(e, 'src') and e.src == tool.id and 
                any(n.id == e.dst and n.kind == NodeKind.TASK_RUN for n in nodes)
            ]
            
            next_indent = indent + ("    " if is_last else "│   ")
            
            for task_edge in task_edges:
                task = None
                for node in nodes:
                    if node.id == task_edge.dst:
                        task = node
                        break
                        
                if task:
                    success = "✓" if (hasattr(task, 'data') and task.data and task.data.get("success", False)) else "✗"
                    print(f"{next_indent}└── Task: {success} ({task!r})")


def print_graph_summary(graph_store: GraphStore) -> None:
    """
    Print a concise summary of the graph.
    
    Parameters
    ----------
    graph_store : GraphStore
        The graph store to summarize
    """
    nodes = []
    if hasattr(graph_store, "nodes"):
        nodes = list(graph_store.nodes.values())
    
    edges = []
    if hasattr(graph_store, "edges"):
        if isinstance(graph_store.edges, dict):
            edges = list(graph_store.edges.values())
        else:
            edges = graph_store.edges
    
    print(f"\n📋 Graph Summary")
    print("=" * 20)
    
    node_count = len(nodes)
    edge_count = len(edges)
    
    if node_count == 0:
        print("Empty graph")
        return
    
    # Basic stats
    print(f"Nodes: {node_count}, Edges: {edge_count}")
    
    # Connectivity
    avg_connections = edge_count / node_count if node_count > 0 else 0
    print(f"Average connections per node: {avg_connections:.1f}")
    
    # Node types summary
    node_types = {}
    for node in nodes:
        node_type = node.kind.value if hasattr(node.kind, 'value') else str(node.kind)
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print(f"Node types: {dict(node_types)}")
    
    # Edge types summary
    edge_types = {}
    for edge in edges:
        if hasattr(edge, 'kind'):
            edge_type = edge.kind.value if hasattr(edge.kind, 'value') else str(edge.kind)
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    if edge_types:
        print(f"Edge types: {dict(edge_types)}")