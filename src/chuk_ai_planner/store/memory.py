# chuk_ai_planner/store/memory.py
"""
Simple in-memory implementation of a graph store.

This provides a quick way to test and prototype with the graph model
without requiring a database.
"""

from typing import Dict, List, Optional

from chuk_ai_planner.models import GraphNode, NodeKind
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind

from .base import GraphStore


class InMemoryGraphStore(GraphStore):
    """
    Simple in-memory graph store for demonstration and testing.
    
    This implementation stores nodes and edges in memory with no persistence.
    """
    
    def __init__(self):
        """Initialize an empty store."""
        self.nodes: Dict[str, GraphNode] = {}  # id -> GraphNode
        self.edges: List[GraphEdge] = []       # list of GraphEdge
    
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the store."""
        self.nodes[node.id] = node
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def update_node(self, node: GraphNode) -> None:
        """Update a node in the store."""
        if node.id in self.nodes:
            self.nodes[node.id] = node
    
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the store."""
        self.edges.append(edge)
    
    def get_edges(
        self, 
        src: Optional[str] = None, 
        dst: Optional[str] = None,
        kind: Optional[EdgeKind] = None
    ) -> List[GraphEdge]:
        """Get edges matching the criteria."""
        result = []
        for edge in self.edges:
            if src is not None and edge.src != src:
                continue
            if dst is not None and edge.dst != dst:
                continue
            if kind is not None and edge.kind != kind:
                continue
            result.append(edge)
        return result
    
    def get_nodes_by_kind(self, kind: NodeKind) -> List[GraphNode]:
        """Get all nodes of a particular kind."""
        return [node for node in self.nodes.values() if node.kind == kind]
    
    def clear(self) -> None:
        """Clear all nodes and edges from the store."""
        self.nodes.clear()
        self.edges.clear()