# chuk_ai_planner/store/memory.py
"""
Simple in-memory implementation of a graph store.

This provides a quick way to test and prototype with the graph model
without requiring a database.
"""

from typing import Dict, List, Optional

from chuk_ai_planner.core.graph import GraphNode, GraphEdge
from chuk_ai_planner.core.graph.types import NodeType, EdgeType

from .base import GraphStore


class InMemoryGraphStore(GraphStore):
    """
    Async-native in-memory graph store for demonstration and testing.

    This implementation stores nodes and edges in memory with no persistence.
    """

    def __init__(self) -> None:
        """Initialize an empty store."""
        self.nodes: Dict[str, GraphNode] = {}  # id -> GraphNode
        self.edges: List[GraphEdge] = []  # list of GraphEdge

    async def add_node(self, node: GraphNode) -> None:
        """Add a node to the store."""
        self.nodes[node.id] = node

    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    async def update_node(self, node: GraphNode) -> None:
        """Update a node in the store."""
        if node.id in self.nodes:
            self.nodes[node.id] = node

    async def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the store."""
        self.edges.append(edge)

    async def get_edges(
        self,
        src: Optional[str] = None,
        dst: Optional[str] = None,
        kind: Optional[EdgeType] = None,
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

    async def get_nodes_by_kind(self, kind: NodeType) -> List[GraphNode]:
        """Get all nodes of a particular kind."""
        return [node for node in self.nodes.values() if node.kind == kind]

    async def list_nodes(self, kind: Optional[str] = None) -> List[GraphNode]:
        """List nodes with optional kind filter."""
        if kind is None:
            return list(self.nodes.values())
        return [node for node in self.nodes.values() if node.kind == kind]

    async def clear(self) -> None:
        """Clear all nodes and edges from the store (async-native!)."""
        self.nodes.clear()
        self.edges.clear()
