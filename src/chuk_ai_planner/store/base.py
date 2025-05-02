# src/chuk_ai_planner/store/base.py
"""
Abstract base class for graph stores.

This defines the interface that all graph store implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from chuk_ai_planner.models import GraphNode, NodeKind
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind


class GraphStore(ABC):
    """
    Abstract base class for graph stores.
    
    Implementations should provide storage and retrieval of graph nodes and edges.
    """
    
    @abstractmethod
    def add_node(self, node: GraphNode) -> None:
        """
        Add a node to the store.
        
        Parameters
        ----------
        node : GraphNode
            The node to add
        """
        pass
    
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """
        Get a node by ID.
        
        Parameters
        ----------
        node_id : str
            The ID of the node to retrieve
            
        Returns
        -------
        Optional[GraphNode]
            The node, or None if not found
        """
        pass
    
    @abstractmethod
    def update_node(self, node: GraphNode) -> None:
        """
        Update a node in the store.
        
        Parameters
        ----------
        node : GraphNode
            The node to update (must have the same ID)
        """
        pass
    
    @abstractmethod
    def add_edge(self, edge: GraphEdge) -> None:
        """
        Add an edge to the store.
        
        Parameters
        ----------
        edge : GraphEdge
            The edge to add
        """
        pass
    
    @abstractmethod
    def get_edges(
        self, 
        src: Optional[str] = None, 
        dst: Optional[str] = None,
        kind: Optional[EdgeKind] = None
    ) -> List[GraphEdge]:
        """
        Get edges matching the given criteria.
        
        Parameters
        ----------
        src : Optional[str]
            Filter by source node ID
        dst : Optional[str]
            Filter by destination node ID
        kind : Optional[EdgeKind]
            Filter by edge kind
            
        Returns
        -------
        List[GraphEdge]
            List of edges matching the criteria
        """
        pass
    
    def get_nodes_by_kind(self, kind: NodeKind) -> List[GraphNode]:
        """
        Get all nodes of a particular kind.
        
        Parameters
        ----------
        kind : NodeKind
            The kind of nodes to retrieve
            
        Returns
        -------
        List[GraphNode]
            List of nodes of the specified kind
        """
        raise NotImplementedError("Subclasses must implement")