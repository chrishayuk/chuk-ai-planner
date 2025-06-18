# chuk_ai_planner/__init__.py
"""
Graph-aware tools for AI agent workflows.

This package provides graph-based tools for planning, execution, and 
visualization of agent workflows, building on graph structures and 
session management systems.

Main components:
- GraphAwareToolProcessor: Executes tools based on graph structure
- UniversalPlan: High-level plan creation and management
- UniversalExecutor: Plan execution with variable resolution
- GraphStore: Interface for storing and retrieving graph elements
- Visualization tools: For rendering sessions and graphs
"""

from .processor import GraphAwareToolProcessor
from .store import GraphStore, InMemoryGraphStore

__version__ = "0.1.0"

__all__ = [
    "GraphAwareToolProcessor",
    "GraphStore", 
    "InMemoryGraphStore",
]