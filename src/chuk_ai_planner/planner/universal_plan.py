# src/chuk_ai_planner/planner/universal_plan.py
from typing import Dict, List, Any, Optional

# planner
from chuk_ai_planner.models import ToolCall
from chuk_ai_planner.models.edges import GraphEdge, EdgeKind
from chuk_ai_planner.store.base import GraphStore

# plan
from .plan import Plan as ChukPlan
from .plan_executor import PlanExecutor

class UniversalPlan(ChukPlan):
    """
    Enhanced Plan class that extends the existing Chuk Plan with universal orchestration capabilities
    """
    
    def __init__(self, title: str, description: str = None, *, graph: Optional['GraphStore'] = None, id: str = None, tags: List[str] = None):
        # Initialize the base ChukPlan
        super().__init__(title, graph=graph)
        
        # If ID was provided, use it instead of generating a new one
        if id:
            self.id = id
            
        # Additional universal plan properties
        self.description = description or f"Plan for: {title}"
        self.tags = tags or []
        self.variables = {}  # Variable dictionary for data flow
        self.metadata = {}   # Metadata for additional information
        
        # Flag to track if the plan has been registered with tools
        self._tools_registered = False
        
        # Cache for variable resolution
        self._variable_cache = {}
    
    # ---- Additional methods for universal capabilities ----
    
    def set_variable(self, name: str, value: Any) -> 'UniversalPlan':
        """Set a plan variable"""
        self.variables[name] = value
        return self
    
    def add_metadata(self, key: str, value: Any) -> 'UniversalPlan':
        """Add metadata to the plan"""
        self.metadata[key] = value
        return self
    
    def add_tag(self, tag: str) -> 'UniversalPlan':
        """Add a tag to the plan"""
        if tag not in self.tags:
            self.tags.append(tag)
        return self
    
    # ---- Enhanced step creation methods ----
    
    def add_tool_step(self, title: str, tool: str, args: Dict[str, Any] = None, 
                    depends_on: List[str] = None, result_variable: str = None) -> str:
        """
        Add a step that executes a tool and save it immediately
        Returns the step ID
        """
        # Add the step to the plan
        step_index = self.add_step(title, parent=None, after=depends_on or [])
        
        # Get the step node
        step_id = None
        for node in self._graph.nodes.values():
            if node.__class__.__name__ == "PlanStep" and node.data.get("index") == step_index:
                step_id = node.id
                break
        
        if not step_id:
            raise ValueError(f"Failed to find step node for index {step_index}")
        
        # Create a tool call node
        tool_call = ToolCall(data={"name": tool, "args": args or {}})
        self._graph.add_node(tool_call)
        
        # Link the step to the tool call
        self._graph.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step_id, dst=tool_call.id))
        
        # Store result variable information in step metadata
        if result_variable:
            # In a real implementation, we would add this to the step's metadata
            # For now, we'll store it in a custom edge
            self._graph.add_edge(GraphEdge(
                kind="RESULT_VARIABLE", 
                src=step_id, 
                dst=tool_call.id,
                data={"variable": result_variable}
            ))
        
        return step_id
    
    def add_plan_step(self, title: str, plan_id: str, args: Dict[str, Any] = None,
                     depends_on: List[str] = None, result_variable: str = None) -> str:
        """
        Add a step that executes another plan
        Returns the step ID
        """
        # Add the basic step
        step_index = self.add_step(title, parent=None, after=depends_on or [])
        
        # Get the step node
        step_id = None
        for node in self._graph.nodes.values():
            if node.__class__.__name__ == "PlanStep" and node.data.get("index") == step_index:
                step_id = node.id
                break
        
        if not step_id:
            raise ValueError(f"Failed to find step node for index {step_index}")
        
        # In a real implementation, we would add a special node for the subplan
        # For simplicity, we'll use a special tool call node with a "subplan" name
        tool_call = ToolCall(data={"name": "subplan", "args": {"plan_id": plan_id, "args": args or {}}})
        self._graph.add_node(tool_call)
        
        # Link the step to the tool call
        self._graph.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step_id, dst=tool_call.id))
        
        # Store result variable information
        if result_variable:
            self._graph.add_edge(GraphEdge(
                kind="RESULT_VARIABLE", 
                src=step_id, 
                dst=tool_call.id,
                data={"variable": result_variable}
            ))
        
        return step_id
    
    def add_function_step(self, title: str, function: str, args: Dict[str, Any] = None,
                        depends_on: List[str] = None, result_variable: str = None) -> str:
        """
        Add a step that executes a function
        Returns the step ID
        """
        # Add the basic step
        step_index = self.add_step(title, parent=None, after=depends_on or [])
        
        # Get the step node
        step_id = None
        for node in self._graph.nodes.values():
            if node.__class__.__name__ == "PlanStep" and node.data.get("index") == step_index:
                step_id = node.id
                break
        
        if not step_id:
            raise ValueError(f"Failed to find step node for index {step_index}")
        
        # Create a special tool call for functions
        tool_call = ToolCall(data={"name": "function", "args": {"function": function, "args": args or {}}})
        self._graph.add_node(tool_call)
        
        # Link the step to the tool call
        self._graph.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step_id, dst=tool_call.id))
        
        # Store result variable information
        if result_variable:
            self._graph.add_edge(GraphEdge(
                kind="RESULT_VARIABLE", 
                src=step_id, 
                dst=tool_call.id,
                data={"variable": result_variable}
            ))
        
        return step_id
    
    # ---- Convenience methods for the fluent interface ----
    
    def tool(self, title: str, tool: str, args: Dict[str, Any] = None, 
            result_variable: str = None) -> 'UniversalPlan':
        """Add a tool step and remain at the same level (fluent interface)"""
        # Current level
        current = self._cursor
        
        # Add the step
        self.step(title)
        
        # Create a tool call node
        if self._indexed:  # If the plan has been indexed
            # Find the current step's ID
            step_id = None
            for node in self._graph.nodes.values():
                if (node.__class__.__name__ == "PlanStep" and 
                    node.data.get("title") == title and
                    node.data.get("index").startswith(str(len(current.children)))):
                    step_id = node.id
                    break
            
            if step_id:
                # Create and link the tool call
                tool_call = ToolCall(data={"name": tool, "args": args or {}})
                self._graph.add_node(tool_call)
                self._graph.add_edge(GraphEdge(kind=EdgeKind.PLAN_LINK, src=step_id, dst=tool_call.id))
                
                # Store result variable information
                if result_variable:
                    self._graph.add_edge(GraphEdge(
                        kind="RESULT_VARIABLE", 
                        src=step_id, 
                        dst=tool_call.id,
                        data={"variable": result_variable}
                    ))
        
        # Go back up
        self.up()
        return self
    
    def subplan(self, title: str, plan_id: str, args: Dict[str, Any] = None,
               result_variable: str = None) -> 'UniversalPlan':
        """Add a subplan step and remain at the same level (fluent interface)"""
        # Add the step
        self.step(title)
        
        # Create a subplan call (similar to tool method)
        # Implementation details would be similar to the tool method
        
        # Go back up
        self.up()
        return self
    
    def function(self, title: str, function: str, args: Dict[str, Any] = None,
                result_variable: str = None) -> 'UniversalPlan':
        """Add a function step and remain at the same level (fluent interface)"""
        # Add the step
        self.step(title)
        
        # Create a function call (similar to tool method)
        # Implementation details would be similar to the tool method
        
        # Go back up
        self.up()
        return self
    
    # ---- Additional methods for plan inspection ----
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the plan to a dictionary representation"""
        # Save the plan if not already saved
        if not self._indexed:
            self.save()
        
        # Basic plan info
        result = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "variables": self.variables,
            "metadata": self.metadata
        }
        
        # Add steps
        result["steps"] = []
        for node in self._graph.nodes.values():
            if node.__class__.__name__ == "PlanStep":
                # Find tool calls linked to this step
                tool_calls = []
                for edge in self._graph.get_edges(src=node.id, kind=EdgeKind.PLAN_LINK):
                    tool_node = self._graph.get_node(edge.dst)
                    if tool_node and tool_node.__class__.__name__ == "ToolCall":
                        tool_calls.append({
                            "id": tool_node.id,
                            "name": tool_node.data.get("name"),
                            "args": tool_node.data.get("args", {})
                        })
                
                # Find result variable
                result_variable = None
                for edge in self._graph.get_edges(src=node.id, kind="RESULT_VARIABLE"):
                    result_variable = edge.data.get("variable")
                    break
                
                # Add step info
                result["steps"].append({
                    "id": node.id,
                    "index": node.data.get("index"),
                    "title": node.data.get("title"),
                    "tool_calls": tool_calls,
                    "result_variable": result_variable
                })
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], graph: Optional['GraphStore'] = None) -> 'UniversalPlan':
        """Create a UniversalPlan from a dictionary representation"""
        # Create a new plan
        plan = cls(
            title=data.get("title", "Untitled Plan"),
            description=data.get("description"),
            id=data.get("id"),
            tags=data.get("tags", []),
            graph=graph
        )
        
        # Set variables and metadata
        plan.variables = data.get("variables", {})
        plan.metadata = data.get("metadata", {})
        
        # Add steps
        # This is complex and would need careful implementation to recreate the graph structure
        # For this example, we'll leave it as a placeholder
        
        return plan
    
    # ---- Integration with ChukPlan's original methods ----
    
    # Override save to add variable and metadata handling
    def save(self) -> str:
        """Persist the plan to the graph store with enhanced metadata"""
        plan_id = super().save()
        
        # In a real implementation, we would add the variables and metadata
        # to the PlanNode in the graph store
        
        return plan_id