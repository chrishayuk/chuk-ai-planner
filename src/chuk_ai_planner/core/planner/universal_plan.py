# chuk_ai_planner/planner/universal_plan.py
from typing import Dict, List, Any, Optional

# planner
from chuk_ai_planner.core.graph import ToolCall, CustomEdge, PlanLinkEdge, PlanStep
from chuk_ai_planner.core.graph import EdgeType, NodeType
from chuk_ai_planner.core.store.base import GraphStore

# plan
from .plan import Plan as ChukPlan
from ._step_tree import iter_steps  # Import the iter_steps function

# Constants for special tool types and edge types
TOOL_TYPE_SUBPLAN = "subplan"
TOOL_TYPE_FUNCTION = "function"
EDGE_TYPE_RESULT_VARIABLE = "result_variable"

# Constants for dictionary keys
KEY_VARIABLE = "variable"
KEY_ARGS = "args"
KEY_PLAN_ID = "plan_id"
KEY_FUNCTION = "function"
KEY_ID = "id"
KEY_TITLE = "title"
KEY_DESCRIPTION = "description"
KEY_TAGS = "tags"
KEY_VARIABLES = "variables"
KEY_METADATA = "metadata"
KEY_STEPS = "steps"
KEY_INDEX = "index"
KEY_TOOL_CALLS = "tool_calls"
KEY_RESULT_VARIABLE = "result_variable"
KEY_NAME = "name"


class UniversalPlan(ChukPlan):
    """
    Enhanced Plan class that extends the existing Chuk Plan with universal orchestration capabilities
    """

    def __init__(
        self,
        title: str,
        description: Optional[str] = None,
        *,
        graph: Optional["GraphStore"] = None,
        id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        # Initialize the base ChukPlan
        super().__init__(title, graph=graph)

        # If ID was provided, use it instead of generating a new one
        if id:
            self.id = id

        # Additional universal plan properties
        self.description = description or f"Plan for: {title}"
        self.tags = tags or []
        self.variables: Dict[str, Any] = {}  # Variable dictionary for data flow
        self.metadata: Dict[str, Any] = {}  # Metadata for additional information

        # Flag to track if the plan has been registered with tools
        self._tools_registered = False

        # Cache for variable resolution
        self._variable_cache: Dict[str, Any] = {}

    # ---- Helper methods ----

    async def _find_step_by_index(self, step_index: str) -> Optional[str]:
        """
        Find a PlanStep node ID by its index using proper GraphStore interface.
        Returns the step ID or None if not found.
        """
        # Use proper GraphStore interface instead of accessing .nodes directly
        steps = await self._graph.get_nodes_by_kind(NodeType.PLAN_STEP)
        for node in steps:
            if isinstance(node, PlanStep) and node.index == step_index:
                return node.id
        return None

    # ---- Additional methods for universal capabilities ----

    def set_variable(self, name: str, value: Any) -> "UniversalPlan":
        """Set a plan variable"""
        self.variables[name] = value
        return self

    def add_metadata(self, key: str, value: Any) -> "UniversalPlan":
        """Add metadata to the plan"""
        self.metadata[key] = value
        return self

    def add_tag(self, tag: str) -> "UniversalPlan":
        """Add a tag to the plan"""
        if tag not in self.tags:
            self.tags.append(tag)
        return self

    # ---- Enhanced step creation methods ----

    async def add_tool_step(
        self,
        title: str,
        tool: str,
        args: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
        result_variable: Optional[str] = None,
    ) -> str:
        """
        Add a step that executes a tool and save it immediately
        Returns the step ID
        """
        # Add the step to the plan
        step_index = await self.add_step(title, parent=None, after=depends_on or [])

        # Get the step node using proper GraphStore interface
        step_id = await self._find_step_by_index(step_index)
        if not step_id:
            raise ValueError(f"Failed to find step node for index {step_index}")

        # Create a tool call node (Pydantic-native!)
        tool_call = ToolCall(name=tool, args=args or {})
        await self._graph.add_node(tool_call)

        # Link the step to the tool call
        await self._graph.add_edge(PlanLinkEdge(src=step_id, dst=tool_call.id))

        # Store result variable information in step metadata
        if result_variable:
            # Store in a custom edge
            await self._graph.add_edge(
                CustomEdge(
                    src=step_id,
                    dst=tool_call.id,
                    custom_type=EDGE_TYPE_RESULT_VARIABLE,
                    metadata={KEY_VARIABLE: result_variable},
                )
            )

        return step_id

    async def add_plan_step(
        self,
        title: str,
        plan_id: str,
        args: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
        result_variable: Optional[str] = None,
    ) -> str:
        """
        Add a step that executes another plan
        Returns the step ID
        """
        # Add the basic step
        step_index = await self.add_step(title, parent=None, after=depends_on or [])

        # Get the step node using proper GraphStore interface
        step_id = await self._find_step_by_index(step_index)
        if not step_id:
            raise ValueError(f"Failed to find step node for index {step_index}")

        # In a real implementation, we would add a special node for the subplan
        # For simplicity, we'll use a special tool call node with the TOOL_TYPE_SUBPLAN name
        tool_call = ToolCall(
            name=TOOL_TYPE_SUBPLAN, args={KEY_PLAN_ID: plan_id, KEY_ARGS: args or {}}
        )
        await self._graph.add_node(tool_call)

        # Link the step to the tool call
        await self._graph.add_edge(PlanLinkEdge(src=step_id, dst=tool_call.id))

        # Store result variable information
        if result_variable:
            await self._graph.add_edge(
                CustomEdge(
                    src=step_id,
                    dst=tool_call.id,
                    custom_type=EDGE_TYPE_RESULT_VARIABLE,
                    metadata={KEY_VARIABLE: result_variable},
                )
            )

        return step_id

    async def add_function_step(
        self,
        title: str,
        function: str,
        args: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
        result_variable: Optional[str] = None,
    ) -> str:
        """
        Add a step that executes a function
        Returns the step ID

        In CTP-first architecture, functions are registered as tools,
        so we treat function steps the same as tool steps.
        """
        # Add the basic step
        step_index = await self.add_step(title, parent=None, after=depends_on or [])

        # Get the step node using proper GraphStore interface
        step_id = await self._find_step_by_index(step_index)
        if not step_id:
            raise ValueError(f"Failed to find step node for index {step_index}")

        # CTP-first: Create a tool call with the function name directly
        # Functions are registered as tools via register_function(), so they're
        # executed the same way as tools
        tool_call = ToolCall(
            name=function,  # Use function name directly (not "function")
            args=args or {},
        )
        await self._graph.add_node(tool_call)

        # Link the step to the tool call
        await self._graph.add_edge(PlanLinkEdge(src=step_id, dst=tool_call.id))

        # Store result variable information
        if result_variable:
            await self._graph.add_edge(
                CustomEdge(
                    src=step_id,
                    dst=tool_call.id,
                    custom_type=EDGE_TYPE_RESULT_VARIABLE,
                    metadata={KEY_VARIABLE: result_variable},
                )
            )

        return step_id

    # ---- Convenience methods for the fluent interface ----

    # ---- Additional methods for plan inspection ----

    async def to_dict(self) -> Dict[str, Any]:
        """Convert the plan to a dictionary representation"""
        # Save the plan if not already saved
        if not self._indexed:
            await self.save()

        # Basic plan info
        result = {
            KEY_ID: self.id,
            KEY_TITLE: self.title,
            KEY_DESCRIPTION: self.description,
            KEY_TAGS: self.tags,
            KEY_VARIABLES: self.variables,
            KEY_METADATA: self.metadata,
        }

        # Add steps
        result[KEY_STEPS] = []
        # Use proper GraphStore interface
        all_steps = await self._graph.get_nodes_by_kind(NodeType.PLAN_STEP)
        for node in all_steps:
            if isinstance(node, PlanStep):
                # Find tool calls linked to this step
                tool_calls = []
                for edge in await self._graph.get_edges(
                    src=node.id, kind=EdgeType.PLAN_LINK
                ):
                    tool_node = await self._graph.get_node(edge.dst)
                    # Type-safe access to ToolCall
                    if tool_node and isinstance(tool_node, ToolCall):
                        tool_calls.append(
                            {
                                KEY_ID: tool_node.id,
                                KEY_NAME: tool_node.name,
                                KEY_ARGS: tool_node.args,
                            }
                        )

                # Find result variable
                result_variable = None
                from chuk_ai_planner.core.graph import CustomEdge

                for edge in await self._graph.get_edges(
                    src=node.id, kind=EdgeType.CUSTOM
                ):
                    # Type-safe access to CustomEdge
                    if (
                        isinstance(edge, CustomEdge)
                        and edge.custom_type == EDGE_TYPE_RESULT_VARIABLE
                    ):
                        result_variable = edge.metadata.get(KEY_VARIABLE)
                        break

                # Get step title - prioritize the 'description' field
                title = node.description

                # Add step info
                steps_list = result.get(KEY_STEPS)
                if isinstance(steps_list, list):
                    steps_list.append(
                        {
                            KEY_ID: node.id,
                            KEY_INDEX: node.index,
                            KEY_TITLE: title,
                            KEY_TOOL_CALLS: tool_calls,
                            KEY_RESULT_VARIABLE: result_variable,
                        }
                    )

        return result

    @classmethod
    async def from_dict(
        cls, data: Dict[str, Any], graph: Optional["GraphStore"] = None
    ) -> "UniversalPlan":
        """Create a UniversalPlan from a dictionary representation"""
        # Create a new plan
        plan = cls(
            title=data.get(KEY_TITLE, "Untitled Plan"),
            description=data.get(KEY_DESCRIPTION),
            id=data.get(KEY_ID),
            tags=data.get(KEY_TAGS, []),
            graph=graph,
        )

        # Set variables and metadata
        plan.variables = data.get(KEY_VARIABLES, {})
        plan.metadata = data.get(KEY_METADATA, {})

        # Add steps from the dictionary
        if KEY_STEPS in data:
            for step_data in data[KEY_STEPS]:
                # Get step info
                index = step_data.get(KEY_INDEX)
                title = step_data.get(KEY_TITLE)

                # Find or recreate the step based on index using proper GraphStore interface
                step_node = None
                all_steps = await plan._graph.get_nodes_by_kind(NodeType.PLAN_STEP)
                for node in all_steps:
                    if isinstance(node, PlanStep) and node.index == index:
                        step_node = node
                        break

                if not step_node:
                    # Add a new step (must await!)
                    step_index = await plan.add_step(title, parent=None)

                    # Find the created step using proper interface
                    all_steps = await plan._graph.get_nodes_by_kind(NodeType.PLAN_STEP)
                    for node in all_steps:
                        if isinstance(node, PlanStep) and node.index == step_index:
                            step_node = node
                            break

                if step_node and KEY_TOOL_CALLS in step_data:
                    # Add tool calls
                    for tool_call_data in step_data[KEY_TOOL_CALLS]:
                        tool_name = tool_call_data.get(KEY_NAME)
                        tool_args = tool_call_data.get(KEY_ARGS, {})

                        # Create tool call
                        tool_call = ToolCall(name=tool_name, args=tool_args)
                        await plan._graph.add_node(tool_call)

                        # Link to step
                        await plan._graph.add_edge(
                            PlanLinkEdge(src=step_node.id, dst=tool_call.id)
                        )

                        # Add result variable if present
                        if step_data.get(KEY_RESULT_VARIABLE):
                            await plan._graph.add_edge(
                                CustomEdge(
                                    src=step_node.id,
                                    dst=tool_call.id,
                                    custom_type=EDGE_TYPE_RESULT_VARIABLE,
                                    metadata={
                                        KEY_VARIABLE: step_data[KEY_RESULT_VARIABLE]
                                    },
                                )
                            )

        return plan

    # ---- Integration with ChukPlan's original methods ----

    # Be careful not to try to modify the data directly
    # We'll need to create new nodes if we want to update them

    # Override outline method to handle None titles
    def outline(self) -> str:
        """Return a plain-text outline (helpful for humans or LLMs)."""
        if not self._indexed:
            self._number_steps()

        lines: List[str] = [f"Plan: {self.title}   (id: {self.id[:8]})"]
        for st in iter_steps(self._root):
            deps = f"  depends on {st.after}" if st.after else ""
            # Use description field if available, otherwise title
            title = st.title or getattr(st, "description", None) or "Untitled Step"
            lines.append(f"  {st.index:<6} {title:<35} (step_id: {st.id[:8]}){deps}")
        return "\n".join(lines)
