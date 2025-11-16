# chuk_ai_planner/agents/graph_plan_agent.py
from __future__ import annotations
from typing import Any, Dict

from chuk_ai_planner.core.planner import Plan
from chuk_ai_planner.core.store.base import GraphStore
from chuk_ai_planner.core.store.memory import InMemoryGraphStore

from .plan_agent import PlanAgent, _Validate  # ← your existing file

__all__ = ["GraphPlanAgent"]


class GraphPlanAgent(PlanAgent):
    """
    Thin wrapper around PlanAgent that ALSO:

      • materialises a Plan DSL tree
      • saves it to the supplied GraphStore
      • gives you (Plan, plan_id, graph) back
    """

    def __init__(
        self,
        *,
        graph: GraphStore | None,
        system_prompt: str,
        validate_step: _Validate,
        model: str = "gpt-5-mini",
        temperature: float = 1.0,
        max_retries: int = 3,
    ):
        super().__init__(
            system_prompt=system_prompt,
            validate_step=validate_step,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
        )
        self._graph = graph or InMemoryGraphStore()

    # ------------------------------------------- public convenience API
    async def plan_into_graph(self, user_prompt: str) -> tuple[Plan, str, GraphStore]:
        """
        • gets a *valid* JSON plan from the LLM (inherited logic)
        • builds & saves a `Plan` object
        • returns (plan_obj, plan_node_id, graph_store)
        """
        json_plan: Dict[str, Any] = await super().plan(user_prompt)

        # 1 — build the DSL tree
        plan = Plan(json_plan["title"], graph=self._graph)
        for step in json_plan["steps"]:
            depends = [str(i) for i in step.get("depends_on", [])]
            plan.step(step["title"], after=depends).up()
        plan_node_id = await plan.save()

        # 2 — attach ToolCall placeholders (async-native, Pydantic-native)
        from chuk_ai_planner.core.graph import ToolCall, PlanStep, EdgeType, GraphEdge

        # Build index-to-id map (only for InMemoryGraphStore)
        idx2id = {}
        if hasattr(self._graph, "nodes"):
            idx2id = {
                n.index: n.id
                for n in self._graph.nodes.values()  # type: ignore
                if isinstance(n, PlanStep)
            }

        for idx, step in enumerate(json_plan["steps"], 1):
            # Create ToolCall with proper typed fields (no dict goop!)
            tc = ToolCall(
                name=step["tool"],
                args=step.get("args", {}),
            )
            await self._graph.add_node(tc)
            await self._graph.add_edge(
                GraphEdge(
                    kind=EdgeType.PLAN_LINK,
                    src=idx2id[str(idx)],
                    dst=tc.id,
                )
            )

        return plan, plan_node_id, self._graph
