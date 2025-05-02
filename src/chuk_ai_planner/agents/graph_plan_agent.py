# chuk_ai_planner/agents/graph_plan_agent.py
from __future__ import annotations
import asyncio, json, textwrap
from typing import Any, Callable, Dict, List, Tuple

from chuk_ai_planner.planner import Plan
from chuk_ai_planner.store.base import GraphStore
from chuk_ai_planner.store.memory import InMemoryGraphStore

from .plan_agent import PlanAgent, _Validate   # ← your existing file

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
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
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
    async def plan_into_graph(
        self, user_prompt: str
    ) -> tuple[Plan, str, GraphStore]:
        """
        • gets a *valid* JSON plan from the LLM (inherited logic)  
        • builds & saves a `Plan` object  
        • returns (plan_obj, plan_node_id, graph_store)
        """
        json_plan: Dict[str, Any] = await super().plan(user_prompt)

        # 1 — build the DSL tree
        plan = Plan(json_plan["title"], graph=self._graph)
        for step in json_plan["steps"]:
            depends = [str(i) for i in step.get("depends_on", [])]
            plan.step(step["title"], after=depends).up()
        plan_node_id = plan.save()

        # 2 — attach *empty* ToolCall placeholders
        #    (real args can be wired later by your processor)
        from chuk_ai_planner.models import ToolCall
        from chuk_ai_planner.models.edges import EdgeKind, GraphEdge

        idx2id = {n.data["index"]: n.id
                  for n in self._graph.nodes.values()
                  if n.__class__.__name__ == "PlanStep"}

        for idx, step in enumerate(json_plan["steps"], 1):
            tc = ToolCall(
                data={
                    "name": step["tool"],
                    "args": step.get("args", {}),
                }
            )
            self._graph.add_node(tc)
            self._graph.add_edge(
                GraphEdge(
                    kind=EdgeKind.PLAN_LINK,
                    src=idx2id[str(idx)],
                    dst=tc.id,
                )
            )

        return plan, plan_node_id, self._graph
