# chuk_ai_planner/agents/plan_agemt.py
"""
chuk_ai_planner.agents.plan_agent
==========================

Mini-agent that keeps asking GPT for a JSON plan until the plan
conforms to an application-supplied validator.

Typical usage
-------------
    from chuk_ai_planner.agent.plan_agent import PlanAgent

    async def my_validator(step: dict[str, any]) -> tuple[bool, str]:
        ...

    agent = PlanAgent(
        system_prompt=SYS_MSG,
        validate_step=my_validator,
        model="gpt-4o-mini",
    )
    plan_dict = await agent.plan("user request")
"""

from __future__ import annotations
import json, textwrap
from typing import Any, Callable, Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()  # makes OPENAI_API_KEY available

from openai import AsyncOpenAI

__all__ = ["PlanAgent"]

_Validate = Callable[[Dict[str, Any]], Tuple[bool, str]]

class PlanAgent:
    """Loop-until-valid plan generator with a transparent `history` log."""

    def __init__(
        self,
        *,
        system_prompt: str,
        validate_step: _Validate,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_retries: int = 3,
    ):
        self.system_prompt   = textwrap.dedent(system_prompt).strip()
        self.validate_step   = validate_step
        self.model           = model
        self.temperature     = temperature
        self.max_retries     = max_retries
        self.history: List[Dict[str, Any]] = []
        self._client = AsyncOpenAI()

    # ---------------------------------------------------------------- private
    async def _chat(self, messages: List[Dict[str, str]]) -> str:
        rsp = await self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )
        return rsp.choices[0].message.content

    # ---------------------------------------------------------------- public
    async def plan(self, user_prompt: str) -> Dict[str, Any]:
        """Return a syntactically *and* semantically valid JSON plan."""
        prompt = user_prompt

        for attempt in range(1, self.max_retries + 1):
            raw = await self._chat(
                [{"role": "system", "content": self.system_prompt},
                 {"role": "user",   "content": prompt}]
            )

            record = {"attempt": attempt, "raw": raw}
            try:
                plan = json.loads(raw)
            except json.JSONDecodeError:
                record["errors"] = ["response is not valid JSON"]
                self.history.append(record)
            else:
                errors = [
                    msg
                    for ok, msg in (self.validate_step(s) for s in plan.get("steps", []))
                    if not ok
                ]
                record["errors"] = errors
                self.history.append(record)
                if not errors:
                    return plan  # ✓

            # prepare corrective message
            prompt = (
                "Your previous JSON was invalid:\n"
                + "\n".join(f"- {e}" for e in record["errors"])
                + "\nPlease return a *complete* corrected JSON plan."
            )

        raise RuntimeError(
            "GPT never produced a valid plan. Debug trace:\n"
            + json.dumps(self.history, indent=2)[:1500]
        )
