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
        model="gpt-5-mini",
    )
    plan_dict = await agent.plan("user request")
"""

from __future__ import annotations
import json
import textwrap
from typing import Any, Callable, Dict, List, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()  # makes OPENAI_API_KEY available

__all__ = ["PlanAgent"]

_Validate = Callable[[Dict[str, Any]], Tuple[bool, str]]


class PlanAgent:
    """Loop-until-valid plan generator with a transparent `history` log."""

    def __init__(
        self,
        *,
        system_prompt: str,
        validate_step: _Validate,
        model: str = "gpt-5-mini",
        temperature: float = 1.0,
        max_retries: int = 3,
    ):
        self.system_prompt = textwrap.dedent(system_prompt).strip()
        self.validate_step = validate_step
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.history: List[Dict[str, Any]] = []
        self._client = AsyncOpenAI()

    # ---------------------------------------------------------------- private
    async def _chat(self, messages: List[Dict[str, str]]) -> str:
        rsp = await self._client.chat.completions.create(  # type: ignore[call-overload]
            model=self.model,
            temperature=self.temperature,
            messages=messages,  # type: ignore[arg-type]
            response_format={"type": "json_object"},  # Force JSON output
        )
        content = rsp.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned no content")
        return content

    # ---------------------------------------------------------------- public
    async def plan(self, user_prompt: str) -> Dict[str, Any]:
        """Return a syntactically *and* semantically valid JSON plan."""
        prompt = user_prompt

        for attempt in range(1, self.max_retries + 1):
            raw = await self._chat(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]
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
                    for ok, msg in (
                        self.validate_step(s) for s in plan.get("steps", [])
                    )
                    if not ok
                ]
                record["errors"] = errors
                self.history.append(record)
                if not errors:
                    return plan  # ✓

            # prepare corrective message
            error_list = record["errors"]
            if not isinstance(error_list, list):
                error_list = [str(error_list)]
            prompt = (
                "Your previous JSON was invalid:\n"
                + "\n".join(f"- {e}" for e in error_list)
                + "\nPlease return a *complete* corrected JSON plan."
            )

        raise RuntimeError(
            "GPT never produced a valid plan. Debug trace:\n"
            + json.dumps(self.history, indent=2)[:1500]
        )
