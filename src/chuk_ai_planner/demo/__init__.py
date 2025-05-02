# chuk_ai_planner/demo/__init__.py
"""
Demo helpers for chuk_ai_planner

• simulate_llm_call – tiny fake-LLM utility used by the demo scripts
"""

from __future__ import annotations

# keep the simulator (still useful for demos / tests)
from .llm_simulator import simulate_llm_call

__all__: list[str] = ["simulate_llm_call"]
