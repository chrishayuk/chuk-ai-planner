# chuk_ai_planner/planner/_ids.py
"""
planner._ids
============

Tiny helper that gives every Plan / Step / ToolCall a stable UUID.

Only three public names are exported so that external code can do:

    from chuk_ai_planner.planner._ids import new_uuid
"""

from __future__ import annotations
import uuid

__all__ = ["new_uuid", "new_plan_id", "new_step_id"]


def _uid() -> str:
    """Return a random UUID *string* (no dashes shortened is fine)."""
    return str(uuid.uuid4())


# canonical helper – *always* prefer this name inside the package
def new_uuid() -> str:
    return _uid()


# backward-compat aliases
new_plan_id = new_uuid
new_step_id = new_uuid
