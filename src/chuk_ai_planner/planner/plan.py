# chuk_ai_planner/planner/plan.py
from __future__ import annotations

"""
chuk_ai_planner.planner.plan
======================

Author-facing DSL for composing a plan **before** it is persisted to a
`GraphStore`.

Components are split up to keep the public API small and the internal
pieces focused:

* `_step_tree.py` - the in-memory tree of `_Step` nodes and helpers
* `_ids.py`       - tiny UUID helper
* `_persist.py`   - functions that turn a fully-built plan or a single
                    late-added step into graph nodes / edges

Only the `Plan` class below is re-exported by `chuk_ai_planner.planner`.
"""

from typing import Dict, List, Sequence

from chuk_ai_planner.store.base import GraphStore
from chuk_ai_planner.store.memory import InMemoryGraphStore

from ._ids import new_plan_id
from ._persist import persist_full_plan, persist_single_step
from ._step_tree import _Step, assign_indices, iter_steps

__all__ = ["Plan"]


class Plan:
    """A mutable hierarchy of plan-steps."""

    # ------------------------------------------------------------------ init
    def __init__(self, title: str, *, graph: GraphStore | None = None):
        self.title: str = title
        self.id: str = new_plan_id()
        self._graph: GraphStore = graph or InMemoryGraphStore()

        self._root: _Step = _Step("[ROOT]")
        self._cursor: _Step = self._root
        self._indexed: bool = False              # lazy numbering
        self._by_index: Dict[str, _Step] = {}    # "1.2" -> _Step

    # ---------------------------------------------------------------- builder
    def step(self, title: str, *, after: Sequence[str] = ()) -> "Plan":
        """Add a **child** step and descend into it (fluent style)."""
        self._cursor = self._cursor.step(title, after=after)
        self._indexed = False
        return self

    def up(self) -> "Plan":
        """Move the cursor one level **up** (root-safe)."""
        self._cursor = self._cursor.up()
        return self

    # ------------------------------------------------------ runtime addition
    def add_step(
        self,
        title: str,
        *,
        parent: str | None = None,
        after: Sequence[str] = (),
    ) -> str:
        """
        Insert a new step **after** the plan has been saved and persist it
        immediately.

        Returns the hierarchical index assigned to the new step (e.g. "1.4").
        """
        if not self._indexed:
            self._number_steps()

        parent_step = self._by_index.get(parent) if parent else self._root
        if parent and not parent_step:
            raise ValueError(f"Parent index {parent!r} does not exist")

        new_step = parent_step.step(title, after=after)
        new_idx_num = len(parent_step.children)
        new_step.index = (
            f"{parent_step.index}.{new_idx_num}"
            if parent_step.index else str(new_idx_num)
        )
        self._by_index[new_step.index] = new_step

        # ── persist the single new step (fixed signature) ────────────
        persist_single_step(
            new_step,           # the *_Step* object
            parent_step,        # its parent
            self._graph,        # target graph store
            self.id,            # id of the PlanNode
        )
        return new_step.index

    # ---------------------------------------------------------------- numbering
    def _number_steps(self) -> None:
        assign_indices(self._root)
        self._by_index = {st.index: st for st in iter_steps(self._root)}
        self._indexed = True

    # ---------------------------------------------------------------- helpers
    def outline(self) -> str:
        """Return a plain-text outline (helpful for humans or LLMs)."""
        if not self._indexed:
            self._number_steps()

        lines: List[str] = [f"Plan: {self.title}   (id: {self.id[:8]})"]
        for st in iter_steps(self._root):
            deps = f"  depends on {st.after}" if st.after else ""
            lines.append(
                f"  {st.index:<6} {st.title:<35} (step_id: {st.id[:8]}){deps}"
            )
        return "\n".join(lines)

    # ---------------------------------------------------------------- save
    def save(self) -> str:
        """Persist the **entire** plan tree to the associated graph store."""
        if not self._indexed:
            self._number_steps()

        persist_full_plan(self, self._by_index, self._graph)
        return self.id

    # ---------------------------------------------------------------- misc
    @property
    def graph(self) -> GraphStore:  # read-only accessor
        return self._graph
