# chuk_ai_planner/planner/_step_tree.py
"""
Internal in-memory representation of a **plan tree**.

Why a separate module?
----------------------
* `planner.plan` (the public DSL) needs a mutable data-structure while the
  user is still building the plan.
* `planner._persist` only needs to *read* that structure in order to write
  it to a `GraphStore`.
* Keeping those details in one small file means they can evolve without
  changing public APIs.

Nothing here is exported from the `chuk_ai_planner.planner` package.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Iterator, List, Sequence


def _uid() -> str:               # helper for UUID strings
    return str(uuid.uuid4())


@dataclass
class _Step:
    """
    A single **plan step** that can contain nested sub-steps.
    """

    # ── user-supplied fields ──────────────────────────────────────
    title: str
    after: List[str]               = field(default_factory=list)

    # ── generated fields ─────────────────────────────────────────
    id: str                        = field(default_factory=_uid, init=False)
    parent: _Step | None           = field(default=None,    init=False)
    index: str                     = field(default="",      init=False)
    children: List["_Step"]        = field(default_factory=list, init=False)

    # ---------------------------------------------------------------- builder helpers
    def step(self, title: str, *, after: Sequence[str] = ()) -> "_Step":
        """Create a child and **return** it so the caller can descend."""
        child = _Step(title=title, after=list(after))
        child.parent = self
        self.children.append(child)
        return child

    def up(self) -> "_Step":
        """Return the parent (or self if already at root)."""
        return self.parent or self


# ─────────────────────── numbering helpers ──────────────────────────
def assign_indices(root: _Step) -> None:
    """
    Fill every node's ``.index`` with a *hierarchical* index
    ( ``"1"`` , ``"1.2"`` , ``"1.2.3"`` …).

    Mutates the tree in-place.
    """
    stack: list[tuple[_Step, str]] = [(root, "")]
    while stack:
        node, prefix = stack.pop()
        for pos, child in reversed(list(enumerate(node.children, 1))):
            child.index = f"{prefix}{pos}" if prefix else str(pos)
            stack.append((child, f"{child.index}."))


# ─────────────────────── traversal helpers ─────────────────────────
def iter_steps(root: _Step) -> Iterator[_Step]:
    """
    Depth-first iterator that yields **all** steps except the dummy root.

    The tree is assumed to have already been numbered ( `assign_indices()` ).
    """
    stack = list(reversed(root.children))
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))
