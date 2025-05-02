# tests/models/test_edges.py
import pytest
import re

from chuk_ai_planner.models.edges import (
    ParentChildEdge,
    NextEdge,
    PlanEdge,
    StepEdge,
    EdgeKind,
    GraphEdge,
)

UUID_V4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.I,
)


@pytest.mark.parametrize(
    "cls, expected_kind",
    [
        (ParentChildEdge, EdgeKind.PARENT_CHILD),
        (NextEdge, EdgeKind.NEXT),
        (PlanEdge, EdgeKind.PLAN_LINK),
        (StepEdge, EdgeKind.STEP_ORDER),
    ],
)
def test_edge_defaults(cls, expected_kind):
    edge: GraphEdge = cls(src="A", dst="B")  # type: ignore[call-arg]

    assert edge.kind == expected_kind
    assert UUID_V4_RE.match(edge.id)
    assert edge.src == "A" and edge.dst == "B"
    assert edge.data == {}


def test_edge_immutable():
    e = NextEdge(src="n1", dst="n2")
    with pytest.raises(TypeError):
        e.src = "XXX"  # frozen


def test_edge_repr():
    e = ParentChildEdge(src="abcdef00", dst="deadbeef")
    text = repr(e)
    # Shape: <parent_child:abcdef→deadbe>
    assert text.startswith(f"<{e.kind.value}:")
    assert "abcdef" in text and "deadbe" in text and text.endswith(">")
