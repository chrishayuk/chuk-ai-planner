# tests/models/test_nodes.py
import re
from datetime import timezone
import pytest

from chuk_ai_planner.models import (
    NodeKind,
    GraphNode,
    SessionNode,
    PlanNode,
    PlanStep,
    UserMessage,
    AssistantMessage,
    ToolCall,
    TaskRun,
    Summary,
)

UUID_V4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.I,
)


@pytest.mark.parametrize(
    "cls, expected_kind",
    [
        (SessionNode, NodeKind.SESSION),
        (PlanNode, NodeKind.PLAN),
        (PlanStep, NodeKind.PLAN_STEP),
        (UserMessage, NodeKind.USER_MSG),
        (AssistantMessage, NodeKind.ASSIST_MSG),
        (ToolCall, NodeKind.TOOL_CALL),
        (TaskRun, NodeKind.TASK_RUN),
        (Summary, NodeKind.SUMMARY),
    ],
)
def test_defaults_and_enum(cls, expected_kind):
    """A freshly-constructed node should have:
       * a UUIDv4 id,
       * a tz-aware timestamp in UTC,
       * the correct hard-wired `kind`.
    """
    node: GraphNode = cls()  # type: ignore[call-arg]

    # id
    assert isinstance(node.id, str)
    assert UUID_V4_RE.match(node.id)

    # ts
    assert node.ts.tzinfo is timezone.utc
    assert node.ts <= node.ts.now(timezone.utc)

    # kind
    assert node.kind == expected_kind


def test_node_immutable():
    node = PlanNode()
    with pytest.raises(TypeError):
        node.kind = NodeKind.USER_MSG  # noqa: F841  (pydantic frozen error)
    with pytest.raises(TypeError):
        node.data["foo"] = "bar"       # underlying mapping is frozen too


def test_repr_format():
    msg = UserMessage()
    text = repr(msg)
    # Expected shape: <user_message:1234abcd>
    assert text.startswith(f"<{msg.kind.value}:")
    assert text.endswith(">")
    # hex prefix of id is included
    assert msg.id[:8] in text
