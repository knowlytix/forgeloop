import pytest
from pydantic import ValidationError

from agentlab.core import (
    ActionKind,
    AskUser,
    Escalate,
    Finish,
    ToolCall,
    parse_action,
)


def test_tool_call_construction():
    a = ToolCall(tool_name="search", arguments={"q": "x"})
    assert a.kind == "tool_call"
    assert a.tool_name == "search"


def test_ask_user_construction():
    a = AskUser(question="confirm?")
    assert a.kind == "ask_user"


def test_finish_construction():
    a = Finish(output={"answer": 42})
    assert a.kind == "finish"
    assert a.output == {"answer": 42}


def test_escalate_construction():
    a = Escalate(reason="PII detected", context={"field": "email"})
    assert a.kind == "escalate"


def test_action_is_frozen():
    a = ToolCall(tool_name="x")
    with pytest.raises(ValidationError):
        a.tool_name = "y"


def test_wrong_kind_rejected():
    with pytest.raises(ValidationError):
        ToolCall(kind="ask_user", tool_name="x")


def test_parse_action_dispatches():
    cases = [
        ({"kind": "tool_call", "tool_name": "s", "arguments": {}}, ToolCall),
        ({"kind": "ask_user", "question": "q"}, AskUser),
        ({"kind": "finish", "output": 1}, Finish),
        ({"kind": "escalate", "reason": "r"}, Escalate),
    ]
    for d, cls in cases:
        a = parse_action(d)
        assert isinstance(a, cls)


def test_parse_action_unknown_kind():
    with pytest.raises(ValueError):
        parse_action({"kind": "totally_made_up"})


def test_action_kind_enum_values_match():
    assert ActionKind.TOOL_CALL.value == "tool_call"
    assert ActionKind.ASK_USER.value == "ask_user"
    assert ActionKind.FINISH.value == "finish"
    assert ActionKind.ESCALATE.value == "escalate"
