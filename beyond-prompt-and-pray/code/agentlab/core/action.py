"""Actions as a discriminated union: ToolCall, AskUser, Finish, Escalate.

The kind field is constrained by Literal so constructing a subclass with
the wrong kind raises ValidationError. parse_action dispatches on kind
for deserialization from audit logs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class ActionKind(str, Enum):
    TOOL_CALL = "tool_call"
    ASK_USER = "ask_user"
    FINISH = "finish"
    ESCALATE = "escalate"


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["tool_call"] = "tool_call"
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AskUser(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["ask_user"] = "ask_user"
    question: str


class Finish(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["finish"] = "finish"
    output: Any = None


class Escalate(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["escalate"] = "escalate"
    reason: str
    context: dict[str, Any] = Field(default_factory=dict)


Action = Union[ToolCall, AskUser, Finish, Escalate]


def parse_action(d: dict[str, Any]) -> Action:
    kind = d.get("kind")
    if kind == "tool_call":
        return ToolCall.model_validate(d)
    if kind == "ask_user":
        return AskUser.model_validate(d)
    if kind == "finish":
        return Finish.model_validate(d)
    if kind == "escalate":
        return Escalate.model_validate(d)
    raise ValueError(f"unknown action kind: {kind!r}")
