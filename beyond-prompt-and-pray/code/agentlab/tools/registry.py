"""ToolRegistry: register, look up and schema-validate tool calls."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from agentlab.tools.base import Tool
from agentlab.tools.schemas import validate_arguments

_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not _NAME_RE.match(tool.name):
            raise ValueError(
                f"invalid tool name {tool.name!r}: must match {_NAME_RE.pattern}"
            )
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def validate(self, name: str, arguments: dict[str, Any]) -> BaseModel:
        return validate_arguments(self.get(name), arguments)
