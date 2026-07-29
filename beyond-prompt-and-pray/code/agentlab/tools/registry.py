"""ToolRegistry: register, look up and schema-validate tool calls."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from agentlab.tools.base import Tool
from agentlab.tools.schemas import validate_arguments

_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class ToolRegistry:
    """Holds tools by name and validates arguments against their input schemas."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry.

        Args:
            tool: The tool to register.

        Raises:
            ValueError: If the tool name is not a valid identifier or is already
                registered.
        """
        if not _NAME_RE.match(tool.name):
            raise ValueError(
                f"invalid tool name {tool.name!r}: must match {_NAME_RE.pattern}"
            )
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the tool registered under name.

        Raises:
            KeyError: If no tool is registered under name.
        """
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def all(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def validate(self, name: str, arguments: dict[str, Any]) -> BaseModel:
        """Validate arguments against the named tool's input schema and return the parsed model."""
        return validate_arguments(self.get(name), arguments)
