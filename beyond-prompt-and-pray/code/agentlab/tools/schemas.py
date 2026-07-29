"""Tool input/output schema base classes and argument validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentlab.tools.base import Tool


class ToolInput(BaseModel):
    """Marker base class for tool input schemas."""


class ToolOutput(BaseModel):
    """Marker base class for tool output schemas."""


def validate_arguments(tool: Tool, arguments: dict[str, Any]) -> BaseModel:
    """Validate arguments against a tool's input schema and return the parsed model.

    Args:
        tool: The tool whose input_schema validates the arguments.
        arguments: The raw argument mapping to validate.

    Returns:
        The validated input-schema instance.

    Raises:
        ValidationError: If the arguments do not conform to the input schema.
    """
    return tool.input_schema.model_validate(arguments)
