"""Tool record and risk level enum."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class Tool:
    """A registered tool: its schemas, risk level, contract and implementation.

    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable summary used for routing and listings.
        input_schema: Pydantic model validating the tool's arguments.
        output_schema: Pydantic model describing the tool's output.
        risk: Risk level of invoking the tool.
        preconditions: Named conditions expected to hold before the call.
        postconditions: Named conditions expected to hold after the call.
        fn: The callable implementing the tool, or None if unimplemented.
    """

    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    risk: RiskLevel = RiskLevel.LOW
    preconditions: tuple[str, ...] = field(default_factory=tuple)
    postconditions: tuple[str, ...] = field(default_factory=tuple)
    fn: Callable[..., Any] | None = None
