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
    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    risk: RiskLevel = RiskLevel.LOW
    preconditions: tuple[str, ...] = field(default_factory=tuple)
    postconditions: tuple[str, ...] = field(default_factory=tuple)
    fn: Callable[..., Any] | None = None
