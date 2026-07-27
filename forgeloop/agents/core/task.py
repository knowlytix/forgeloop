"""TaskSpec and ValidationRule.

A task is more than a prompt. It carries goal, inputs, expected outputs,
constraints and validation criteria. Budget arrives in Chapter 7.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ValidationRule(BaseModel):
    name: str
    description: str


class TaskSpec(BaseModel):
    goal: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    expected_outputs: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    validation: list[ValidationRule] = Field(default_factory=list)
    budget_hint: dict[str, Any] | None = None

    @field_validator("goal")
    @classmethod
    def _goal_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("goal must be non-empty")
        return v
