"""AgentState: where the agent is, not where the conversation is.

State is serializable so it can be persisted, replayed and audited. Other
sub-packages (reasoning, tools, memory) store their items as opaque dicts
here; this avoids cross-chapter import dependencies.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from forgeloop.agents.core.task import TaskSpec


class AgentState(BaseModel):
    task: TaskSpec
    step: int = 0
    messages: list[dict[str, Any]] = Field(default_factory=list)
    scratchpad_entries: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "running"
    final_output: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Return the state as a plain dict via Pydantic model_dump."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentState":
        """Construct and validate an AgentState from a dict."""
        return cls.model_validate(d)
