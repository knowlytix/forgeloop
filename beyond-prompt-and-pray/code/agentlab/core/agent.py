"""BaseAgent: the minimum surface an agent must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentlab.core.action import Action
from agentlab.core.state import AgentState


class BaseAgent(ABC):
    @abstractmethod
    def propose_action(self, state: AgentState) -> Action: ...

    def update(
        self,
        state: AgentState,
        action: Action,
        observation: dict[str, Any],
    ) -> AgentState:
        new = state.model_copy(deep=True)
        new.step += 1
        if observation:
            new.tool_results.append(observation)
        return new
