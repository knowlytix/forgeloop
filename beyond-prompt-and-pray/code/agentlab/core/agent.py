"""BaseAgent: the minimum surface an agent must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentlab.core.action import Action
from agentlab.core.state import AgentState


class BaseAgent(ABC):
    @abstractmethod
    def propose_action(self, state: AgentState) -> Action:
        """Return the next action the agent proposes given the current state."""
        ...

    def update(
        self,
        state: AgentState,
        action: Action,
        observation: dict[str, Any],
    ) -> AgentState:
        """Produce the next state after an action and its observation.

        Increments the step counter on a deep copy of the state and appends the
        observation to tool_results when it is non-empty.

        Args:
            state: The state before the action was taken.
            action: The action that was proposed.
            observation: The environment's response to the action.

        Returns:
            A new AgentState with the step advanced and the observation recorded.
        """
        new = state.model_copy(deep=True)
        new.step += 1
        if observation:
            new.tool_results.append(observation)
        return new
