"""The agent loop as a generator of step records.

Every step yields a StepRecord. The loop is inspectable: callers see what
the agent proposed, what the environment returned and how state changed.
A Finish or Escalate action terminates the loop. If a BudgetTracker is
passed and any axis is exhausted, the loop synthesizes a final Escalate
record with source=budget and sets state.status = "failed".
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agentlab.core.action import Action, Escalate
from agentlab.core.agent import BaseAgent
from agentlab.core.budget import BudgetTracker
from agentlab.core.state import AgentState


@runtime_checkable
class Environment(Protocol):
    def step(self, action: Action) -> dict[str, Any]: ...


@dataclass
class StepRecord:
    step: int
    state_before: AgentState
    action: Action
    observation: dict[str, Any]
    state_after: AgentState


def run_loop(
    agent: BaseAgent,
    env: Environment | None,
    initial_state: AgentState,
    max_steps: int = 32,
    budget_tracker: BudgetTracker | None = None,
) -> Iterator[StepRecord]:
    state = initial_state
    for step_i in range(max_steps):
        if state.status != "running":
            break

        if budget_tracker is not None and budget_tracker.exhausted():
            reason = budget_tracker.reason_exhausted() or "budget exhausted"
            new_state = state.model_copy(deep=True)
            new_state.step += 1
            new_state.status = "failed"
            yield StepRecord(
                step=step_i,
                state_before=state,
                action=Escalate(reason=reason, context={"source": "budget"}),
                observation={"budget_reason": reason},
                state_after=new_state,
            )
            break

        action = agent.propose_action(state)

        if action.kind == "finish":
            new_state = state.model_copy(deep=True)
            new_state.step += 1
            new_state.status = "done"
            new_state.final_output = action.output
            yield StepRecord(
                step=step_i,
                state_before=state,
                action=action,
                observation={},
                state_after=new_state,
            )
            state = new_state
            break

        if action.kind == "escalate":
            new_state = state.model_copy(deep=True)
            new_state.step += 1
            new_state.status = "escalated"
            yield StepRecord(
                step=step_i,
                state_before=state,
                action=action,
                observation={"reason": action.reason},
                state_after=new_state,
            )
            state = new_state
            break

        observation = env.step(action) if env is not None else {}
        if budget_tracker is not None and action.kind == "tool_call":
            budget_tracker.record_tool_call()
        new_state = agent.update(state, action, observation)
        yield StepRecord(
            step=step_i,
            state_before=state,
            action=action,
            observation=observation,
            state_after=new_state,
        )
        state = new_state
