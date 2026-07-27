"""The agent loop as a generator of step records.

Every step yields a StepRecord. The loop is inspectable: callers see what
the agent proposed, what the environment returned and how state changed.
A Finish or Escalate action terminates the loop. If a BudgetTracker is
passed and any axis is exhausted, the loop synthesizes a final Escalate
record with source=budget and sets state.status = "failed".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Protocol, runtime_checkable

from forgeloop.agents.core.action import Action, Escalate
from forgeloop.agents.core.agent import BaseAgent
from forgeloop.agents.core.budget import BudgetTracker
from forgeloop.agents.core.state import AgentState


@runtime_checkable
class Environment(Protocol):
    def step(self, action: Action) -> dict[str, Any]:
        """Execute an action and return the observation it produces."""
        ...


@dataclass
class StepRecord:
    """One step of the agent loop: the action taken and the state around it.

    Attributes:
        step: Zero-based index of the step within the loop.
        state_before: The agent state before the action was proposed.
        action: The action proposed at this step.
        observation: The environment's response to the action.
        state_after: The agent state after the action and observation.
    """

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
    """Run the agent loop, yielding one StepRecord per step until it terminates.

    The loop stops when the agent proposes finish or escalate, when the state
    status is no longer "running", when max_steps is reached, or when the budget
    tracker reports exhaustion. On exhaustion it yields a synthetic Escalate
    record with context source "budget" and sets the state status to "failed".

    Args:
        agent: The agent that proposes actions and updates state.
        env: The environment stepped for tool-call observations, or None to use
            empty observations.
        initial_state: The starting agent state.
        max_steps: Maximum number of loop iterations.
        budget_tracker: Optional tracker; when supplied, tool calls are recorded
            and the loop halts once any budget axis is exhausted.

    Yields:
        A StepRecord for each executed step.
    """
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
