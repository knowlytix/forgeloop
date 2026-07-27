"""Trajectory: the full record of an agent's run, suitable for evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from forgeloop.agents.core.loop import StepRecord
from forgeloop.agents.core.state import AgentState
from forgeloop.agents.core.task import TaskSpec


@dataclass
class Trajectory:
    """The full record of an agent run for evaluation.

    Attributes:
        task: The task specification the run addressed.
        records: The per-step records in execution order.
        started_at: Wall-clock start time in seconds.
        ended_at: Wall-clock end time in seconds, or None while running.
    """

    task: TaskSpec
    records: list[StepRecord] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    @property
    def final_state(self) -> AgentState:
        if not self.records:
            return AgentState(task=self.task)
        return self.records[-1].state_after

    @property
    def duration(self) -> float:
        return (self.ended_at if self.ended_at is not None else time.time()) - self.started_at


def collect(task: TaskSpec, step_iter: Iterable[StepRecord]) -> Trajectory:
    """Drain an iterator of step records into a completed Trajectory.

    Args:
        task: The task specification for the run.
        step_iter: An iterable of step records to collect.

    Returns:
        A Trajectory holding the collected records with ended_at set.
    """
    traj = Trajectory(task=task)
    for rec in step_iter:
        traj.records.append(rec)
    traj.ended_at = time.time()
    return traj
