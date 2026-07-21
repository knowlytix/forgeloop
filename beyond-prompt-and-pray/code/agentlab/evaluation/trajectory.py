"""Trajectory: the full record of an agent's run, suitable for evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from agentlab.core.loop import StepRecord
from agentlab.core.state import AgentState
from agentlab.core.task import TaskSpec


@dataclass
class Trajectory:
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
    traj = Trajectory(task=task)
    for rec in step_iter:
        traj.records.append(rec)
    traj.ended_at = time.time()
    return traj
