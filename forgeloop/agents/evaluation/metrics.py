"""Pure functions over a Trajectory. Each metric is independent and composable.

Metrics are deliberately simple here. The point of the chapter is to teach
trajectory-level evaluation, not to ship a complete metric library.
"""

from __future__ import annotations

import json
from typing import Any

from forgeloop.agents.evaluation.trajectory import Trajectory


def task_success(traj: Trajectory) -> float:
    """Return 1.0 when the run ended in the done status, else 0.0."""
    return 1.0 if traj.final_state.status == "done" else 0.0


def step_count(traj: Trajectory) -> int:
    """Return the number of steps recorded in the trajectory."""
    return len(traj.records)


def tool_call_count(traj: Trajectory) -> int:
    """Return the number of tool-call steps in the trajectory."""
    return sum(1 for r in traj.records if r.action.kind == "tool_call")


def tool_failure_count(traj: Trajectory) -> int:
    """Return the number of tool calls whose observation reported failure."""
    return sum(
        1
        for r in traj.records
        if r.action.kind == "tool_call" and r.observation.get("success") is False
    )


def escalated(traj: Trajectory) -> bool:
    """Return whether the run ended in the escalated status."""
    return traj.final_state.status == "escalated"


def failed(traj: Trajectory) -> bool:
    """Return whether the run ended in the failed status."""
    return traj.final_state.status == "failed"


def finished_cleanly(traj: Trajectory) -> bool:
    """Return whether the run ended in the done or escalated status."""
    return traj.final_state.status in ("done", "escalated")


def average_observation_size(traj: Trajectory) -> float:
    """Return the mean JSON-serialized size of the step observations.

    Args:
        traj: The trajectory to measure.

    Returns:
        The mean observation size in characters, or 0.0 when empty.
    """
    if not traj.records:
        return 0.0
    sizes = [len(json.dumps(r.observation, default=str)) for r in traj.records]
    return sum(sizes) / len(sizes)


def summarize(traj: Trajectory) -> dict[str, Any]:
    """Return the trajectory metrics collected into a single dict.

    Args:
        traj: The trajectory to summarize.

    Returns:
        A dict of status, success, step and tool counts, terminal flags and
        duration.
    """
    return {
        "status": traj.final_state.status,
        "success": task_success(traj),
        "steps": step_count(traj),
        "tool_calls": tool_call_count(traj),
        "tool_failures": tool_failure_count(traj),
        "escalated": escalated(traj),
        "failed": failed(traj),
        "finished_cleanly": finished_cleanly(traj),
        "duration_s": traj.duration,
    }
