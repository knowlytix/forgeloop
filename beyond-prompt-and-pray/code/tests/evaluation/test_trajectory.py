from agentlab.core import (
    AgentState,
    BaseAgent,
    Finish,
    StepRecord,
    TaskSpec,
    ToolCall,
    run_loop,
)
from agentlab.evaluation import Trajectory, collect


class _PingAgent(BaseAgent):
    def __init__(self, max_ping: int) -> None:
        self._max = max_ping

    def propose_action(self, state):
        if state.step >= self._max:
            return Finish(output="done")
        return ToolCall(tool_name="ping", arguments={})


class _Env:
    def step(self, action):
        return {"pong": True}


def test_collect_from_run_loop():
    task = TaskSpec(goal="g")
    state = AgentState(task=task)
    traj = collect(task, run_loop(_PingAgent(max_ping=2), _Env(), state, max_steps=10))
    assert isinstance(traj, Trajectory)
    assert len(traj.records) == 3  # 2 pings + 1 finish
    assert traj.final_state.status == "done"
    assert all(isinstance(r, StepRecord) for r in traj.records)


def test_empty_trajectory_has_default_final_state():
    task = TaskSpec(goal="g")
    traj = Trajectory(task=task)
    assert traj.final_state.status == "running"
    assert traj.final_state.step == 0


def test_duration_nonnegative():
    task = TaskSpec(goal="g")
    state = AgentState(task=task)
    traj = collect(task, run_loop(_PingAgent(max_ping=1), _Env(), state, max_steps=5))
    assert traj.duration >= 0
