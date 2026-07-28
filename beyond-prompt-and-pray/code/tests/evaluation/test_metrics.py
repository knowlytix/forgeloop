from agentlab.core import (
    AgentState,
    BaseAgent,
    Budget,
    BudgetTracker,
    Escalate,
    Finish,
    TaskSpec,
    ToolCall,
    run_loop,
)
from agentlab.evaluation import (
    collect,
    escalated,
    failed,
    finished_cleanly,
    step_count,
    summarize,
    task_success,
    tool_call_count,
    tool_failure_count,
)


class _PingAgent(BaseAgent):
    def __init__(self, max_ping: int) -> None:
        self._max = max_ping

    def propose_action(self, state):
        if state.step >= self._max:
            return Finish(output="done")
        return ToolCall(tool_name="ping", arguments={})


class _EscalateAgent(BaseAgent):
    def propose_action(self, state):
        return Escalate(reason="need human")


class _FailingEnv:
    def step(self, action):
        return {"success": False, "error": "tool broke"}


class _OkEnv:
    def step(self, action):
        return {"success": True, "result": "ok"}


def _run(agent, env, max_steps=10, budget_tracker=None):
    task = TaskSpec(goal="g")
    state = AgentState(task=task)
    return collect(task, run_loop(agent, env, state, max_steps=max_steps, budget_tracker=budget_tracker))


def test_task_success_true_on_finish():
    traj = _run(_PingAgent(max_ping=1), _OkEnv())
    assert task_success(traj) == 1.0
    assert finished_cleanly(traj) is True
    assert not failed(traj)
    assert not escalated(traj)


def test_task_success_false_on_escalate():
    traj = _run(_EscalateAgent(), _OkEnv())
    assert task_success(traj) == 0.0
    assert escalated(traj) is True


def test_step_and_tool_counts():
    traj = _run(_PingAgent(max_ping=3), _OkEnv())
    assert step_count(traj) == 4
    assert tool_call_count(traj) == 3


def test_tool_failure_count():
    traj = _run(_PingAgent(max_ping=2), _FailingEnv())
    assert tool_failure_count(traj) == 2


def test_failed_on_budget_exhaustion():
    tracker = BudgetTracker(Budget(tool_calls=1))
    traj = _run(_PingAgent(max_ping=10), _OkEnv(), budget_tracker=tracker)
    assert failed(traj)


def test_summarize_returns_expected_keys():
    traj = _run(_PingAgent(max_ping=1), _OkEnv())
    s = summarize(traj)
    expected = {"status", "success", "steps", "tool_calls", "tool_failures",
                "escalated", "failed", "finished_cleanly", "duration_s"}
    assert expected <= s.keys()
