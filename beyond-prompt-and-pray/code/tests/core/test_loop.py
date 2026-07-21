from agentlab.core import (
    AgentState,
    BaseAgent,
    Escalate,
    Finish,
    StepRecord,
    TaskSpec,
    ToolCall,
    run_loop,
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
        return Escalate(reason="needs human")


class _NeverFinishAgent(BaseAgent):
    def propose_action(self, state):
        return ToolCall(tool_name="ping", arguments={})


class _Env:
    def step(self, action):
        return {"pong": True}


def test_loop_terminates_on_finish():
    state = AgentState(task=TaskSpec(goal="g"))
    records = list(run_loop(_PingAgent(max_ping=3), _Env(), state, max_steps=10))
    assert records[-1].state_after.status == "done"
    assert records[-1].state_after.final_output == "done"


def test_loop_terminates_on_escalate():
    state = AgentState(task=TaskSpec(goal="g"))
    records = list(run_loop(_EscalateAgent(), _Env(), state, max_steps=10))
    assert len(records) == 1
    assert records[-1].state_after.status == "escalated"


def test_loop_respects_max_steps():
    state = AgentState(task=TaskSpec(goal="g"))
    records = list(run_loop(_NeverFinishAgent(), _Env(), state, max_steps=5))
    assert len(records) == 5


def test_loop_yields_step_records():
    state = AgentState(task=TaskSpec(goal="g"))
    records = list(run_loop(_PingAgent(max_ping=1), _Env(), state, max_steps=10))
    assert all(isinstance(r, StepRecord) for r in records)


def test_loop_without_environment():
    state = AgentState(task=TaskSpec(goal="g"))
    records = list(run_loop(_PingAgent(max_ping=2), None, state, max_steps=10))
    assert records[-1].state_after.status == "done"
