import time

from agentlab.core import (
    AgentState,
    BaseAgent,
    Budget,
    BudgetTracker,
    Consumption,
    TaskSpec,
    ToolCall,
    run_loop,
)


def test_budget_defaults_unlimited():
    b = Budget()
    t = BudgetTracker(b)
    t.record_tokens(10_000_000)
    t.record_tool_call(1000)
    t.record_dollars(1e9)
    assert not t.exhausted()


def test_tokens_exhausted():
    t = BudgetTracker(Budget(tokens=100))
    t.record_tokens(50)
    assert not t.exhausted()
    t.record_tokens(50)
    assert t.exhausted()
    assert "tokens" in t.reason_exhausted()


def test_tool_calls_exhausted():
    t = BudgetTracker(Budget(tool_calls=2))
    t.record_tool_call()
    t.record_tool_call()
    assert t.exhausted()
    assert "tool" in t.reason_exhausted()


def test_dollars_exhausted():
    t = BudgetTracker(Budget(dollars=0.10))
    t.record_dollars(0.11)
    assert t.exhausted()
    assert "dollars" in t.reason_exhausted()


def test_consumption_snapshot_includes_elapsed():
    t = BudgetTracker(Budget())
    time.sleep(0.01)
    c: Consumption = t.consumption()
    assert c.seconds > 0


class _NeverFinish(BaseAgent):
    def propose_action(self, state):
        return ToolCall(tool_name="ping", arguments={})


class _Env:
    def step(self, action):
        return {"pong": True}


def test_loop_aborts_when_budget_exhausted():
    state = AgentState(task=TaskSpec(goal="g"))
    tracker = BudgetTracker(Budget(tool_calls=2))
    records = list(run_loop(_NeverFinish(), _Env(), state, max_steps=10, budget_tracker=tracker))
    assert records[-1].state_after.status == "failed"
    assert "tool" in records[-1].observation.get("budget_reason", "")


def test_loop_records_tool_calls_into_tracker():
    state = AgentState(task=TaskSpec(goal="g"))
    tracker = BudgetTracker(Budget(tool_calls=100))
    list(run_loop(_NeverFinish(), _Env(), state, max_steps=3, budget_tracker=tracker))
    assert tracker.consumption().tool_calls == 3
