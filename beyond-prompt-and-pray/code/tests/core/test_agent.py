import pytest

from agentlab.core import AgentState, BaseAgent, Finish, TaskSpec


class _StubAgent(BaseAgent):
    def propose_action(self, state):
        return Finish(output="done")


def test_base_agent_is_abstract():
    with pytest.raises(TypeError):
        BaseAgent()


def test_default_update_increments_step():
    a = _StubAgent()
    s = AgentState(task=TaskSpec(goal="g"))
    new = a.update(s, Finish(output=1), {"x": 1})
    assert new.step == 1
    assert new.tool_results == [{"x": 1}]


def test_default_update_does_not_mutate_input():
    a = _StubAgent()
    s = AgentState(task=TaskSpec(goal="g"))
    a.update(s, Finish(output=1), {"x": 1})
    assert s.step == 0
    assert s.tool_results == []
