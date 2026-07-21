from agentlab.core import AgentState, TaskSpec


def test_default_state():
    s = AgentState(task=TaskSpec(goal="g"))
    assert s.step == 0
    assert s.status == "running"
    assert s.messages == []


def test_roundtrip_serialization():
    s = AgentState(
        task=TaskSpec(goal="g", constraints=["c"]),
        step=3,
        messages=[{"role": "user", "content": "hi"}],
        tool_results=[{"tool": "search", "result": [1, 2, 3]}],
        status="running",
    )
    d = s.to_dict()
    s2 = AgentState.from_dict(d)
    assert s2 == s


def test_status_can_change():
    s = AgentState(task=TaskSpec(goal="g"))
    s.status = "done"
    s.final_output = {"answer": 42}
    assert s.status == "done"
    assert s.final_output == {"answer": 42}
