import pytest

from agentlab.core import TaskSpec
from agentlab.planning import (
    GraphSearchPlanner,
    LMPlanner,
    Plan,
    PlanStep,
    Planner,
    WorkflowPlanner,
)


def test_workflow_planner_returns_fixed_plan():
    steps = [PlanStep(id="a", description="first"), PlanStep(id="b", description="second")]
    p = WorkflowPlanner(steps).plan(TaskSpec(goal="g"))
    assert isinstance(p, Plan)
    assert len(p) == 2


class _ScriptedLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, prompt: str, **kwargs) -> str:
        return self._response

    def token_count(self, text: str) -> int:
        return len(text.split())


def test_lm_planner_parses_valid_json():
    lm = _ScriptedLM(
        '[{"id":"a","description":"do x","action_hint":"search"},'
        '{"id":"b","description":"do y","action_hint":"compose"}]'
    )
    p = LMPlanner(lm).plan(TaskSpec(goal="g"))
    assert len(p) == 2
    assert p.get("a").action_hint == "search"


def test_lm_planner_rejects_non_json():
    lm = _ScriptedLM("not json")
    with pytest.raises(ValueError):
        LMPlanner(lm).plan(TaskSpec(goal="g"))


def test_lm_planner_rejects_non_array():
    lm = _ScriptedLM('{"steps": []}')
    with pytest.raises(ValueError):
        LMPlanner(lm).plan(TaskSpec(goal="g"))


def test_graph_search_finds_path():
    graph = {
        "start": {"a": "mid"},
        "mid": {"b": "goal"},
    }
    p = GraphSearchPlanner(graph).plan(
        TaskSpec(goal="reach goal", inputs={"start": "start"}, expected_outputs=["goal"])
    )
    hints = [s.action_hint for s in p.steps]
    assert hints == ["a", "b"]


def test_graph_search_no_path():
    graph = {"start": {"a": "dead_end"}}
    p = GraphSearchPlanner(graph).plan(
        TaskSpec(goal="reach goal", inputs={"start": "start"}, expected_outputs=["goal"])
    )
    assert len(p) == 0


def test_planner_protocol_runtime_check():
    assert isinstance(WorkflowPlanner([]), Planner)
    assert isinstance(LMPlanner(_ScriptedLM("[]")), Planner)
    assert isinstance(GraphSearchPlanner({}), Planner)
