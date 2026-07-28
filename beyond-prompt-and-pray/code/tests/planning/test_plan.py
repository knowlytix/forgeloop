import pytest

from agentlab.planning import Plan, PlanStep


def test_plan_get_by_id():
    p = Plan(steps=[PlanStep(id="a", description="first"), PlanStep(id="b", description="second")])
    assert p.get("a").description == "first"
    assert p.get("b").description == "second"


def test_plan_get_unknown_raises():
    p = Plan(steps=[PlanStep(id="a", description="first")])
    with pytest.raises(KeyError):
        p.get("missing")


def test_plan_by_index():
    p = Plan(steps=[PlanStep(id="a", description="first"), PlanStep(id="b", description="second")])
    assert p.by_index(1).id == "b"


def test_plan_length():
    p = Plan(steps=[PlanStep(id="a", description="x")])
    assert len(p) == 1


def test_plan_step_is_frozen():
    s = PlanStep(id="a", description="x")
    with pytest.raises(Exception):
        s.description = "y"
