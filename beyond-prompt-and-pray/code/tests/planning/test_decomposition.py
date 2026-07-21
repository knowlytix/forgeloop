from agentlab.core import TaskSpec
from agentlab.planning import Plan, PlanStep, decompose


def test_decompose_without_plan_returns_self():
    t = TaskSpec(goal="g")
    subs = decompose(t, None)
    assert subs == [t]


def test_decompose_empty_plan_returns_self():
    t = TaskSpec(goal="g")
    subs = decompose(t, Plan(steps=[]))
    assert subs == [t]


def test_decompose_produces_one_task_per_step():
    t = TaskSpec(goal="parent goal", constraints=["no PII"])
    plan = Plan(steps=[
        PlanStep(id="a", description="step a", expected_output_type="doc"),
        PlanStep(id="b", description="step b"),
    ])
    subs = decompose(t, plan)
    assert len(subs) == 2
    assert subs[0].goal == "step a"
    assert subs[0].inputs["_parent_goal"] == "parent goal"
    assert subs[0].inputs["_step_id"] == "a"
    assert subs[0].constraints == ["no PII"]
    assert subs[0].expected_outputs == ["doc"]
    assert subs[1].expected_outputs == []
