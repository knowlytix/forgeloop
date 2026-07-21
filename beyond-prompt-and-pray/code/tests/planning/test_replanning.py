from agentlab.core import TaskSpec
from agentlab.planning import (
    PlanStep,
    ReplanReason,
    ReplanTrigger,
    WorkflowPlanner,
    replan,
    should_replan,
)


def test_should_replan_none_when_no_result():
    assert should_replan(None) is None


def test_should_replan_none_on_success():
    assert should_replan({"success": True, "output": {}}) is None


def test_should_replan_triggers_on_failure():
    trigger = should_replan({"success": False, "error": "timeout"})
    assert trigger is not None
    assert trigger.reason == ReplanReason.TOOL_FAILURE
    assert "timeout" in trigger.detail


def test_replan_annotates_task_inputs():
    planner = WorkflowPlanner([PlanStep(id="a", description="x")])
    trigger = ReplanTrigger(reason=ReplanReason.TOOL_FAILURE, detail="rate limit")

    captured: dict = {}

    class _Capture:
        def plan(self, task):
            captured["task"] = task
            return planner.plan(task)

    new_plan = replan(_Capture(), TaskSpec(goal="g"), trigger)
    assert len(new_plan) == 1
    assert captured["task"].inputs["_replan_reason"] == "tool_failure"
    assert captured["task"].inputs["_replan_detail"] == "rate limit"
