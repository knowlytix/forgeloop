"""Replanning triggers and a default replanner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from forgeloop.agents.core.task import TaskSpec
from forgeloop.agents.planning.plan import Plan
from forgeloop.agents.planning.planner import Planner


class ReplanReason(str, Enum):
    VERIFICATION_FAILED = "verification_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TOOL_FAILURE = "tool_failure"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    MISSING_TOOL = "missing_tool"


@dataclass(frozen=True)
class ReplanTrigger:
    """A reason to re-plan together with supporting detail.

    Attributes:
        reason: The category of event prompting a re-plan.
        detail: Optional context describing the specific trigger.
    """

    reason: ReplanReason
    detail: str = ""


def should_replan(last_tool_result: dict[str, Any] | None) -> ReplanTrigger | None:
    """Inspect the last tool result. Return a trigger if replanning is warranted."""
    if last_tool_result is None:
        return None
    if last_tool_result.get("success") is False:
        return ReplanTrigger(
            reason=ReplanReason.TOOL_FAILURE,
            detail=str(last_tool_result.get("error") or ""),
        )
    return None


def replan(planner: Planner, task: TaskSpec, trigger: ReplanTrigger) -> Plan:
    """Re-plan with the trigger annotated into the task inputs."""
    annotated = TaskSpec(
        goal=task.goal,
        inputs={
            **task.inputs,
            "_replan_reason": trigger.reason.value,
            "_replan_detail": trigger.detail,
        },
        expected_outputs=list(task.expected_outputs),
        constraints=list(task.constraints),
        validation=list(task.validation),
    )
    return planner.plan(annotated)
