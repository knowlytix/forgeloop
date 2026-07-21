"""Turn a task and its plan into typed subtasks."""

from __future__ import annotations

from agentlab.core.task import TaskSpec
from agentlab.planning.plan import Plan


def decompose(task: TaskSpec, plan: Plan | None = None) -> list[TaskSpec]:
    if plan is None or len(plan) == 0:
        return [task]
    subtasks: list[TaskSpec] = []
    for step in plan.steps:
        sub = TaskSpec(
            goal=step.description,
            inputs={**task.inputs, "_parent_goal": task.goal, "_step_id": step.id},
            expected_outputs=[step.expected_output_type] if step.expected_output_type else [],
            constraints=list(task.constraints),
            validation=list(task.validation),
        )
        subtasks.append(sub)
    return subtasks
