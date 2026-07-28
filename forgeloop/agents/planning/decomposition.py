"""Turn a task and its plan into typed subtasks."""

from __future__ import annotations

from forgeloop.agents.core.task import TaskSpec
from forgeloop.agents.planning.plan import Plan


def decompose(task: TaskSpec, plan: Plan | None = None) -> list[TaskSpec]:
    """Turn a task and its plan into one subtask per plan step.

    Each subtask takes the step description as its goal and carries the parent
    goal and step id in its inputs, along with the parent's constraints and
    validation.

    Args:
        task: The parent task being decomposed.
        plan: The plan whose steps become subtasks; when None or empty the
            original task is returned unchanged.

    Returns:
        The list of subtasks, or a single-element list holding the task when
        there is no plan.
    """
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
