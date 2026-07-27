"""Plan and PlanStep: a plan is a typed list of steps, not a paragraph."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlanStep:
    """A single step in a plan.

    Attributes:
        id: Identifier of the step, referenced by dependencies.
        description: What the step is to accomplish.
        action_hint: Optional hint toward the action or tool for the step.
        expected_output_type: Optional name of the output type the step produces.
        validation: Optional description of how to validate the step's result.
        requires: Ids of steps that must complete before this one.
        fallback: Optional id or description of a fallback step.
    """

    id: str
    description: str
    action_hint: str = ""
    expected_output_type: str | None = None
    validation: str | None = None
    requires: tuple[str, ...] = field(default_factory=tuple)
    fallback: str | None = None


@dataclass
class Plan:
    """An ordered list of plan steps.

    Attributes:
        steps: The steps making up the plan, in order.
    """

    steps: list[PlanStep] = field(default_factory=list)

    def get(self, step_id: str) -> PlanStep:
        """Return the step with the given id.

        Raises:
            KeyError: If no step has that id.
        """
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def by_index(self, i: int) -> PlanStep:
        """Return the step at position i in the plan."""
        return self.steps[i]

    def __len__(self) -> int:
        return len(self.steps)
