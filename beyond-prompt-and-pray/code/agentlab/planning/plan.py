"""Plan and PlanStep: a plan is a typed list of steps, not a paragraph."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlanStep:
    id: str
    description: str
    action_hint: str = ""
    expected_output_type: str | None = None
    validation: str | None = None
    requires: tuple[str, ...] = field(default_factory=tuple)
    fallback: str | None = None


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)

    def get(self, step_id: str) -> PlanStep:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def by_index(self, i: int) -> PlanStep:
        return self.steps[i]

    def __len__(self) -> int:
        return len(self.steps)
