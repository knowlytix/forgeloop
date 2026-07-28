"""GMSPlanGate: validate a whole plan before any step runs.

This is the plan-level invariant from the GMS practitioner framing. Where
``GMSPlausibilityGate`` (see ``gates.py``) checks a single tool call at
execution time, the plan gate checks an entire proposed plan at plan time,
before any tool fires --- so a bad plan costs nothing.

A plan is a sequence of workflow steps. Each consecutive transition
``(step_i, relation, step_{i+1})`` is scored against the store; a step that
its predecessor does not enable scores high. Per-step admissibility is the
job of the Chapter 6 gate applied to each transition; the *plan* gate adds
the whole-plan criterion the practitioner framing calls cumulative
coherence drift: the mean transition distance across the plan must stay
under a calibrated ``drift_budget``. A plan that skips or reorders steps
raises that mean and is rejected before any tool fires, so a bad plan costs
nothing.

The adapter does not import the GMS library. The caller constructs a store
and passes it in; tests pass a mock with a ``score_triple`` method. The
relation defaults to ``has_enables`` --- the store's workflow-authorization
graph --- and scoring starts from ``start``, the conventional source node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class _ScoreTripleStore(Protocol):
    """Minimal protocol the gate needs from a GMS-shaped store."""

    def score_triple(self, head: str, relation: str, tail: str) -> float | None: ...


@dataclass(frozen=True)
class PlanVerdict:
    """Result of validating a plan.

    ``admissible`` is the binary verdict (``drift`` within budget).
    ``transitions`` lists every ``(from_step, to_step, score)`` in order, so
    the caller can see which transition contributed most; ``drift`` is the
    mean transition score across the plan.
    """

    admissible: bool
    drift: float
    transitions: list[tuple[str, str, float | None]]


class GMSPlanGate:
    """Reject a plan whose cumulative coherence drift exceeds a budget.

    Parameters
    ----------
    store : object with ``score_triple(head, relation, tail) -> float | None``
        A GMS-shaped store trained on the workflow-authorization graph.
    drift_budget : float
        Maximum admissible mean transition distance across the whole plan,
        calibrated per domain in Appendix C.
    relation : str
        The relation scored between consecutive steps. Defaults to
        ``"has_enables"``.
    start : str
        The source node the first step is scored against (the plan does not
        include it). Defaults to ``"start"``.
    """

    name = "gms_plan"

    def __init__(
        self,
        store: _ScoreTripleStore,
        drift_budget: float = 1.35,
        relation: str = "has_enables",
        start: str = "start",
    ) -> None:
        self._store = store
        self._drift_budget = float(drift_budget)
        self._relation = relation
        self._start = start

    def validate(self, steps: list[str]) -> PlanVerdict:
        """Score every consecutive transition of ``steps`` and return a verdict."""
        transitions: list[tuple[str, str, float | None]] = []
        total = 0.0
        scored = 0
        prev = self._start
        for step in steps:
            score = self._store.score_triple(prev, self._relation, step)
            transitions.append((prev, step, score))
            if score is not None:
                total += score
                scored += 1
            prev = step
        drift = total / scored if scored else 0.0
        return PlanVerdict(
            admissible=drift <= self._drift_budget,
            drift=drift,
            transitions=transitions,
        )
