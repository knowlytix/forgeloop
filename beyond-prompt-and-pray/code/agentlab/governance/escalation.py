"""Escalation requests and human reviewers.

An EscalationRequest carries enough context for a human to make a
decision: the proposed action, the gate results that escalated and a
short reason. A HumanReviewer maps the request to a HumanResponse.
ScriptedReviewer is for tests; CLIReviewer is for notebooks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class HumanDecision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    DEFER = "defer"


@dataclass(frozen=True)
class HumanResponse:
    decision: HumanDecision
    note: str = ""


@dataclass(frozen=True)
class EscalationRequest:
    run_id: str
    step: int
    reason: str
    proposed_action: dict[str, Any]
    gate_results: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)

    @classmethod
    def from_json(cls, s: str) -> "EscalationRequest":
        d = json.loads(s)
        return cls(**d)


@runtime_checkable
class HumanReviewer(Protocol):
    def review(self, request: EscalationRequest) -> HumanResponse: ...


class ScriptedReviewer:
    """Returns responses from a list, cycling on overflow."""

    def __init__(self, responses: list[HumanResponse] | HumanResponse) -> None:
        if isinstance(responses, HumanResponse):
            responses = [responses]
        if not responses:
            raise ValueError("ScriptedReviewer needs at least one response")
        self._responses = list(responses)
        self._i = 0
        self.requests: list[EscalationRequest] = []

    def review(self, request: EscalationRequest) -> HumanResponse:
        self.requests.append(request)
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r


class CLIReviewer:
    """Reads a decision from stdin. Useful in notebooks; not for tests."""

    def review(self, request: EscalationRequest) -> HumanResponse:
        print(request.to_json(indent=2))
        decision_str = input("[approve/deny/defer]: ").strip().lower()
        try:
            decision = HumanDecision(decision_str)
        except ValueError:
            decision = HumanDecision.DEFER
        note = input("note (optional): ").strip()
        return HumanResponse(decision=decision, note=note)
