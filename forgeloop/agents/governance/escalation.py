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
    """A reviewer's decision on an escalation request.

    Attributes:
        decision: The reviewer's decision (approve, deny or defer).
        note: An optional free-text note explaining the decision.
    """

    decision: HumanDecision
    note: str = ""


@dataclass(frozen=True)
class EscalationRequest:
    """Context handed to a human reviewer for a single escalated step.

    Attributes:
        run_id: Identifier of the run the escalation belongs to.
        step: The review index within the run.
        reason: A short summary of why the step escalated.
        proposed_action: The action awaiting a decision, as a dict.
        gate_results: The gate results that produced the escalation.
    """

    run_id: str
    step: int
    reason: str
    proposed_action: dict[str, Any]
    gate_results: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the request to a JSON string.

        Args:
            indent: Optional indentation passed to json.dumps.

        Returns:
            The request as a JSON string.
        """
        return json.dumps(asdict(self), indent=indent, default=str)

    @classmethod
    def from_json(cls, s: str) -> "EscalationRequest":
        """Deserialize a request from a JSON string.

        Args:
            s: A JSON string produced by to_json.

        Returns:
            The reconstructed EscalationRequest.
        """
        d = json.loads(s)
        return cls(**d)


@runtime_checkable
class HumanReviewer(Protocol):
    def review(self, request: EscalationRequest) -> HumanResponse:
        """Return a decision for an escalation request."""
        ...


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
        """Record the request and return the next scripted response.

        Args:
            request: The escalation request to review.

        Returns:
            The next response from the configured list, cycling on overflow.
        """
        self.requests.append(request)
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r


class CLIReviewer:
    """Reads a decision from stdin. Useful in notebooks; not for tests."""

    def review(self, request: EscalationRequest) -> HumanResponse:
        """Print the request and read a decision and note from stdin.

        Args:
            request: The escalation request to review.

        Returns:
            The response parsed from stdin; an unrecognized decision defers.
        """
        print(request.to_json(indent=2))
        decision_str = input("[approve/deny/defer]: ").strip().lower()
        try:
            decision = HumanDecision(decision_str)
        except ValueError:
            decision = HumanDecision.DEFER
        note = input("note (optional): ").strip()
        return HumanResponse(decision=decision, note=note)
