"""Audit event dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    """One audited step of an agent run.

    Attributes:
        run_id: Identifier of the run the event belongs to.
        step: Zero-based step index within the run.
        timestamp: Time the event was recorded, as a Unix timestamp.
        state_hash: Hash of the state at the step.
        proposed_action: The action proposed at the step, as a dict.
        observation: The observation returned for the action, as a dict.
        final_state_status: The state status after the step.
    """

    run_id: str
    step: int
    timestamp: float
    state_hash: str
    proposed_action: dict[str, Any]
    observation: dict[str, Any]
    final_state_status: str


@dataclass(frozen=True)
class SealedEvent:
    """An audit event bound into a hash chain.

    Attributes:
        event: The audited event.
        prev_hash: Hash of the preceding sealed event.
        event_hash: Hash of this event combined with prev_hash.
    """

    event: AuditEvent
    prev_hash: str
    event_hash: str
