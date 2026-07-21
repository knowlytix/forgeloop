"""Audit event dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    run_id: str
    step: int
    timestamp: float
    state_hash: str
    proposed_action: dict[str, Any]
    observation: dict[str, Any]
    final_state_status: str


@dataclass(frozen=True)
class SealedEvent:
    event: AuditEvent
    prev_hash: str
    event_hash: str
