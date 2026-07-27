"""SHA-256 hash chain. Each event's hash incorporates the previous one."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from forgeloop.agents.audit.event import AuditEvent, SealedEvent

GENESIS = "0" * 64


def _serialize(event: AuditEvent, prev_hash: str) -> str:
    d = asdict(event)
    d["prev_hash"] = prev_hash
    return json.dumps(d, sort_keys=True, default=str)


class HashChain:
    """Stateful SHA-256 hash chain that seals events against the previous hash."""

    def __init__(self) -> None:
        self._prev_hash = GENESIS

    def seal(self, event: AuditEvent) -> SealedEvent:
        """Hash the event with the current head and return the sealed event.

        Advances the chain head to the new hash.

        Args:
            event: The audit event to seal.

        Returns:
            The sealed event holding the event, the previous hash and its hash.
        """
        payload = _serialize(event, self._prev_hash)
        h = hashlib.sha256(payload.encode()).hexdigest()
        sealed = SealedEvent(event=event, prev_hash=self._prev_hash, event_hash=h)
        self._prev_hash = h
        return sealed

    def head(self) -> str:
        """Return the current head hash of the chain."""
        return self._prev_hash


def verify_chain(sealed: list[SealedEvent]) -> bool:
    """Return True if the sealed events form an unbroken hash chain from genesis.

    Args:
        sealed: The sealed events in order.

    Returns:
        True if every event links to the prior hash and its recomputed hash
        matches, False otherwise.
    """
    prev = GENESIS
    for s in sealed:
        if s.prev_hash != prev:
            return False
        payload = _serialize(s.event, prev)
        h = hashlib.sha256(payload.encode()).hexdigest()
        if h != s.event_hash:
            return False
        prev = h
    return True
