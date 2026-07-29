"""Append-only audit logger backed by a HashChain."""

from __future__ import annotations

from agentlab.audit.event import AuditEvent, SealedEvent
from agentlab.audit.hash_chain import HashChain, verify_chain


class AuditLogger:
    """Append-only log of audit events sealed into a hash chain."""

    def __init__(self) -> None:
        self._chain = HashChain()
        self._events: list[SealedEvent] = []

    def log(self, event: AuditEvent) -> SealedEvent:
        """Seal the event into the chain, append it and return the sealed event."""
        sealed = self._chain.seal(event)
        self._events.append(sealed)
        return sealed

    @property
    def events(self) -> list[SealedEvent]:
        return list(self._events)

    def head(self) -> str:
        """Return the current head hash of the underlying chain."""
        return self._chain.head()

    def verify(self) -> bool:
        """Return True if the logged sealed events form a valid hash chain."""
        return verify_chain(self._events)

    def replay(self) -> list[AuditEvent]:
        """Return the logged audit events in order, without their seals."""
        return [s.event for s in self._events]
