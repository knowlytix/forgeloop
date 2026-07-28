"""Append-only audit logger backed by a HashChain."""

from __future__ import annotations

from agentlab.audit.event import AuditEvent, SealedEvent
from agentlab.audit.hash_chain import HashChain, verify_chain


class AuditLogger:
    def __init__(self) -> None:
        self._chain = HashChain()
        self._events: list[SealedEvent] = []

    def log(self, event: AuditEvent) -> SealedEvent:
        sealed = self._chain.seal(event)
        self._events.append(sealed)
        return sealed

    @property
    def events(self) -> list[SealedEvent]:
        return list(self._events)

    def head(self) -> str:
        return self._chain.head()

    def verify(self) -> bool:
        return verify_chain(self._events)

    def replay(self) -> list[AuditEvent]:
        return [s.event for s in self._events]
