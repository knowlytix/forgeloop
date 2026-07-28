"""Typed agent messages and an in-memory bus.

Free-form chatter between agents is the failure mode this module is
designed to prevent. AgentMessage forces every cross-agent communication
to carry a sender, receiver, message_type and structured payload.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentMessage:
    sender: str
    receiver: str
    message_type: str
    payload: dict[str, Any]
    parent_id: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = field(default_factory=time.time)


class MessageBus:
    """In-memory append-only message log. Production systems use a real queue."""

    def __init__(self) -> None:
        self._messages: list[AgentMessage] = []

    def send(self, msg: AgentMessage) -> None:
        self._messages.append(msg)

    def messages_for(self, receiver: str) -> list[AgentMessage]:
        return [m for m in self._messages if m.receiver == receiver]

    def messages_from(self, sender: str) -> list[AgentMessage]:
        return [m for m in self._messages if m.sender == sender]

    def all(self) -> list[AgentMessage]:
        return list(self._messages)

    def thread(self, root_id: str) -> list[AgentMessage]:
        """Return messages that share a thread with `root_id` (root or any descendant)."""
        thread = [m for m in self._messages if m.id == root_id or m.parent_id == root_id]
        # one more hop for chained replies
        ids = {m.id for m in thread}
        for m in self._messages:
            if m.parent_id in ids and m.id not in ids:
                thread.append(m)
                ids.add(m.id)
        return thread

    def __len__(self) -> int:
        return len(self._messages)
