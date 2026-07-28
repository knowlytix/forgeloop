"""Short-term memory: bounded queue plus a typed working-memory record."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from forgeloop.agents.memory.base import MemoryItem


class ShortTermMemory:
    """Bounded ring buffer of recent memory items with substring query."""

    def __init__(self, maxlen: int = 32) -> None:
        self._buf: deque[MemoryItem] = deque(maxlen=maxlen)

    def add(self, item: MemoryItem) -> None:
        """Append an item, dropping the oldest once the buffer is full."""
        self._buf.append(item)

    def query(self, q: str, k: int = 5) -> list[MemoryItem]:
        """Return up to k most recent items whose content contains q (case-insensitive)."""
        ql = q.lower()
        hits = [m for m in reversed(self._buf) if ql in m.content.lower()]
        return hits[:k]

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class WorkingMemory:
    """Task-scoped working record of accumulated facts and open questions.

    Attributes:
        task_id: Identifier of the task this record belongs to.
        facts: Named facts gathered while working the task.
        open_questions: Questions still to be resolved.
    """

    task_id: str
    facts: dict[str, Any] = field(default_factory=dict)
    open_questions: list[str] = field(default_factory=list)
