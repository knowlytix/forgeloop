"""Short-term memory: bounded queue plus a typed working-memory record."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from agentlab.memory.base import MemoryItem


class ShortTermMemory:
    def __init__(self, maxlen: int = 32) -> None:
        self._buf: deque[MemoryItem] = deque(maxlen=maxlen)

    def add(self, item: MemoryItem) -> None:
        self._buf.append(item)

    def query(self, q: str, k: int = 5) -> list[MemoryItem]:
        ql = q.lower()
        hits = [m for m in reversed(self._buf) if ql in m.content.lower()]
        return hits[:k]

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class WorkingMemory:
    task_id: str
    facts: dict[str, Any] = field(default_factory=dict)
    open_questions: list[str] = field(default_factory=list)
