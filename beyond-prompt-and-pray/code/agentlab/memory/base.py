"""Memory protocol, MemoryItem and MemoryKind taxonomy."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class MemoryKind(str, Enum):
    SHORT_TERM = "short_term"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    POLICY = "policy"
    TOOL = "tool"


@dataclass
class MemoryItem:
    content: str
    kind: MemoryKind
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = field(default_factory=time.time)


@runtime_checkable
class Memory(Protocol):
    def add(self, item: MemoryItem) -> None: ...
    def query(self, q: str, k: int = 5) -> list[MemoryItem]: ...
