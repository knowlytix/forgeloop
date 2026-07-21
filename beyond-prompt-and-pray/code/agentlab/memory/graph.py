"""Graph memory: triple store with multi-hop neighbor search."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from agentlab.memory.base import MemoryItem, MemoryKind


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    source: str | None = None


class GraphMemory:
    def __init__(self) -> None:
        self._triples: list[Triple] = []

    def add_triple(self, t: Triple) -> None:
        self._triples.append(t)

    def add(self, item: MemoryItem) -> None:
        t = item.metadata.get("triple")
        if isinstance(t, Triple):
            self.add_triple(t)

    def neighbors(self, entity: str, hops: int = 1) -> list[Triple]:
        seen: set[str] = {entity}
        frontier: deque[str] = deque([entity])
        out: list[Triple] = []
        for _ in range(hops):
            next_frontier: deque[str] = deque()
            while frontier:
                e = frontier.popleft()
                for t in self._triples:
                    if t.subject == e and t.object not in seen:
                        out.append(t)
                        seen.add(t.object)
                        next_frontier.append(t.object)
                    elif t.object == e and t.subject not in seen:
                        out.append(t)
                        seen.add(t.subject)
                        next_frontier.append(t.subject)
            frontier = next_frontier
        return out

    def query(self, q: str, k: int = 5) -> list[MemoryItem]:
        ql = q.lower()
        hits = []
        for t in self._triples:
            if ql in t.subject.lower() or ql in t.relation.lower() or ql in t.object.lower():
                hits.append(
                    MemoryItem(
                        content=f"{t.subject} {t.relation} {t.object}",
                        kind=MemoryKind.SEMANTIC,
                        metadata={"triple": t},
                    )
                )
        return hits[:k]
