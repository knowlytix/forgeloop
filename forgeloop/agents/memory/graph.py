"""Graph memory: triple store with multi-hop neighbor search."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from forgeloop.agents.memory.base import MemoryItem, MemoryKind


@dataclass(frozen=True)
class Triple:
    """A subject-relation-object fact with an optional source.

    Attributes:
        subject: The head entity of the fact.
        relation: The relation linking subject and object.
        object: The tail entity of the fact.
        source: Optional provenance label for the fact.
    """

    subject: str
    relation: str
    object: str
    source: str | None = None


class GraphMemory:
    """Triple store supporting multi-hop neighbor traversal and substring queries."""

    def __init__(self) -> None:
        self._triples: list[Triple] = []

    def add_triple(self, t: Triple) -> None:
        """Append a triple to the store."""
        self._triples.append(t)

    def add(self, item: MemoryItem) -> None:
        """Store the item's "triple" metadata entry if it is a Triple."""
        t = item.metadata.get("triple")
        if isinstance(t, Triple):
            self.add_triple(t)

    def neighbors(self, entity: str, hops: int = 1) -> list[Triple]:
        """Return triples reachable from an entity within a number of hops.

        Traverses triples in both directions by breadth-first search, visiting
        each connected entity once.

        Args:
            entity: The entity to start traversal from.
            hops: Number of hops to expand from the entity.

        Returns:
            The triples encountered during traversal.
        """
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
        """Return up to k memory items for triples whose subject, relation or object contains q.

        Args:
            q: Case-insensitive substring matched against each triple field.
            k: Maximum number of items to return.

        Returns:
            Semantic-kind memory items wrapping the matching triples.
        """
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
