"""GMSMemory: agentlab Memory protocol on top of a GMS store.

This adapter exposes the six-primitive vocabulary the chapter teaches:
`score_triple`, `tension_energy`, `check_holonomy`, `lookup_enm`,
`link_predict`, `query_triples`. It also conforms to `agentlab.memory.Memory`
(`add` / `query`) by delegating to the store's triple-add path.

We do not ingest documents here. The reader is expected to build a GMS
expert store separately (see `docgms.ingest.ingest_document` in the GMS
library) and pass it in. This adapter is plumbing, not ingestion.
"""

from __future__ import annotations

from typing import Protocol

from agentlab.memory.base import MemoryItem, MemoryKind


class _GMSStore(Protocol):
    """Minimal protocol the memory adapter needs from a GMS store."""

    def score_triple(self, head: str, relation: str, tail: str) -> float | None: ...
    def tension_energy(self, entity_a: str, entity_b: str) -> float | None: ...
    def check_holonomy(self, relation_path: list[str], direct_relation: str) -> float | None: ...
    def lookup_enm(self, category: str, entity_id: str) -> float | None: ...
    def link_predict(self, head: str, relation: str, top_k: int = 10) -> list[tuple[str, float]]: ...
    def query_triples(
        self,
        head: str | None = None,
        relation: str | None = None,
        tail: str | None = None,
    ) -> list[tuple[str, str, str]]: ...


class GMSMemory:
    """Wraps a GMS store and exposes the six-primitive vocabulary.

    Conforms to the `agentlab.memory.Memory` protocol for `add` and `query`.
    The richer primitives below are what the chapter encourages the agent
    to use directly.
    """

    def __init__(self, store: _GMSStore) -> None:
        self._store = store

    @property
    def store(self) -> _GMSStore:
        return self._store

    # --- Six-primitive vocabulary ---------------------------------------

    def score_triple(self, head: str, relation: str, tail: str) -> float | None:
        """Geodesic plausibility of (head, relation, tail). Lower is more plausible."""
        return self._store.score_triple(head, relation, tail)

    def tension_energy(self, a: str, b: str) -> float | None:
        """Contradiction signal in [0, 2]. 0 = agreement, ~sqrt(2) = unrelated, 2 = contradiction."""
        return self._store.tension_energy(a, b)

    def check_holonomy(self, relation_path: list[str], direct_relation: str) -> float | None:
        """Path-consistency check on a multi-hop chain. ~0 means the chain is consistent."""
        return self._store.check_holonomy(relation_path, direct_relation)

    def lookup_enm(self, category: str, entity_id: str) -> float | None:
        """Exact numerical recall from the ENM register. None if not present."""
        return self._store.lookup_enm(category, entity_id)

    def link_predict(self, head: str, relation: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Rank tails by plausibility for (head, relation, ?). For exploration and routing."""
        return self._store.link_predict(head, relation, top_k=top_k)

    def query_triples(
        self,
        head: str | None = None,
        relation: str | None = None,
        tail: str | None = None,
    ) -> list[tuple[str, str, str]]:
        """Structural pattern match on the triple store."""
        return self._store.query_triples(head=head, relation=relation, tail=tail)

    # --- agentlab Memory protocol ----------------------------------------

    def add(self, item: MemoryItem) -> None:
        """Recording a MemoryItem in a GMS store requires an ingestion pipeline.

        The lean tutorial does not perform live ingestion; the store is built
        offline. If the caller wants to attach a fact to a known entity, the
        item's metadata should carry a `triple=(head, relation, tail)`; in
        that case the GMS store's add path would handle it. The adapter
        raises here to make it explicit that live training is not in scope.
        """
        triple = item.metadata.get("triple")
        if triple is None:
            raise NotImplementedError(
                "GMSMemory.add requires item.metadata['triple'] = (head, relation, tail). "
                "Document ingestion happens offline; see docgms.ingest in the GMS library."
            )
        if not hasattr(self._store, "add_triple"):
            raise NotImplementedError(
                "Underlying store has no add_triple method; "
                "live triple insertion is not supported by this adapter."
            )
        self._store.add_triple(*triple)

    def query(self, q: str, k: int = 5) -> list[MemoryItem]:
        """Substring query against the store's triples. For richer retrieval,
        use `link_predict` or `query_triples` directly."""
        matches = self._store.query_triples(head=q) or []
        items: list[MemoryItem] = []
        for h, r, t in matches[:k]:
            items.append(
                MemoryItem(
                    content=f"{h} {r} {t}",
                    kind=MemoryKind.SEMANTIC,
                    metadata={"triple": (h, r, t)},
                )
            )
        return items
