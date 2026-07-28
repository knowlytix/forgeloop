import hashlib

from agentlab.memory import (
    GraphMemory,
    HybridMemory,
    MemoryItem,
    MemoryKind,
    ShortTermMemory,
    Triple,
    VectorMemory,
)


class MockEmbedder:
    dim = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.lower().encode()).digest()
            v = [b / 255.0 for b in h[: self.dim]]
            out.append(v)
        return out


def _hybrid() -> HybridMemory:
    return HybridMemory(
        vector=VectorMemory(MockEmbedder()),
        graph=GraphMemory(),
        short_term=ShortTermMemory(maxlen=8),
    )


def test_add_propagates_to_backends():
    h = _hybrid()
    h.add(MemoryItem(content="overdraft fee policy", kind=MemoryKind.POLICY))
    hits = h.query("overdraft fee policy", k=3)
    assert len(hits) >= 1


def test_graph_only_item_is_findable():
    h = _hybrid()
    h.add(
        MemoryItem(
            content="alice owes bank",
            kind=MemoryKind.SEMANTIC,
            metadata={"triple": Triple("alice", "owes", "bank")},
        )
    )
    hits = h.query("alice", k=5)
    assert any("alice" in m.content for m in hits)


def test_empty_query_returns_empty():
    h = _hybrid()
    assert h.query("nothing here", k=3) == []
