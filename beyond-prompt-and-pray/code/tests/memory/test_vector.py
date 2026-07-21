import hashlib

import pytest

from agentlab.memory import MemoryItem, MemoryKind, VectorMemory, chunk_text


class MockEmbedder:
    dim = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.lower().encode()).digest()
            v = [b / 255.0 for b in h[: self.dim]]
            out.append(v)
        return out


def test_chunk_text_basic():
    text = "a" * 1000
    chunks = chunk_text(text, window=400, overlap=50)
    assert len(chunks) >= 3
    assert all(len(c) <= 400 for c in chunks)


def test_chunk_text_short():
    chunks = chunk_text("short", window=400, overlap=50)
    assert chunks == ["short"]


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", window=10, overlap=10)


def test_vector_add_and_query():
    vm = VectorMemory(MockEmbedder())
    vm.add(MemoryItem(content="the bank charges overdraft fees", kind=MemoryKind.SEMANTIC))
    vm.add(MemoryItem(content="recipes for chocolate cake", kind=MemoryKind.SEMANTIC))
    hits = vm.query("the bank charges overdraft fees", k=1)
    assert len(hits) == 1
    assert "bank" in hits[0].content


def test_vector_rejects_empty():
    vm = VectorMemory(MockEmbedder())
    with pytest.raises(ValueError):
        vm.add(MemoryItem(content="   ", kind=MemoryKind.SEMANTIC))


def test_vector_empty_query():
    vm = VectorMemory(MockEmbedder())
    assert vm.query("anything") == []
