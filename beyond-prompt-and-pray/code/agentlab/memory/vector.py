"""In-process vector memory with pluggable embedder.

Chunking is deliberately simple — character window with overlap. Real
chunking strategies (recursive, semantic, structural) are a topic for the
notebook discussion, not the library.
"""

from __future__ import annotations

import numpy as np

from agentlab.memory.base import MemoryItem
from agentlab.protocols import BaseEmbedder


def chunk_text(text: str, window: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping fixed-width character windows.

    Args:
        text: The text to split.
        window: Number of characters per chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        The list of chunks, empty when text is empty.

    Raises:
        ValueError: If overlap is not smaller than window.
    """
    if overlap >= window:
        raise ValueError("overlap must be smaller than window")
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + window
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


class VectorMemory:
    """In-memory store that embeds item content and retrieves by cosine similarity."""

    def __init__(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder
        self._items: list[MemoryItem] = []
        self._vecs: list[np.ndarray] = []

    def add(self, item: MemoryItem) -> None:
        """Embed the item's content and store it.

        Raises:
            ValueError: If the item content is empty or whitespace only.
        """
        if not item.content.strip():
            raise ValueError("cannot add empty content")
        v = self._embedder.embed([item.content])[0]
        self._items.append(item)
        self._vecs.append(np.asarray(v, dtype=float))

    def query(self, q: str, k: int = 5) -> list[MemoryItem]:
        """Return up to k stored items ranked by cosine similarity to the query.

        Args:
            q: The query text to embed and compare.
            k: Maximum number of items to return.

        Returns:
            The top-k items by similarity, or an empty list when nothing is stored.
        """
        if not self._items:
            return []
        qv = np.asarray(self._embedder.embed([q])[0], dtype=float)
        scored = [(_cosine(qv, v), i) for i, v in enumerate(self._vecs)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._items[i] for _, i in scored[:k]]

    def __len__(self) -> int:
        return len(self._items)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
