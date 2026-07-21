"""Hybrid memory: weighted combination of vector, graph and short-term."""

from __future__ import annotations

import math
import time

from agentlab.memory.base import MemoryItem
from agentlab.memory.graph import GraphMemory
from agentlab.memory.short_term import ShortTermMemory
from agentlab.memory.vector import VectorMemory


class HybridMemory:
    def __init__(
        self,
        vector: VectorMemory,
        graph: GraphMemory,
        short_term: ShortTermMemory,
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
        tau_seconds: float = 3600.0,
    ) -> None:
        self._vector = vector
        self._graph = graph
        self._short = short_term
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._tau = tau_seconds

    def add(self, item: MemoryItem) -> None:
        self._short.add(item)
        if item.content.strip():
            try:
                self._vector.add(item)
            except ValueError:
                pass
        self._graph.add(item)

    def query(self, q: str, k: int = 5) -> list[MemoryItem]:
        now = time.time()
        scored: dict[str, tuple[float, MemoryItem]] = {}
        for m in self._vector.query(q, k=k):
            scored[m.id] = (self._alpha, m)
        for m in self._graph.query(q, k=k):
            prev = scored.get(m.id, (0.0, m))[0]
            scored[m.id] = (prev + self._beta, m)
        for m in self._short.query(q, k=k):
            prev = scored.get(m.id, (0.0, m))[0]
            scored[m.id] = (prev + self._gamma * self._recency(now, m.timestamp), m)
        ranked = sorted(scored.values(), key=lambda x: x[0], reverse=True)
        return [m for _, m in ranked[:k]]

    def _recency(self, now: float, ts: float) -> float:
        return math.exp(-(now - ts) / self._tau)
