"""Three router families: rule-based, embedding-based and LLM-based.

All three conform to the Router protocol. Pick the simplest one that works
for your domain. Rules are transparent but brittle; embeddings generalize
but need calibration; LLM routing is flexible but adds latency and cost.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from forgeloop.agents.protocols import BaseEmbedder, BaseLM
from forgeloop.agents.tools.base import Tool
from forgeloop.agents.tools.registry import ToolRegistry


@runtime_checkable
class Router(Protocol):
    def route(self, query: str, registry: ToolRegistry) -> Tool | None:
        """Select the tool best matching the query, or None if none matches."""
        ...


class RuleRouter:
    """Routes by matching keywords in the query to tool names."""

    def __init__(self, keywords: dict[str, str]) -> None:
        self._keywords = keywords

    def route(self, query: str, registry: ToolRegistry) -> Tool | None:
        """Return the tool for the first keyword found in the query, or None.

        Args:
            query: The user query to match against the keyword map.
            registry: The registry used to resolve matched tool names.

        Returns:
            The matched Tool, or None if no keyword matches a registered tool.
        """
        ql = query.lower()
        for kw, tool_name in self._keywords.items():
            if kw.lower() in ql:
                try:
                    return registry.get(tool_name)
                except KeyError:
                    continue
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class EmbeddingRouter:
    """Routes by cosine similarity between the query and tool descriptions."""

    def __init__(self, embedder: BaseEmbedder, threshold: float = 0.5) -> None:
        self._embedder = embedder
        self._threshold = threshold

    def route(self, query: str, registry: ToolRegistry) -> Tool | None:
        """Return the tool whose description is most similar to the query.

        Args:
            query: The user query to embed and compare.
            registry: The registry supplying candidate tools.

        Returns:
            The highest-scoring Tool, or None when there are no tools or the top
            cosine similarity is below the threshold.
        """
        tools = registry.all()
        if not tools:
            return None
        descs = [t.description for t in tools]
        embedded = self._embedder.embed([query] + descs)
        qv, tvs = embedded[0], embedded[1:]
        best_score = -1.0
        best_tool: Tool | None = None
        for tv, t in zip(tvs, tools):
            s = _cosine(qv, tv)
            if s > best_score:
                best_score = s
                best_tool = t
        if best_score < self._threshold:
            return None
        return best_tool


class LMRouter:
    """Routes by asking a language model to name the matching tool."""

    def __init__(self, lm: BaseLM) -> None:
        self._lm = lm

    def route(self, query: str, registry: ToolRegistry) -> Tool | None:
        """Prompt the LM with the tool listing and return the tool it names.

        Args:
            query: The user query included in the prompt.
            registry: The registry supplying the tool listing and resolving the
                LM's chosen name.

        Returns:
            The named Tool, or None when there are no tools, the LM replies
            "NONE" or the reply does not match a registered tool.
        """
        tools = registry.all()
        if not tools:
            return None
        listing = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        prompt = (
            f"Available tools:\n{listing}\n\n"
            f"User query: {query}\n\n"
            "Reply with the single tool name (no other text) that best matches the query, "
            "or NONE if no tool matches."
        )
        response = self._lm.complete(prompt).strip()
        if response == "NONE":
            return None
        try:
            return registry.get(response)
        except KeyError:
            return None
