"""Cross-cutting protocols shared across the agentlab library.

These protocols define pluggable behavior used by more than one chapter.
Concrete data types (AgentState, Action, Tool, Trajectory) are defined in
their owning module and do not appear here.

If you are a worker agent extending this file: add new protocols only when
they are truly cross-cutting. Anything used in a single sub-package belongs
in that sub-package.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseLM(Protocol):
    """Minimal interface for language model adapters.

    Implementations live in agentlab.models. MockLM (Ch 2) is deterministic
    for tests; production adapters (Ch 7) wrap real provider SDKs.
    """

    def complete(self, prompt: str, **kwargs) -> str: ...

    def token_count(self, text: str) -> int: ...


@runtime_checkable
class BaseEmbedder(Protocol):
    """Minimal interface for embedding adapters.

    Used by tools/router.py and memory/vector.py.
    """

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dim(self) -> int: ...
