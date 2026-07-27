"""Local Qwen as a knowlytix ``LLMBackend`` for offline GMS ingestion.

``knowlytix.knowledge.ingest`` runs LLM-assisted triple extraction in its
``hybrid`` / ``llm_only`` ingest modes, and expects a
``knowlytix.knowledge.llm_backend.LLMBackend`` (``call(system, user,
max_tokens) -> str``). The shipped default resolves to a hosted API model; this
backend routes that contract to the book's local Qwen3-4B instead, so building
a GMS *ground truth* from a document stays offline and deterministic (greedy
decoding) -- the same model the agent and the test stand already use.

Use it to build the answer-key GMS the RAG test verifies against: a regex-only
ingest yields a thin ground truth, so the retriever's claims fail to verify for
lack of facts rather than for being unfaithful. LLM-assisted ingest is the right
tool *here* (unlike runtime evidence matching) because building a ground-truth
graph from a document has no runtime message to be blind to.
"""

from __future__ import annotations

from knowlytix.knowledge.llm_backend import LLMBackend


class QwenLLMBackend(LLMBackend):
    """``LLMBackend`` backed by the local Qwen3-4B adapter.

    ``call`` concatenates the system instruction and user content into one
    greedy completion, mirroring how the other local-Qwen shims in this package
    bridge a ``messages`` interface to :meth:`QwenAdapter.complete`.
    """

    def __init__(self, max_new_tokens: int = 2048) -> None:
        from forgeloop.agents.models import QwenAdapter

        self._qwen = QwenAdapter(max_new_tokens=max_new_tokens)

    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Concatenate the system and user text into one greedy Qwen completion.

        Args:
            system: System instruction, prepended when non-empty.
            user: User content.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated completion text.
        """
        prompt = f"{system}\n\n{user}" if system else user
        return self._qwen.complete(prompt, max_tokens=max_tokens)

    @property
    def model_name(self) -> str:
        return "qwen2.5-3b-local"
