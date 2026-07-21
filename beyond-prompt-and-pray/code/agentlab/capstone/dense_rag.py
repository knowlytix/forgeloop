"""Conventional dense-embedding retrieval RAG -- the benchmark baseline.

This is the deliberate foil to the GMS/GEODE triple-mediated retriever in
``policy_rag.py``. It is textbook "chunk and pray": split the policy document
into text chunks, embed each with a *frozen* sentence-transformer, retrieve the
top-k by cosine similarity, stuff them into the prompt and let the LLM answer.
There is no triple extraction, no entity binding, no geometric cap/tension/
relevance gate and no calibrated abstention -- so a benchmark of this against
``PolicyRagRetriever`` isolates exactly what the GMS machinery buys.

Two design choices keep the comparison fair rather than a strawman:

  * **Same corpus.** It reads the same ``data/banking_policy_full.md`` the GEODE
    store is built from -- not a re-derived or degraded copy.
  * **Same LLM.** Answer synthesis uses the same ``Qwen3-4B-Instruct`` via the
    shared :class:`~agentlab.models.qwen_adapter.QwenAdapter`, so the only thing
    that changes between the two systems is the *retrieval engine*.

``search`` returns the same ``[{"id", "text", "answer", "score", ...}]`` shape as
``PolicyRagRetriever.search``, so the *same* ``CapstoneTestHarness.rag_test``
oracle (which verifies answers against the GEODE store, independently of which
retriever produced them) scores both.

Chunking is configurable so the baseline can be swept:

  * ``chunking="fixed"``   -- fixed-size character windows with overlap (the
    canonical naive baseline; tables get split mid-row, which is realistically
    what naive RAG does).
  * ``chunking="section"`` -- one chunk per markdown ``##`` section (respects
    table boundaries; the strongest reasonable conventional baseline).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DOC = _REPO_ROOT / "data" / "banking_policy_full.md"
# The frozen base encoder the GEODE store also starts from -- so the baseline
# uses the *same* embedding model the GMS path tunes, isolating the tuning +
# triple-mediation as the variable, not the encoder family.
_DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

_SYSTEM = (
    "You are a bank policy assistant. Answer the customer's question using ONLY "
    "the policy excerpts provided. State the specific number (dollar amount, "
    "number of days, role) when the question asks for one. Keep the answer to "
    "one or two sentences. If the excerpts do not contain the answer, say you "
    "do not have that information."
)


def _chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size character windows with overlap -- the canonical naive scheme.

    Operates on the raw document, so a wide table can be split across a window
    boundary (mid-row) exactly as it would be in a stock RAG pipeline."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    chunks = []
    for start in range(0, max(len(text), 1), step):
        window = text[start:start + size].strip()
        if window:
            chunks.append(window)
        if start + size >= len(text):
            break
    return chunks


def _chunk_sections(text: str) -> list[str]:
    """One chunk per markdown ``##`` section (header + body).

    Any preamble before the first ``##`` (the title block) becomes its own
    chunk. This respects table boundaries -- a whole fee schedule stays in one
    chunk -- which is the fairest conventional chunking for this document."""
    chunks, current = [], []
    for line in text.splitlines():
        if line.startswith("## "):
            if current:
                block = "\n".join(current).strip()
                if block:
                    chunks.append(block)
            current = [line]
        else:
            current.append(line)
    if current:
        block = "\n".join(current).strip()
        if block:
            chunks.append(block)
    return chunks


def _section_title(chunk: str) -> str:
    """A short id for a chunk: its first markdown header, slugged, else 'chunk'."""
    for line in chunk.splitlines():
        s = line.lstrip("#").strip()
        if s:
            return s.lower().replace(" ", "_")[:48]
    return "chunk"


class DenseRagRetriever:
    """Frozen-embedding dense retrieval RAG over the policy document.

    Parameters
    ----------
    doc_path : Path | None
        The policy markdown. Defaults to the same ``banking_policy_full.md`` the
        GEODE store is built from.
    chunking : str
        ``"fixed"`` (size/overlap windows) or ``"section"`` (per ``##`` block).
    chunk_size, chunk_overlap : int
        Character window + overlap for ``chunking="fixed"`` (ignored otherwise).
    top_k : int
        Number of chunks retrieved and passed to the LLM.
    encoder_model : str
        Sentence-transformer id (frozen). Defaults to all-MiniLM-L6-v2.
    device : torch.device | None
        Defaults to CUDA when available, else CPU.
    """

    def __init__(
        self,
        doc_path: Path | None = None,
        chunking: str = "fixed",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 3,
        encoder_model: str = _DEFAULT_ENCODER,
        device: torch.device | None = None,
    ) -> None:
        if chunking not in ("fixed", "section"):
            raise ValueError(f"unknown chunking {chunking!r}; use 'fixed' or 'section'")
        from sentence_transformers import SentenceTransformer

        doc_path = Path(doc_path) if doc_path is not None else _DEFAULT_DOC
        if not doc_path.exists():
            raise FileNotFoundError(f"policy doc not found at {doc_path!s}")
        text = doc_path.read_text()

        self.chunking = chunking
        self.top_k = int(top_k)
        if chunking == "fixed":
            self.chunks = _chunk_fixed(text, chunk_size, chunk_overlap)
        else:
            self.chunks = _chunk_sections(text)
        if not self.chunks:
            raise RuntimeError("chunking produced no chunks")

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self.encoder = SentenceTransformer(encoder_model, device=str(device))
        # L2-normalize so a plain dot product IS cosine similarity. Embed the
        # corpus once at construction (the index); queries are embedded per call.
        self._matrix = self.encoder.encode(
            self.chunks, normalize_embeddings=True, convert_to_numpy=True
        )

    def _retrieve(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k (chunk_index, cosine_score), highest first."""
        q = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self._matrix @ q
        k = min(k, len(self.chunks))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    def _generate(self, query: str, context: str) -> str:
        from agentlab.models.qwen_adapter import QwenAdapter

        prompt = f"Policy excerpts:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        return QwenAdapter(
            model="Qwen/Qwen3-4B-Instruct-2507", system=_SYSTEM, max_new_tokens=256,
            device=self._device,
        ).complete(prompt, max_tokens=256).strip()

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Retrieve top-k chunks by cosine similarity and let Qwen answer from
        them. Returns a single-element list in the ``search_policy`` shape (one
        synthesized answer), matching ``PolicyRagRetriever.search``.

        Conventional baseline: it ALWAYS answers from the retrieved chunks --
        there is no calibrated gate and no abstention, so ``decision`` is always
        ``"answer"`` and ``verified`` is always ``False`` (no GMS check)."""
        hits = self._retrieve(query, k or self.top_k)
        if not hits:
            return []
        context = "\n\n".join(self.chunks[i] for i, _ in hits)
        answer = self._generate(query, context)
        top_idx, top_score = hits[0]
        return [{
            "id": _section_title(self.chunks[top_idx]),
            "text": context,
            "answer": answer,
            "score": float(top_score),
            "decision": "answer",
            "route": f"dense:{self.chunking}",
            "verified": False,
            "notice": "",
        }]
