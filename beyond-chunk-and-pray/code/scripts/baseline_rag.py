# SPDX-License-Identifier: Apache-2.0
"""Traditional 'chunk and pray' RAG baseline (Appendix E), importable.

Fixed-size overlapping word-window chunks -> MiniLM sentence embeddings -> cosine
top-k retrieval -> stuff the passages into the prompt and generate. No trained
store, no ENM, no verification, and NO abstention path (it always answers). This
is the standard pipeline the book measures the GEODE-RAG against; here it is
wrapped so the DoE test can run the SAME cohort through it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CHUNK_WORDS = 60
OVERLAP_WORDS = 15
TOP_K = 4
_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Chunk:
    id: int
    text: str
    char_start: int
    char_end: int

    @property
    def location(self) -> str:
        return f"chunk#{self.id}:{self.char_start}-{self.char_end}"


def chunk_text(text: str, size: int = CHUNK_WORDS,
               overlap: int = OVERLAP_WORDS) -> list[Chunk]:
    toks, offsets, i = [], [], 0
    for tok in text.split(" "):
        offsets.append(i)
        toks.append(tok)
        i += len(tok) + 1
    chunks, start, cid = [], 0, 0
    stride = max(1, size - overlap)
    while start < len(toks):
        end = min(start + size, len(toks))
        cstart = offsets[start]
        cend = offsets[end - 1] + len(toks[end - 1])
        chunks.append(Chunk(cid, " ".join(toks[start:start + size]), cstart, cend))
        cid += 1
        start += stride
    return chunks


@dataclass
class BaselineAnswer:
    answer: str
    decision: str                       # always "accept" -- chunk-and-pray cannot abstain
    sources: list = field(default_factory=list)   # retrieved Chunks


def make_prompt(question: str, ctx: list[Chunk]) -> str:
    passages = "\n\n".join(f"[{c.id}] {c.text}" for c in ctx)
    return (f"Answer the question using only the passages below.\n\n"
            f"{passages}\n\nQuestion: {question}\nAnswer:")


class BaselineRAG:
    """Chunk-and-pray: retrieve top-k by cosine, stuff, generate. ``generate``
    is ``Callable[[str], str]`` (e.g. a Qwen wrapper)."""

    def __init__(self, text: str, generate, *, k: int = TOP_K,
                 encoder_name: str = _ENCODER, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.chunks = chunk_text(text)
        self.encoder = SentenceTransformer(encoder_name, device=device)
        self.emb = np.asarray(
            self.encoder.encode([c.text for c in self.chunks],
                                normalize_embeddings=True), dtype=np.float32)
        self._generate = generate
        self.k = k

    def retrieve(self, query: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        from sklearn.metrics.pairwise import cosine_similarity
        k = k or self.k
        q = np.asarray(self.encoder.encode([query], normalize_embeddings=True),
                       dtype=np.float32)
        sims = cosine_similarity(q, self.emb)[0]
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in order]

    def query(self, question: str, k: int | None = None) -> BaselineAnswer:
        ctx = [c for c, _ in self.retrieve(question, k)]
        text = (self._generate(make_prompt(question, ctx)) or "").strip()
        return BaselineAnswer(answer=text, decision="accept", sources=ctx)
