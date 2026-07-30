# SPDX-License-Identifier: Apache-2.0
"""Build notebooks/appendix_E_chunk_and_pray_baseline.ipynb (App E) with nbformat.

App E is the *reference* baseline: a from-scratch "Chunk and Pray" RAG --
fixed-size overlapping chunks, sentence-transformers embeddings, cosine top-k
retrieval, and a generator that answers from the retrieved passages with no
binding, no Exact Numerical Memory, no abstention and no verification. It runs
the SAME data/eval_cohort.json the GEODE-RAG pipeline is graded on
(Chapter 13), so the contrast is empirical rather than rhetorical.

This builder only assembles the .ipynb JSON; it executes nothing. The notebook
itself runs on CPU for everything except the optional, CI-marked real-Qwen cell.
The deterministic self-check needs no GPU: it asserts the *architectural*
failures of chunk-and-pray (it never abstains; provenance is chunk-level; it is
confidently wrong on prose the corpus never triplified) that hold regardless of
which generator is plugged in.

Run:  python scripts/build_nb_appendix_E.py
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "notebooks",
                   "appendix_E_chunk_and_pray_baseline.ipynb")

# ---------------------------------------------------------------------------
# Cell sources. Each is copied verbatim into appendix_E_chunk_and_pray_baseline.tex.
# ---------------------------------------------------------------------------

BOOTSTRAP = (
    "import os, sys\n"
    'KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", "")\n'
    "sys.path.insert(0, KNOWLYTIX_SRC)\n"
    "REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir))"
)

LOAD_CORPUS = '''\
# The baseline reads the SAME corpus and the SAME labeled cohort the GEODE-RAG
# pipeline is graded on (Chapter 13). Nothing about the data changes; only the
# retrieval contract does. There is no trained store here -- chunk-and-pray needs
# only raw text and an embedding model.
import json

CORPUS = os.path.join(REPO_ROOT, "data", "annual_report.md")
COHORT = os.path.join(REPO_ROOT, "data", "eval_cohort.json")

report_md = open(CORPUS, encoding="utf-8").read()
cohort = json.loads(open(COHORT, encoding="utf-8").read())
print(f"corpus: {len(report_md)} chars, {report_md.count(chr(10)) + 1} lines")
print(f"cohort: {len(cohort)} cases; types:",
      sorted({r["type"] for r in cohort}))'''

CHUNKER = '''\
# Step 1 -- chunk and pray. The canonical splitter: slide a fixed-size window of
# WORDS over the document with a fixed OVERLAP. This is deliberately blind to
# structure -- it does not know a markdown table from a paragraph -- so it cuts
# across table rows and section boundaries. That is the whole point: the retrieval
# unit becomes a span of text, not an asserted fact.
from dataclasses import dataclass

CHUNK_WORDS = 60          # window size, in whitespace tokens
OVERLAP_WORDS = 15        # stride = CHUNK_WORDS - OVERLAP_WORDS


@dataclass
class Chunk:
    id: int
    text: str
    char_start: int
    char_end: int

    @property
    def location(self) -> str:
        # Chunk-level provenance: an offset range over the file, NOT a cell.
        # Contrast the GEODE-RAG span ":15:553-558" (file:line:char of one cell).
        return f"chunk#{self.id}:{self.char_start}-{self.char_end}"


def chunk_text(text: str, size: int = CHUNK_WORDS,
               overlap: int = OVERLAP_WORDS) -> list[Chunk]:
    # Tokenize on whitespace, keeping each token's char offset so a chunk can
    # report *where* in the file it came from -- the best provenance a chunker has.
    toks, offsets, i = [], [], 0
    for tok in text.split(" "):
        offsets.append(i)
        toks.append(tok)
        i += len(tok) + 1
    chunks, start, cid = [], 0, 0
    stride = max(1, size - overlap)
    while start < len(toks):
        window = toks[start:start + size]
        cstart = offsets[start]
        cend = offsets[min(start + size, len(toks)) - 1] + len(toks[min(start + size, len(toks)) - 1])
        chunks.append(Chunk(cid, " ".join(window), cstart, cend))
        cid += 1
        start += stride
    return chunks


chunks = chunk_text(report_md)
print(f"{len(chunks)} chunks of <= {CHUNK_WORDS} words (overlap {OVERLAP_WORDS})")

# Show the structural damage: find the chunk holding the "Total" segment row and
# check whether the four segment rows that must sum to it are in the SAME chunk.
total_chunks = [c for c in chunks if "| Total | All |" in c.text]
seg_chunks = [c for c in chunks if "Cloud Platform" in c.text]
print("Total row in chunk(s):", [c.id for c in total_chunks])
print("Cloud Platform in chunk(s):", [c.id for c in seg_chunks])
print("segment table split across chunks:",
      {c.id for c in total_chunks} != {c.id for c in seg_chunks})'''

EMBED = '''\
# Step 2 -- embed and index. all-MiniLM-L6-v2 is the textbook chunk-and-pray
# encoder: 384-dim sentence embeddings, ~80MB, no fine-tuning. We embed every
# chunk once (the one heavy offline step) and L2-normalize so cosine similarity
# is a dot product. No FAISS here -- the corpus is tiny, so a numpy matrix and
# scikit-learn cosine are exact; FAISS is the drop-in at scale.
import numpy as np
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunk_emb = encoder.encode([c.text for c in chunks],
                           normalize_embeddings=True,
                           show_progress_bar=False)
chunk_emb = np.asarray(chunk_emb, dtype=np.float32)
print("chunk embedding matrix:", chunk_emb.shape)'''

RETRIEVE = '''\
# Step 3 -- top-k retrieval. Cosine similarity between the query embedding and
# every chunk embedding; return the k nearest. Relevance IS cosine distance --
# there is no notion of whether the chunk actually *holds* the asked-for fact.
from sklearn.metrics.pairwise import cosine_similarity

TOP_K = 4


def retrieve(query: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
    q = encoder.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(np.asarray(q, dtype=np.float32), chunk_emb)[0]
    order = np.argsort(-sims)[:k]
    return [(chunks[i], float(sims[i])) for i in order]


# The exact-number trap: ask for total revenue and look at what surfaces. The top
# chunk is a passage of the report -- it contains 355.0, but also 320.0 (FY2024),
# 200.0, 95.0 ... Several candidate numbers co-occur; nothing marks which is THE
# answer. A chunk is not a cell.
for c, s in retrieve("What was total revenue?"):
    nums = sorted({t for t in c.text.replace("|", " ").split()
                   if t.replace(".", "", 1).isdigit()})
    print(f"  cos={s:.3f} {c.location}  numbers={nums}")'''

GENERATE = '''\
# Step 4 -- stuff and generate. Concatenate the top-k chunks into the prompt and
# ask the model to answer from them. This is the "pray" step: we trust the model
# to pick the right number out of the passages and not to invent one. There is
# NO abstention path -- the pipeline always returns an answer -- NO Exact
# Numerical Memory and NO verification of the answer against structured facts.
from dataclasses import dataclass, field


@dataclass
class BaselineAnswer:
    answer: str
    decision: str                      # always "accept" -- chunk-and-pray cannot abstain
    sources: list                      # the retrieved Chunks (chunk-level provenance)
    contexts: list = field(default_factory=list)


def make_prompt(question: str, ctx: list[Chunk]) -> str:
    passages = "\\n\\n".join(f"[{c.id}] {c.text}" for c in ctx)
    return (f"Answer the question using only the passages below.\\n\\n"
            f"{passages}\\n\\nQuestion: {question}\\nAnswer:")


class BaselineRAG:
    """Chunk-and-pray: retrieve top-k, stuff, generate. No abstention, no ENM,
    no verification. `generate(prompt) -> str` is the only pluggable part."""

    def __init__(self, generate):
        self._generate = generate

    def query(self, question: str, k: int = TOP_K) -> BaselineAnswer:
        hits = retrieve(question, k)
        ctx = [c for c, _ in hits]
        text = self._generate(make_prompt(question, ctx))
        # The pipeline has no branch that returns "I don't know": decision is
        # ALWAYS accept. That is the architectural choice this appendix measures.
        return BaselineAnswer(answer=text.strip(), decision="accept",
                              sources=ctx, contexts=ctx)


# Deterministic CPU generator (no GPU): an extractive reader that returns the
# single number from the retrieved passages closest to the question -- a faithful,
# reproducible stand-in for "the model copies a figure out of a chunk". The real
# Qwen generator is the next, CI-marked cell; the architectural conclusions are
# the same either way.
import re


def extractive_generate(prompt: str) -> str:
    q = set(re.findall(r"[a-z]+", prompt.rsplit("Question:", 1)[-1].lower()))
    passages = prompt.split("Question:", 1)[0]
    # Pick the PASSAGE line (they start with "[id] ") whose words best overlap
    # the question, then the first number on it -- exactly the brittle "read a
    # value off the nearest text" behavior. Instruction scaffolding is skipped.
    best, best_score = "", -1
    for line in passages.splitlines():
        if not line.strip().startswith("["):
            continue
        words = set(re.findall(r"[a-z]+", line.lower()))
        score = len(words & q)
        if score > best_score:
            best, best_score = line, score
    # Prefer a proper decimal figure (e.g. 355.0) over a bare integer like the
    # "3" in a "## 3." header -- the kind of brittle heuristic a hand-rolled
    # extractive reader needs and still gets wrong on a co-occurring number.
    dec = re.findall(r"\\d+\\.\\d+", best)
    ints = re.findall(r"\\d+", best)
    if dec:
        return dec[0]
    return ints[0] if ints else best.split("] ", 1)[-1].strip()[:80]


baseline = BaselineRAG(extractive_generate)
demo = baseline.query("What was total revenue?")
print("answer  :", demo.answer)
print("decision:", demo.decision)
print("sources :", [c.location for c in demo.sources])'''

QWEN_CI = '''\
# CI ONLY -- the real chunk-and-pray run on local Qwen2.5-3B-Instruct (GPU).
# Same retrieval, same prompt; only the generator changes. This is the generator
# a real chunk-and-pray deployment ships. The lead runs this in CI; it is NOT
# executed during authoring. The cohort numbers reported in the appendix come
# from this run.
from knowlytix.knowledge.geode import QWEN_3B
from knowlytix.knowledge.llm_backend import LocalTransformersBackend

qwen = LocalTransformersBackend(QWEN_3B, device="cuda")


def qwen_generate(prompt: str) -> str:
    return qwen.call(system="You are a helpful financial-report assistant. "
                            "Answer concisely from the passages.",
                     user=prompt, max_tokens=64)


qwen_baseline = BaselineRAG(qwen_generate)
print(qwen_baseline.query("What was total revenue?").answer)'''

TRUST = '''\
# CI -- run the cohort through the REAL (Qwen) baseline and compute the SAME
# trust metrics as Chapter 13: provenance rate, abstention precision and the
# confident-wrong count, plus the accept rate. The grader is byte-identical to
# Chapter 13; only the system under test differs. (The extractive `baseline`
# above gives the same architectural verdict on CPU; see the self-check.)
def trust_metrics(rag, rows):
    confident_wrong = 0
    with_prov = answered = 0
    abst_correct = abst_total = 0
    per_case = []
    for r in rows:
        ans = rag.query(r["question"])
        accepted = ans.decision == "accept"
        exp = r.get("expected_answer")
        # case-insensitive substring match (cohort labels are lowercased).
        hit = bool(exp) and exp.lower() in ans.answer.lower()
        if accepted:
            answered += 1
            if any(c.location for c in ans.sources):
                with_prov += 1
        if ans.decision == "abstain":
            abst_total += 1
            if r.get("expect_decision") == "abstain":
                abst_correct += 1
        # confident-wrong: accepted with the expected value missing, OR accepted
        # on a question whose label says abstain (the prose blind spots).
        if accepted and exp is not None and not hit:
            confident_wrong += 1
        if accepted and r.get("expect_decision") == "abstain":
            confident_wrong += 1
        per_case.append((r["id"], r["type"], ans.decision, hit, ans.answer[:40]))
    n = len(rows)
    return {
        "accept_rate": answered / n,
        "abstain_count": abst_total,
        "provenance_rate": with_prov / answered if answered else 0.0,
        "abstention_precision": abst_correct / abst_total if abst_total else None,
        "confident_wrong": confident_wrong,
    }, per_case


metrics, per_case = trust_metrics(qwen_baseline, cohort)
print("Baseline (chunk-and-pray, Qwen) trust report:")
for k, v in metrics.items():
    print(f"  {k:22} {v}")
print()
for cid, typ, dec, hit, ans in per_case:
    print(f"  {cid:16} {typ:13} {dec:8} hit={hit!s:5} {ans!r}")'''

CONTRAST = '''\
# The chunk-level provenance is real but not verifiable. Every accepted answer
# carries a source (provenance_rate = 1.0), yet the source is a 60-word window,
# not a cell, and nothing ties the answer's number to an authoritative figure.
# Compare the two provenance shapes side by side.
demo = qwen_baseline.query("What is Cloud Platform revenue?")
print("baseline source :", demo.sources[0].location,
      "  (a", CHUNK_WORDS, "-word window spanning many cells)")
print("GEODE-RAG source: data/annual_report.md:15:553-558",
      "  (the exact cell 120.0 resolves to)")

# The two prose blind spots (Outlook, Risk Factors) carry no triples; GEODE-RAG
# abstains on them (Chapter 13, abstention_precision = 1.0). Chunk-and-pray has
# no abstention path, so it answers anyway -- a confident-wrong on each.
prose = [r for r in cohort if r["type"] == "unanswerable"]
for r in prose:
    a = qwen_baseline.query(r["question"])
    print(f"{r['id']}: decision={a.decision!r} (GEODE-RAG abstains) answer={a.answer[:48]!r}")'''

SELF_CHECK = '''\
# Self-check (CPU, deterministic): the appendix's claim is architectural, so it
# holds for ANY generator. (1) chunk-and-pray never abstains; (2) provenance is
# chunk-level, never a file:line:char cell span; (3) it is confidently wrong on
# every prose blind spot the corpus never triplified -- at least the two
# unanswerable cases GEODE-RAG correctly abstains on.
metrics, _ = trust_metrics(baseline, cohort)

# (1) no abstention path: every case is accepted.
assert metrics["accept_rate"] == 1.0, metrics
assert metrics["abstain_count"] == 0, metrics

# (2) provenance exists but is chunk-level, not a cell span.
loc = baseline.query("What was total revenue?").sources[0].location
assert loc.startswith("chunk#") and ":" in loc
assert "annual_report.md:" not in loc       # never a file:line:char cell span

# (3) confident-wrong on the prose blind spots GEODE-RAG abstains on.
n_unanswerable = sum(1 for r in cohort if r["type"] == "unanswerable")
assert metrics["confident_wrong"] >= n_unanswerable, metrics

print("App E self-check passed: chunk-and-pray never abstains, gives only "
      "chunk-level provenance, and is confidently wrong on prose blind spots.")
print("confident_wrong =", metrics["confident_wrong"],
      " accept_rate =", metrics["accept_rate"])'''


def build() -> None:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# Appendix E --- A Runnable \"Chunk and Pray\" Baseline\n"
            "\n"
            "This appendix builds the system the book argues against, end to end, "
            "and runs it through the **same** labeled cohort the GEODE-RAG "
            "pipeline is graded on (Chapter 13). It is a yardstick, not a straw "
            "man: a textbook dense-retrieval RAG --- fixed-size chunks, "
            "sentence-transformers embeddings, cosine top-$k$, stuff-and-generate "
            "--- with no triple binding, no Exact Numerical Memory, no abstention "
            "and no verification. The contrast Chapter 1 and Appendix B describe "
            "in prose becomes a table of numbers here."
        ),
        new_code_cell(BOOTSTRAP),
        new_markdown_cell("## The corpus and the cohort (unchanged)"),
        new_code_cell(LOAD_CORPUS),
        new_markdown_cell("## Step 1 --- chunk the document (structure-blind)"),
        new_code_cell(CHUNKER),
        new_markdown_cell("## Step 2 --- embed and index"),
        new_code_cell(EMBED),
        new_markdown_cell("## Step 3 --- top-$k$ retrieval"),
        new_code_cell(RETRIEVE),
        new_markdown_cell("## Step 4 --- stuff and generate (no abstention)"),
        new_code_cell(GENERATE),
        new_markdown_cell("## The real generator: local Qwen (CI)"),
        new_code_cell(QWEN_CI),
        new_markdown_cell("## The cohort, scored exactly as in Chapter 13"),
        new_code_cell(TRUST),
        new_markdown_cell("## Provenance and the prose blind spots"),
        new_code_cell(CONTRAST),
        new_markdown_cell("## Self-check"),
        new_code_cell(SELF_CHECK),
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
