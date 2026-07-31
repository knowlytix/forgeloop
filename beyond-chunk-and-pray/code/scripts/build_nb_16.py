# SPDX-License-Identifier: Apache-2.0
"""Builder for notebooks/16_pluggable_llms_and_dense_fallback_a_inline.ipynb.

CPU-only: emits a valid nbformat-4 notebook via the ``nbformat`` package. Does
NOT execute any cell (no store load, no Qwen). Run:  python scripts/build_nb_14.py
"""

from __future__ import annotations

import os

import nbformat as nbf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "notebooks",
                   "16_pluggable_llms_and_dense_fallback_a_inline.ipynb")

nb = nbf.v4.new_notebook()
cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# Ch14 — Pluggable LLMs and the distrusted dense fallback

A GEODE-RAG deployment has three LLM-shaped jobs: **extract** the query triples,
**synthesize** the grounded answer, and (optionally) **verify** the answer's own
claims. This chapter shows how `RagConfig` lets you assign a *different* backend
to each role, and then turns to the one place the book makes a deliberate
exception to its own thesis: the **dense vector fallback**. Dense retrieval is
off by default, opt-in, and — when it does answer — quarantined with
`verified=False` and a notice, because its hits are not GMS-verified.

The grounded triple path always abstains on a prose-only question
(see Ch11). The dense fallback is the *only* way to get a (caveated) answer to
such a question, and `strict_mode` lets a bank-grade deployment turn even that
off. Every claim below is grounded in `data/corpus_facts.md`; the Qwen/store
cells are marked **CI** for the lead to execute.
""")

# --- bootstrap (FIRST cell) ---
code(r"""
import os, sys
KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", "/path/to/GMS-knowlytix")
sys.path.insert(0, KNOWLYTIX_SRC)
""")

md(r"""
## 1. The three roles, one config

`RagConfig` carries three backend slots. Only `llm` (synthesis) is required;
`llm_extract` falls back to `llm`, and `llm_verify` falls back to `llm_extract`.
This is how you mix a small, cheap extractor with a stronger synthesizer, or
pin every role to the same local Qwen for a fully offline deployment.
""")

code(r"""
from knowlytix.knowledge.rag import RagConfig
from knowlytix.knowledge.llm_backend import LLMBackend


class ScriptedBackend(LLMBackend):
    # Deterministic fake backend for CI -- returns a fixed reply.
    # The library accepts any LLMBackend, so a scripted one makes the
    # notebook reproducible without a GPU. The real Qwen path is shown below.

    def __init__(self, name: str, reply: str = ""):
        self._name = name
        self._reply = reply

    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
        return self._reply

    @property
    def model_name(self) -> str:
        return self._name


# Per-role assignment: a different backend for extract / synthesize / verify.
cfg = RagConfig(
    llm=ScriptedBackend("synth"),
    llm_extract=ScriptedBackend("extract"),
    llm_verify=ScriptedBackend("verify"),
)
print("synthesize :", cfg.llm.model_name)
print("extract    :", cfg.extract_llm().model_name)
print("verify     :", cfg.verify_llm().model_name)
""")

md(r"""
The resolution helpers `extract_llm()` and `verify_llm()` implement the
fallback chain. Drop a slot and it inherits from the one above it:
""")

code(r"""
# One backend for synthesis + extraction; verify inherits the extractor.
shared = ScriptedBackend("qwen-shared")
cfg2 = RagConfig(llm=shared)
assert cfg2.extract_llm() is shared      # llm_extract defaults to llm
assert cfg2.verify_llm() is shared       # llm_verify defaults to extract_llm()
print("all roles ->", cfg2.verify_llm().model_name)
""")

md(r"""
### The real Qwen path (CI)

For a real deployment every role is a local Qwen3-4B-Instruct. The agent
runtime is **Qwen-locked** (see Ch5 and the EXTRACTION_AGENT_DESIGN note); the
same model serves all three roles here. This cell needs the GPU — the lead runs
it in CI.
""")

code(r"""
# CI ONLY -- loads Qwen on GPU. Do not run during authoring.
from knowlytix.knowledge.llm_backend import LocalTransformersBackend
from knowlytix.knowledge.geode import QWEN_3B

qwen = LocalTransformersBackend(QWEN_3B, device="cuda")
qwen_cfg = RagConfig(llm=qwen)        # all three roles -> local Qwen
assert qwen_cfg.extract_llm().model_name == QWEN_3B
""")

md(r"""
## 2. Loading the trained store

The pipeline runs over the store F2 built. We load it from disk rather than
rebuild it (rebuilding runs GEODE + training on the GPU). This cell is marked
**CI**; the listings that follow describe the grounded behaviour from
`data/corpus_facts.md`.
""")

code(r"""
# CI ONLY -- loads the trained store from disk (GPU for the model tensors).
import torch
from knowlytix.knowledge.store import GMSExpertStore
from knowlytix.knowledge.config import DocGMSConfig

STORE = os.path.join(os.path.dirname(os.getcwd()), "data", "gms_annual_report_store")
store = GMSExpertStore(DocGMSConfig(store_path=STORE),
                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
assert store.load(), "store not found -- run scripts/build_store.py first"
print("loaded store at", store.store_path)
""")

md(r"""
## 3. Default: a prose question abstains

Section 6 (Management Discussion and Analysis), Section 7 (Risk Factors), and
Section 8 (Outlook) of the annual report carry **no triples** — they are
coverage blind spots (`coverage_ratio = 0.56` in `corpus_facts.md`). A question
that can only be answered from that prose has nothing to bind to, so the default
triple-mediated pipeline **abstains**. Nothing is fabricated.
""")

code(r"""
from knowlytix.knowledge.rag import RagPipeline

# Default config: dense_fallback is False (the bank-grade default).
default_cfg = RagConfig(llm=ScriptedBackend("synth"))
assert default_cfg.dense_fallback is False
assert default_cfg.strict_mode is False
""")

code(r"""
# CI ONLY -- runs the pipeline against the loaded store.
PROSE_Q = "What does management say about customer concentration risk?"

pipe_default = RagPipeline(store, default_cfg)
ans_default = pipe_default.query(PROSE_Q)

# The MD&A prose has no triples to bind, so the default route abstains.
assert ans_default.decision == "abstain"
assert ans_default.route == "triple"
assert ans_default.verified is True          # an abstention is still verified-true
print(ans_default.notice)                    # bind-check abstention notice
""")

md(r"""
## 4. Opt-in dense fallback — answered, but quarantined

Set `dense_fallback=True` and the pipeline indexes the document's paragraph
spans (`build_spans`) into an `InMemoryVectorBackend`. When nothing binds, it
searches that index and synthesizes from the raw passages — but the answer is
flagged `verified=False`, routed as `dense_fallback`, and carries the notice
*"Answer from dense fallback; NOT GMS-verified."* This is the book's distrust
thesis made operational (Ch1): a dense hit is a lead, never proof.
""")

code(r"""
# Same prose question, dense fallback enabled.
dense_cfg = RagConfig(
    llm=ScriptedBackend(
        "synth",
        reply=("Management states it does not consider any single customer "
               "material to consolidated results."),
    ),
    dense_fallback=True,        # opt in to the distrusted path
    strict_mode=False,          # actually return the (flagged) dense answer
    query_parse_mode="llm",     # scripted LLM returns prose -> no triples -> dense path
)
assert dense_cfg.dense_fallback is True
assert dense_cfg.vector_backend is None     # defaults to InMemoryVectorBackend
""")

code(r"""
# CI ONLY -- dense fallback answers the prose question, flagged unverified.
pipe_dense = RagPipeline(store, dense_cfg)
ans_dense = pipe_dense.query(PROSE_Q)

assert ans_dense.route == "dense_fallback"
assert ans_dense.verified is False                       # quarantined
assert ans_dense.notice == "Answer from dense fallback; NOT GMS-verified."
assert ans_dense.dense_sources                           # the spans it used
# The dense answer points at MD&A prose (section 6), a known blind spot.
print(ans_dense.dense_sources[0].location)
print(ans_dense.answer)
""")

md(r"""
## 5. strict_mode — refuse even the dense answer

A bank-grade deployment may decide that an unverifiable answer is worse than no
answer. `strict_mode=True` keeps the dense index wired but **downgrades the
dense route back to an abstention** — you get the coverage signal without ever
returning an unverified figure.
""")

code(r"""
strict_cfg = RagConfig(
    llm=ScriptedBackend("synth"),
    dense_fallback=True,
    strict_mode=True,           # dense route -> abstain
)
assert strict_cfg.dense_fallback is True
assert strict_cfg.strict_mode is True
""")

code(r"""
# CI ONLY -- strict mode abstains on the prose question despite dense being on.
pipe_strict = RagPipeline(store, strict_cfg)
ans_strict = pipe_strict.query(PROSE_Q)

assert ans_strict.decision == "abstain"
assert ans_strict.route == "triple"          # never reaches the dense route
assert ans_strict.verified is True
print(ans_strict.notice)
""")

md(r"""
## 6. Pointing the fallback at your own backend

The dense index is pluggable: implement `VectorBackend` (two methods, `index`
and `search`) and pass it as `vector_backend`. The library default is
`InMemoryVectorBackend` with an injectable encoder, so the fallback is
offline-testable and needs no external service. Below we build the in-memory
backend directly over the report's spans — the same machinery the pipeline uses
internally — to show the contract.
""")

code(r"""
from knowlytix.knowledge.rag import InMemoryVectorBackend, build_spans
from knowlytix.knowledge.rag.dense import VectorBackend, DenseSpan

# build_spans chunks markdown into paragraph spans with line-range provenance.
report_md = "## 6. Management Discussion and Analysis\n\nManagement does not consider any single customer to be material.\n"
spans = build_spans(report_md, source_path="data/annual_report.md")
assert spans and spans[0].location.startswith("data/annual_report.md:")
assert spans[0].line_start >= 1                 # header-only blocks are skipped
print(spans[0].location, "->", spans[0].text[:48])

# Any VectorBackend subclass can replace the default -- e.g. an external DB.
assert issubclass(InMemoryVectorBackend, VectorBackend)
""")

md(r"""
## Self-check

The chapter's claim: the default pipeline abstains on a prose question, the
opt-in dense fallback answers it but flags the answer **unverified** with a
notice, and `strict_mode` refuses even that. We assert the config-level
contract here (no GPU); the **CI**-marked cells assert the runtime behaviour.
""")

code(r"""
# Config-level contract for the three modes (CPU-only, deterministic).
default_cfg = RagConfig(llm=ScriptedBackend("synth"))
dense_cfg = RagConfig(llm=ScriptedBackend("synth"), dense_fallback=True)
strict_cfg = RagConfig(llm=ScriptedBackend("synth"),
                       dense_fallback=True, strict_mode=True)

# Default: dense quarantine is off.
assert default_cfg.dense_fallback is False and default_cfg.strict_mode is False
# Opt-in: dense on, strict off -> a dense answer is returned (flagged unverified).
assert dense_cfg.dense_fallback is True and dense_cfg.strict_mode is False
# Strict: dense on but downgraded to abstain.
assert strict_cfg.dense_fallback is True and strict_cfg.strict_mode is True

# Per-role fallback chain resolves correctly.
shared = ScriptedBackend("qwen")
assert RagConfig(llm=shared).verify_llm() is shared
print("Ch14 self-check passed: pluggable roles + quarantined dense fallback.")
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python"},
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    nbf.write(nb, f)
print("wrote", OUT, "with", len(cells), "cells")
