# Beyond "Chunk and Pray" — Building Trustworthy RAG with Geometric Memory

The retrieval companion to *Beyond Prompt and Pray*. A tutorial book on building a
RAG system that grounds every answer, cites its source, gets numbers byte-exact,
and **abstains rather than guess**.

**Thesis.** RAG is a *grounding* discipline, not a similarity search. Retrieval is
**triple-mediated** — questions are answered *through* a verified knowledge graph
with provenance, not by a top-k vector lookup the generator is trusted to use
well. The dense vector index is **distrusted**: optional, off by default, flagged
when used. Every answer is verifiable; the system abstains when it cannot ground
an answer.

## Source of truth

The **library is authoritative**: every type and function comes from
`knowlytix.knowledge.rag` and `knowlytix.knowledge.geode` (shipped and tested in
the GMS repo). The runnable notebooks in `notebooks/` are the walkthrough; if a
notebook ever disagrees with the library, the library is right.

## Layout

```
notebooks/   00_setup.ipynb (build the store data) + one notebook per chapter
scripts/     build_store.py (corpus -> trained store + corpus_facts.md), build_notebooks.py
data/        annual_report.md (the corpus), gms_annual_report_store/, eval_cohort.json
```

The trained store (`data/gms_annual_report_store/`) is a build artifact — **not
committed and not shipped in the package** (git-ignored). Regenerate it by running
**`notebooks/00_setup.ipynb`** (or `python scripts/build_store.py`); only the input
corpus is tracked.

## Corpus

A synthetic but realistic **Annual Financial Report** (Northwind Industries,
FY2025), designed to exercise every GEODE-RAG feature: exact-number tables → ENM +
sum anchors, a segment→division→region hierarchy → multi-hop, single-valued facts
→ contradiction, aliases → binding, and prose sections (MD&A, Risk Factors,
Outlook) with no authoritative numbers → coverage blind spots and the dense
fallback demo.

## Run

Requires the licensed `knowlytix` substrate — `pip install knowlytix` and a
developer license from <https://knowlytix.ai/signup/> at `~/.knowlytix/license.key`
(see the repo README, "The knowlytix substrate").

```bash
# 1. build the store data (or run notebooks/00_setup.ipynb, which does this)
python scripts/build_store.py          # trained store + corpus_facts.md
# 2. run the chapters
jupyter nbconvert --execute notebooks/*.ipynb
```

The GEODE stage runs locally on Qwen2.5-3B-Instruct; no API key.
