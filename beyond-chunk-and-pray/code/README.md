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

## Requirements — two tiers

The book runs in two tiers, with different hardware needs:

- **Tier 1 — pure geometry (CPU, no LLM).** The GEODE store build, triple-mediated
  retrieval, binding, provenance, and Exact Numerical Memory. This is the core of
  the book — *grounding is geometry, not generation* — and it runs on a **CPU-only**
  machine once the base store is built. No GPU, no Qwen.
- **Tier 2 — answer synthesis (GPU + Qwen).** The chapters that generate and verify
  prose answers, the calibration/evaluation chapters, and the full data-enrichment +
  encoder-fine-tuning pipeline run a local **Qwen3-4B-Instruct** (no API key).
  These **expect a CUDA GPU**: the model loads in float16 and the synthesis cells
  assume a GPU device, so CPU inference is not supported out of the box.

## Run

Requires the licensed `knowlytix` substrate — `pip install knowlytix` and a
developer license from <https://knowlytix.ai/signup/> at `~/.knowlytix/license.key`
(see the repo README, "The knowlytix substrate").

```bash
# 1. Build the base store (Tier 1; CPU-only). Or run notebooks/00_setup.ipynb.
python scripts/build_store.py          # trained store + corpus_facts.md

# 2. Run the chapters. Tier-1 (geometry) chapters run on CPU; the Tier-2
#    synthesis/calibration/evaluation chapters need a GPU + the Qwen download.
jupyter nbconvert --execute notebooks/*.ipynb

# 3. For the advanced chapters, build the full pipeline first (GPU + Qwen):
#    run notebooks/00_setup.ipynb with RUN_FULL = True
#    (enrichment -> encoder fine-tuning -> gate calibration).
```
