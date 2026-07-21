# Beyond Vibe and Pray — `reasonloop`

*A semantic execution substrate: the LLM as compiler and grounded synthesizer
over a deterministic GMS reasoning engine.* **Work in progress.**

The thesis: don't let the LLM traverse the graph, emit the query, and decide the
answer, then hope it was right (*vibe reasoning*). Instead confine the LLM to what
it is reliable at — understanding language and producing explanations — and hand
every consequential step (retrieval, reasoning, policy, evidence) to a
deterministic execution engine:

```
Natural language → LLM compiler → ontology-based IR / GMS query → validator → planner
  → GMS store (retrieval · geometric operators · rule engine · policy)
  → results + provenance + source → grounded LLM synthesis → answer with citations
```

The LLM never reaches the store directly; it only emits a validated query and
later reads back grounded context to synthesize a cited answer.

## Layout

```
code/
  reasonloop/        # the package
    ingest/          # web crawler (Playwright) → markdown
  scripts/
    fetch_ffiec.py   # pull FFIEC BSA/AML source pages
    build_stores.py  # build GMS stores from the fetched corpus
  data/ffiec/        # fetched FFIEC CIP / SAR source documents (md/json/html)
```

The trained GMS stores under `data/stores/` are **not committed** (regenerable
via `scripts/build_stores.py`); see the repo-root README for the model-weights
policy.

## Install

```bash
python -m pip install -e code
python -m playwright install chromium   # for the crawler
```

The geometric substrate (`knowlytix.core` / `.knowledge` / `.harness`) is the
licensed dependency — not on public PyPI. See the repo-root README.
