# SPDX-License-Identifier: Apache-2.0
"""Build the Chapter 6 notebooks (DoE data enrichment), in two parallel styles:

  06_doe_enrichment_a_inline.ipynb   self-contained: the suite API calls inline,
                                     run on the shared store -- the didactic read.
  06_doe_enrichment_b_project.ipynb  runs the real project step
                                     (scripts/enrich_data.py) and inspects the
                                     artifacts -- the reproducible, drift-proof read.

Both operate on the SAME shared artifacts (data/gms_annual_report_store built in
Ch4, data/enrichment written here) so the chapters compose into one cumulative
build by the capstone. CPU-only authoring: cells are emitted, not executed.
"""
from __future__ import annotations

import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
NBDIR = os.path.join(HERE, "..", "notebooks")

_BOOT = (
    'import os, sys\n'
    'KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", "")\n'
    'sys.path.insert(0, KNOWLYTIX_SRC)\n'
    'REPO = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()\n'
    'sys.path.insert(0, os.path.join(REPO, "scripts"))   # project modules'
)


def _nb(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s)
                for k, s in cells]
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}}
    return nb


def _write(name, cells):
    path = os.path.abspath(os.path.join(NBDIR, name))
    with open(path, "w") as fh:
        nbf.write(_nb(cells), fh)
    print("wrote", path, f"({len(cells)} cells)")


# --- Style A: inline, didactic ------------------------------------------------
A = [
    ("md", "# Ch6 (A, inline) - Generating Data with a Designed Experiment\n\n"
           "The store from Ch4 is an *oracle*: every fact it holds is a known answer. "
           "We mine it for questions, enrich each across a DoE of presentation "
           "factors, and emit one corpus that trains the encoders (Ch7) and tests "
           "the RAG (Ch15). This notebook shows the suite API inline."),
    ("code", _BOOT),
    ("code",
     'import torch\n'
     'from knowlytix.knowledge.config import DocGMSConfig, GeometryConfig\n'
     'from knowlytix.knowledge.store import GMSExpertStore\n'
     'from knowlytix.harness.suite import (\n'
     '    Catalog, resolve, CatalogBaseSource, compose, graphdoe_design)\n'
     '\n'
     'STORE = os.path.join(REPO, "data", "gms_annual_report_store")\n'
     'dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
     'store = GMSExpertStore(DocGMSConfig(store_path=STORE, geometry=GeometryConfig(d_v=64, d_u=64, m=32, d=32)), device=dev)\n'
     'assert store.load(), "build the store first (Ch4)"'),
    ("md", "## Mine base questions (content base types from the suite catalog)"),
    ("code",
     'CAT = Catalog.load()\n'
     'suite = resolve(CAT, ["exact_recall", "multi_hop"],\n'
     '                ["clarity", "style", "length", "expertise", "paraphrase_depth"],\n'
     '                mode="embedded")\n'
     'items = CatalogBaseSource(store, max_per_category=8, seed=42).items(suite)\n'
     'print(len(items), "base questions, each graph-derived")'),
    ("md", "## Design the experiment (embedded mode: #scenarios == n_runs)"),
    ("code",
     'from functools import partial\n'
     'scns = compose(suite, items, n_runs=150, seed=42,\n'
     '               design_fn=partial(graphdoe_design, method="sobol+refine"),\n'
     '               balance_base=True)\n'
     'print(len(scns), "scenarios over", suite.factor_names)'),
    ("md", "## Materialize with Qwen, then emit the corpus\n\n"
           "Factor levels become a natural question via a batched, guarded Qwen "
           "rewrite (full body in `scripts/enrich_data.py`). It emits the cohort, "
           "the v-space SFT rows, the u-space groups and the draft pairs; here we "
           "inspect the result."),
    ("code",
     'import json\n'
     'cohort = json.load(open(os.path.join(REPO, "data", "enrichment", "rag_cohort.json")))\n'
     'print(len(cohort), "cohort cases; sample:", cohort[0]["question"][:70])'),
    ("md", "**Self-check** - every case has a graph-derived answer and balanced factors."),
    ("code",
     'import collections\n'
     'assert all(c["expected_answer"] is not None for c in cohort)\n'
     'clar = collections.Counter(c["_factors"]["clarity"] for c in cohort)\n'
     'assert min(clar.values()) >= 30\n'
     'print("OK: designed, ground-truthed, balanced")'),
]

# --- Style B: project, reproducible ------------------------------------------
B = [
    ("md", "# Ch6 (B, project) - run the real enrichment step\n\n"
           "Same chapter, the drift-proof read: run the actual project script "
           "`scripts/enrich_data.py` and inspect the artifacts it writes - the exact "
           "step the capstone pipeline depends on."),
    ("code", _BOOT),
    ("md", "## Run the project's DoE enrichment\n"
           "`enrich_data.py` = mine the store (suite generators) -> embedded DoE "
           "(sobol+refine) -> batched Qwen naturalization -> emit the corpus."),
    ("code",
     'import subprocess, sys\n'
     'subprocess.run([sys.executable, os.path.join(REPO, "scripts", "enrich_data.py"),\n'
     '                "--n-runs", "150"], check=True)'),
    ("md", "## Inspect the emitted artifacts (shared with Ch7 and Ch15)"),
    ("code",
     'import json, collections\n'
     'ENR = os.path.join(REPO, "data", "enrichment")\n'
     'cohort = json.load(open(os.path.join(ENR, "rag_cohort.json")))\n'
     'ugroups = json.load(open(os.path.join(ENR, "embedding_u_groups.json")))\n'
     'print("rag_cohort.json     :", len(cohort), "cases")\n'
     'print("embedding_u_groups  :", sorted(ugroups))\n'
     'print("by clarity          :", dict(collections.Counter(\n'
     '      c["_factors"]["clarity"] for c in cohort)))'),
    ("md", "**Self-check** - the artifacts Ch7 (SFT) and Ch15 (evaluation) consume exist."),
    ("code",
     'for f in ["rag_cohort.json", "embedding_sft.jsonl",\n'
     '          "embedding_u_groups.json", "llm_draft_sft.jsonl"]:\n'
     '    assert os.path.isfile(os.path.join(ENR, f)), f\n'
     'print("OK: enrichment corpus ready for Ch7 and Ch15")'),
]


def build() -> None:
    _write("06_doe_enrichment_a_inline.ipynb", A)
    _write("06_doe_enrichment_b_project.ipynb", B)


if __name__ == "__main__":
    build()
