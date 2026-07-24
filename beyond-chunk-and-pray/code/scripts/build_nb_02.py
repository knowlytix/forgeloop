# SPDX-License-Identifier: Apache-2.0
"""Build notebooks/02_geometric_memory_a_inline.ipynb (Ch2 — Geometric memory in one
chapter) deterministically with nbformat. CPU-only emit; does NOT execute the
notebook (no store load / no Qwen). The lead executes it in CI.

Run:  python scripts/build_nb_02.py
"""

from __future__ import annotations

import os

import nbformat as nbf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "notebooks", "02_geometric_memory_a_inline.ipynb")


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


CELLS: list[nbf.NotebookNode] = []

CELLS.append(md(
    "# Chapter 2 — Geometric memory in one chapter\n"
    "\n"
    "The GMS store is a **trained triple register**, not a vector database.\n"
    "This notebook exercises the six primitives at agent-author altitude:\n"
    "`lookup_enm`, `score_triple`, `query_triples`, `link_predict`,\n"
    "`check_holonomy`, `tension_energy` — over the Northwind Industries FY2025\n"
    "store built by `scripts/build_store.py`. Geometry internals (rotors, caps,\n"
    "transport) are deferred to Appendix C and the GMS monograph.\n"
    "\n"
    "The claim this notebook proves: **exact figures come back byte-exact from\n"
    "Exact Numerical Memory — never re-parsed from prose — and a false fact sits\n"
    "measurably farther from the manifold than the true one.**"
))

# --- Cell 1: bootstrap (verbatim from global brief) ---
CELLS.append(code(
    "import os, sys\n"
    "KNOWLYTIX_SRC = os.environ.get(\"KNOWLYTIX_SRC\", \"/path/to/GMS-knowlytix\")\n"
    "sys.path.insert(0, KNOWLYTIX_SRC)"
))

CELLS.append(md(
    "## Load the trained store\n"
    "\n"
    "A store is reopened with the *same* geometry/loss configuration it was\n"
    "trained under (`scripts/build_store.py`): cap-admissibility loss and a\n"
    "64/64/32/32 geometry. `load()` rebuilds the model, the entity/relation\n"
    "adapter, the document graph, and the Exact Numerical Memory from disk."
))

# --- Cell 2: load store ---
CELLS.append(code(
    "import torch\n"
    "from knowlytix.core.config import GeometryConfig\n"
    "from knowlytix.knowledge.config import DocGMSConfig\n"
    "from knowlytix.knowledge.store import GMSExpertStore\n"
    "\n"
    "REPO_ROOT = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == \"notebooks\" else os.getcwd()\n"
    "STORE = os.path.join(REPO_ROOT, \"data\", \"gms_annual_report_store\")\n"
    "\n"
    "cfg = DocGMSConfig(\n"
    "    store_path=STORE,\n"
    "    ingest_mode=\"regex\",\n"
    "    loss_mode=\"cap\",\n"
    "    geometry=GeometryConfig(d_v=64, d_u=64, m=32, d=32),\n"
    ")\n"
    "store = GMSExpertStore(cfg, device=torch.device(\"cpu\"))\n"
    "assert store.load(), f\"no store at {STORE} — run scripts/build_store.py first\"\n"
    "print(\"relations:\", sorted(store.adapter.relation_to_idx))"
))

CELLS.append(md(
    "## Primitive 1 — `lookup_enm`: exact recall vs re-parsing prose\n"
    "\n"
    "Every authoritative number lives in **Exact Numerical Memory** under a\n"
    "`(category, id)` key with SHA-256 integrity. `lookup_enm` returns the value\n"
    "byte-exact. The chunk-and-pray alternative — regexing a figure out of\n"
    "retrieved text — is fragile: the same digits appear in multiple sentences,\n"
    "thousands separators and currency glyphs vary, and a near-miss looks like a\n"
    "hit. Below we read FY2025 revenue from ENM, then deliberately break a naive\n"
    "parse of the same number from prose."
))

# --- Cell 3: lookup_enm vs parse (break the parse) ---
CELLS.append(code(
    "import re\n"
    "\n"
    "# Authoritative path: byte-exact read from Exact Numerical Memory.\n"
    "revenue_fy2025 = store.lookup_enm(\"income_statement\", \"Revenue/FY2025\")\n"
    "total_segment_rev = store.lookup_enm(\"segment_performance\", \"Total/All/Revenue\")\n"
    "print(\"ENM income_statement/Revenue/FY2025 =\", revenue_fy2025)\n"
    "print(\"ENM segment_performance/Total/All/Revenue =\", total_segment_rev)\n"
    "\n"
    "# Chunk-and-pray path: parse a figure out of a retrieved sentence.\n"
    "# Two sentences mention '355' — and one mentions the WRONG prior-year basis.\n"
    "retrieved_prose = (\n"
    "    \"Total revenue grew to $355M in FY2025, up from $320M, while the \"\n"
    "    \"Cloud Platform segment alone contributed $120M.\"\n"
    ")\n"
    "first_number = float(re.search(r\"\\$(\\d+)M\", retrieved_prose).group(1))\n"
    "print(\"naive parse (first $NM match) =\", first_number)\n"
    "\n"
    "# The parse is right here only by luck of word order. Reorder the clause and\n"
    "# the same regex now returns the SEGMENT figure, not total revenue:\n"
    "reordered = (\n"
    "    \"The Cloud Platform segment contributed $120M as total revenue \"\n"
    "    \"grew to $355M in FY2025.\"\n"
    ")\n"
    "wrong = float(re.search(r\"\\$(\\d+)M\", reordered).group(1))\n"
    "print(\"naive parse after reorder =\", wrong, \"(should be 355, got\", wrong, \")\")\n"
    "assert wrong != revenue_fy2025, \"the prose parse silently returned the wrong number\"\n"
    "print(\"ENM is order-invariant and exact; the parse is neither.\")"
))

CELLS.append(md(
    "## Primitive 2 — `score_triple`: a true fact vs a false one\n"
    "\n"
    "`score_triple(head, rel, tail)` returns a geodesic distance on the trained\n"
    "manifold — **lower is more plausible**. A fact the store was trained on sits\n"
    "close to the conditioned cap center; a fabricated tail sits farther away. We\n"
    "read the gap between the true Cloud Platform headcount (340) and a fabricated\n"
    "one. Tails are canonicalized numeric strings (`\"340.0\"`), matching the\n"
    "triples in `data/corpus_facts.md`."
))

# --- Cell 4: score_triple true vs false ---
CELLS.append(code(
    "true_d = store.score_triple(\"cloud platform\", \"has_headcount\", \"340.0\")\n"
    "false_d = store.score_triple(\"cloud platform\", \"has_headcount\", \"520.0\")  # that's retail's\n"
    "print(f\"d(cloud platform, has_headcount, 340.0) = {true_d:.4f}  [asserted]\")\n"
    "print(f\"d(cloud platform, has_headcount, 520.0) = {false_d:.4f}  [false]\")\n"
    "print(f\"geodesic gap (false - true) = {false_d - true_d:.4f}\")\n"
    "\n"
    "# The asserted fact scores strictly closer than the swapped tail.\n"
    "assert true_d < false_d, \"true fact must score closer than the false one\"\n"
    "rho = store.cap_radius(\"has_headcount\")\n"
    "print(\"cap radius rho for has_headcount =\", rho)"
))

CELLS.append(md(
    "## Primitive 3 — `query_triples`: asserted edges over a segment\n"
    "\n"
    "`query_triples` is exact pattern-matching over the document graph — no\n"
    "geometry, no guessing. Leave a slot `None` to wildcard it. This is the path\n"
    "the retriever prefers: asserted edges with provenance, never link-prediction\n"
    "guesses."
))

# --- Cell 5: query_triples over a segment ---
CELLS.append(code(
    "cloud_edges = store.query_triples(head=\"cloud platform\")\n"
    "for h, r, t in cloud_edges:\n"
    "    print(f\"{h:>16} {r:>14} {t}\")\n"
    "\n"
    "# All four segments that report a revenue edge:\n"
    "rev_edges = store.query_triples(relation=\"has_revenue\")\n"
    "segments = {h: t for (h, r, t) in rev_edges if h != \"total\"}\n"
    "print(\"\\nper-segment revenue:\", segments)\n"
    "assert (\"cloud platform\", \"has_division\", \"technology\") in cloud_edges"
))

CELLS.append(md(
    "## Primitive 4 — `link_predict`: ranked tails for an open slot\n"
    "\n"
    "When you have `(head, relation, ?)` and want the store's *ranked* guess,\n"
    "`link_predict` scores every type-valid tail and returns the closest ones\n"
    "(lower distance first). It is type-constrained: only entities ever seen as a\n"
    "tail of this relation are scored, so numeric and categorical tails never mix.\n"
    "Note this is a **prediction**, not an assertion — Chapter 8 shows why the\n"
    "retriever prefers an asserted `query_triples` edge over a `link_predict`\n"
    "guess whenever one exists."
))

# --- Cell 6: link_predict ---
CELLS.append(code(
    "ranked = store.link_predict(\"cloud platform\", \"has_division\", top_k=3)\n"
    "for tail, dist in ranked:\n"
    "    print(f\"{tail:>14}  d={dist:.4f}\")\n"
    "\n"
    "# Cloud Platform's division is Technology — it should rank first.\n"
    "assert ranked, \"link_predict returned no candidates\"\n"
    "top_tail, _ = ranked[0]\n"
    "print(\"top-ranked division:\", top_tail)"
))

CELLS.append(md(
    "## Primitive 5 — `check_holonomy`: is a multi-hop path consistent?\n"
    "\n"
    "`check_holonomy(path, direct)` measures the *holonomy defect* — how far\n"
    "composing a relation path drifts from a direct edge. **0 = consistent.** It\n"
    "is the geometric backbone of multi-hop retrieval (Chapter 8) and of GEODE's\n"
    "composition critic (Chapter 5). Here we ask whether composing\n"
    "`has_division` then `has_region` is consistent with a hypothetical direct\n"
    "`has_region` edge from a segment. The store has no direct segment→region\n"
    "edge, so we read the defect of the path against the `has_region` rotor."
))

# --- Cell 7: check_holonomy ---
CELLS.append(code(
    "defect = store.check_holonomy([\"has_division\", \"has_region\"], \"has_region\")\n"
    "print(\"holonomy defect for has_division ∘ has_region vs has_region =\", defect)\n"
    "tau = store.config.verify.tau_path\n"
    "print(\"path-consistency threshold tau_path =\", tau)\n"
    "# A defect at or below tau_path is treated as a consistent composition.\n"
    "assert defect is not None, \"both relations must exist for a holonomy read\""
))

CELLS.append(md(
    "## Primitive 6 — `tension_energy`: contradiction as distance\n"
    "\n"
    "`tension_energy(a, b)` reads the relationship between two entities on a\n"
    "0..2 scale: **0 = agree, √2 ≈ 1.41 = unrelated, 2 = contradict.** Functional\n"
    "facts (one division head, one fiscal-year-end) make contradiction a\n"
    "*geometric* signal rather than a string compare. Below: the two real\n"
    "division heads (`dana cole`, `sam reyes`) are distinct people running\n"
    "distinct divisions — the energy reads them as not-agreeing, which is exactly\n"
    "what flags a 'two CEOs' style contradiction in Chapter 5 and Chapter 10."
))

# --- Cell 8: tension_energy ---
CELLS.append(code(
    "te_heads = store.tension_energy(\"dana cole\", \"sam reyes\")\n"
    "te_self = store.tension_energy(\"dana cole\", \"dana cole\")\n"
    "print(f\"tension(dana cole, sam reyes) = {te_heads:.4f}  [two distinct heads]\")\n"
    "print(f\"tension(dana cole, dana cole) = {te_self:.4f}  [identical -> agree]\")\n"
    "assert te_self <= te_heads, \"an entity must not contradict itself\"\n"
    "print(\"contradiction is a distance, not a string mismatch.\")"
))

CELLS.append(md(
    "## Self-check — the chapter's claim\n"
    "\n"
    "ENM recall is byte-exact, and the asserted fact scores strictly closer than\n"
    "the false one. If this cell passes, the store behaved as a trained register,\n"
    "not a fuzzy index."
))

# --- Final cell: self-check assert (the chapter's claim) ---
CELLS.append(code(
    "# 1. ENM is exact: FY2025 revenue and total segment revenue both equal 355.0.\n"
    "assert store.lookup_enm(\"income_statement\", \"Revenue/FY2025\") == 355.0\n"
    "assert store.lookup_enm(\"segment_performance\", \"Total/All/Revenue\") == 355.0\n"
    "assert store.lookup_enm(\"segment_performance\", \"Cloud Platform/Technology/Headcount\") == 340.0\n"
    "\n"
    "# 2. A true fact sits closer on the manifold than a fabricated one.\n"
    "d_true = store.score_triple(\"cloud platform\", \"has_headcount\", \"340.0\")\n"
    "d_false = store.score_triple(\"cloud platform\", \"has_headcount\", \"520.0\")\n"
    "assert d_true < d_false\n"
    "\n"
    "print(\"OK: ENM byte-exact (355.0 / 340.0) and true-fact geodesic gap =\",\n"
    "      f\"{d_false - d_true:.4f}\")"
))


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", OUT, "with", len(CELLS), "cells")


if __name__ == "__main__":
    main()
