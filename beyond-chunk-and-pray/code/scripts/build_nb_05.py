# SPDX-License-Identifier: Apache-2.0
"""Builder for notebooks/05_geode_self_correction_a_inline.ipynb (Ch5).

Emits a valid nbformat-4 notebook. CPU only — does NOT execute the notebook
(no store load, no Qwen). The lead runs it in CI. Grounded in
data/corpus_facts.md.
"""
from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "notebooks", "05_geode_self_correction_a_inline.ipynb",
)

cells = []


def md(text: str) -> None:
    cells.append(new_markdown_cell(text))


def code(src: str, *, ci_gpu: bool = False) -> None:
    src = src.strip("\n")
    cell = new_code_cell(src)
    if ci_gpu:
        cell["metadata"]["tags"] = ["ci-gpu"]
    cells.append(cell)


# ── Title ──────────────────────────────────────────────────────────────────
md(
    "# Chapter 5 — GEODE: self-correcting extraction\n"
    "\n"
    "Chapter 4 built a store from the annual report by trusting the extractor.\n"
    "This chapter does not trust it. Before a triple is indexed, GEODE runs a\n"
    "closed loop that lets the **geometry judge the extraction**: a composition\n"
    "critic catches relational triples that contradict the rest of the graph, and\n"
    "external **anchors** (a declared sum, a duplicate value) catch numeric errors\n"
    "the geometry is blind to. We corrupt the corpus on purpose and watch the loop\n"
    "flag each break **with provenance** — then state, honestly, the one class of\n"
    "error neither can catch.\n"
    "\n"
    "All symbols come from `knowlytix.knowledge.geode`. The actor LLM, when used,\n"
    "is Qwen 3B only (research-integrity lock) — Claude is never in the loop."
)

# ── Cell 0: bootstrap (global brief, verbatim) ────────────────────────────
code(
    """
import os, sys
KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", "/home/user/jupyterlab/GMS-knowlytix")
sys.path.insert(0, KNOWLYTIX_SRC)
"""
)

# ── Section: the corpus triples ──────────────────────────────────────────
md(
    "## The triples, before correction\n"
    "\n"
    "We work over the canonical Northwind FY2025 triples (see `data/corpus_facts.md`).\n"
    "The segment table declares a `Total` row, and `Total` revenue equals the sum of\n"
    "the four segment revenues — `120 + 80 + 95 + 60 == 355`. That redundancy is what\n"
    "the **sum anchor** checks. The `segment -> division -> region` chain is the\n"
    "redundancy the **composition critic** checks."
)

code(
    """
# Canonical triples from the annual report (grounded in data/corpus_facts.md).
# In a real run these come from `ingest_markdown` / `build_rag_store`; here we
# state them explicitly so the corruption is unambiguous.
clean_triples = [
    ("cloud platform", "has_revenue", "120.0"),
    ("devices", "has_revenue", "80.0"),
    ("logistics", "has_revenue", "95.0"),
    ("retail", "has_revenue", "60.0"),
    ("total", "has_revenue", "355.0"),
    ("cloud platform", "has_division", "technology"),
    ("devices", "has_division", "technology"),
    ("logistics", "has_division", "operations"),
    ("retail", "has_division", "operations"),
    ("technology", "has_region", "north america"),
    ("operations", "has_region", "europe"),
]

# The Total row declares: Total = sum of the four segments.
parts = [120.0, 80.0, 95.0, 60.0]
assert sum(parts) == 355.0
"""
)

# ── Listing 1: sum anchor ────────────────────────────────────────────────
md(
    "## Listing 1 — corrupt a segment figure: the sum anchor flags it\n"
    "\n"
    "Change Devices revenue from `80.0` to `88.0`. Now `120 + 88 + 95 + 60 = 363`,\n"
    "but the `Total` row still says `355.0`. No single triple is *internally* wrong —\n"
    "the geometry cannot see it. The **sum anchor** can: it reads the exact values\n"
    "back through the ENM register and compares the declared total to the sum of\n"
    "parts. `AnchorChecker.auto_sum_constraints` derives the constraint from the\n"
    "`Total` row the way a financial table already declares it."
)

code(
    """
from knowlytix.knowledge.geode import (
    enm_from_triples, AnchorChecker, ProvenanceLedger,
)

# Inject the sum break: Devices 80.0 -> 88.0.
corrupted = [
    (h, r, "88.0") if (h, r) == ("devices", "has_revenue") else (h, r, t)
    for (h, r, t) in clean_triples
]

# Build the integrity-checked exact register, then the anchor checker.
enm = enm_from_triples(corrupted)
ledger = ProvenanceLedger.from_text(open("../data/annual_report.md").read(),
                                    "../data/annual_report.md")
checker = AnchorChecker.from_enm(enm, ledger)

# The sum constraint is derived automatically from the "total" row.
constraints = checker.auto_sum_constraints()
for c in constraints:
    print("constraint:", c.total, "= sum", [p[0] for p in c.parts])

violations = checker.check_all(corrupted)
for v in violations:
    print(v.kind, "|", v.message)
    for loc in v.locations:
        print("   provenance:", loc.location(), "raw:", repr(loc.raw))
"""
)

md(
    "**Expected.** `auto_sum_constraints` derives `('total','has_revenue') = sum of\n"
    "[cloud platform, devices, logistics, retail]`. `check_all` returns a `sum`\n"
    "violation: declared `total.has_revenue=355` `!=` `sum(parts)=363` (off by `8`),\n"
    "with a provenance location pointing back into the segment table. The anchor\n"
    "**detects** the break and names the candidate parts; it does not auto-rewrite\n"
    "a number — which part is wrong is a review decision (the honest localization\n"
    "limit of a sum constraint)."
)

# ── Listing 2: composition critic ────────────────────────────────────────
md(
    "## Listing 2 — corrupt a relational triple: the composition critic catches and fixes it\n"
    "\n"
    "Now break a *relational* fact: claim `cloud platform has_division operations`\n"
    "(it is `technology`). This contradicts the redundant chain — `cloud platform`'s\n"
    "division, composed with that division's region, must land on the same region the\n"
    "rest of the technology segments reach. The `CompositionCritic` measures the\n"
    "geodesic gap between the composed-path prediction and the asserted tail; a\n"
    "clean edge closes (~0), a contradicting one does not. `GeodeLoop` localizes the\n"
    "flag to its source span and repairs it with the geometry's prediction.\n"
    "\n"
    "This cell trains a small GMS, so it is **GPU/CI-only** — do not run it here."
)

code(
    """
from knowlytix.knowledge.geode import GeodeLoop, make_default_trainer

# Break a relational triple: cloud platform's division technology -> operations.
relational_break = [
    (h, r, "operations") if (h, r) == ("cloud platform", "has_division") else (h, r, t)
    for (h, r, t) in clean_triples
]

# Write the corrupted triples to a tiny markdown table so the loop's regex
# ingest + provenance ledger can re-read the source span.
import tempfile, textwrap
md_doc = textwrap.dedent('''
    # Northwind segments

    | Segment | Division |
    | :--- | :--- |
    | Cloud Platform | Operations |
    | Devices | Technology |
    | Logistics | Operations |
    | Retail | Operations |

    | Division | Region |
    | :--- | :--- |
    | Technology | North America |
    | Operations | Europe |
''').strip()
path = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False).name
open(path, "w").write(md_doc)

# Geometry is authoritative; no LLM is needed to FIND or FIX the relational error
# (the Qwen actor only raises confidence when it agrees). Trainer is injected so
# orchestration is testable; this still trains a GMS -> GPU/CI only.
loop = GeodeLoop(make_default_trainer(epochs=300), check_anchors=False)
result = loop.run(path)

print("converged:", result.converged, "| iterations:", result.iterations)
for c in result.corrections:
    print("fixed:", c["triple"], "-> geometry:", c["geometry"],
          "| residual:", round(c["residual"], 3), "| at", c["location"])
""",
    ci_gpu=True,
)

md(
    "**Expected.** The critic flags `('cloud platform','has_division','operations')`\n"
    "with a high composition residual; the loop's geometry prediction is\n"
    "`technology`, it rewrites the triple, and re-converges. Each correction carries\n"
    "the source `file:line:char` location. Unlike the numeric case, the relational\n"
    "error **is** auto-fixed — because the graph itself contains a second source of\n"
    "truth (the composition path), so the correct value is recoverable, not just\n"
    "detectable. This is the GMS box: geometry catches and repairs what an LLM\n"
    "re-reading the same wrong cell cannot."
)

# ── Listing 3: the honest limit ──────────────────────────────────────────
md(
    "## Listing 3 — the honest limit: a lone value with no redundancy\n"
    "\n"
    "Consider `total_assets has_amount 540.0` (Balance Sheet). It appears once, with\n"
    "no `Total = Σ parts` row over the balance-sheet line items and no second\n"
    "statement of the figure. Corrupt it to `999.0` and **neither** mechanism fires:\n"
    "the composition critic needs a redundant relational chain (there is none), and\n"
    "the sum anchor needs declared parts (there are none). This is the boundary the\n"
    "design doc states plainly."
)

code(
    """
# A lone numeric fact: one value, no parts, no duplicate, no composition chain.
lone = [("total assets", "has_amount", "999.0")]   # truth is 540.0

enm_lone = enm_from_triples(lone)
checker_lone = AnchorChecker.from_enm(enm_lone)

# No sum constraint can be derived (no "total" row over parts), and there is no
# duplicate of the same (entity, relation) -> nothing to cross-check.
print("derived sum constraints:", checker_lone.auto_sum_constraints())
print("anchor violations:", checker_lone.check_all(lone))
"""
)

md(
    "**Expected.** `auto_sum_constraints` returns `[]` and `check_all` returns `[]`:\n"
    "the corrupted `999.0` passes silently. Geometry corrects what violates\n"
    "redundancy; it cannot invent a cross-check that the document never provided.\n"
    "The remedy is upstream — declare a constraint, add a duplicate statement, or\n"
    "carry the figure as an ENM-checked exact value sourced to its one cell — not a\n"
    "better critic."
)

# ── Exercise ─────────────────────────────────────────────────────────────
md(
    "## Exercise — the duplicate anchor\n"
    "\n"
    "Assert the *same* `(entity, relation)` twice with conflicting exact values and\n"
    "confirm the **duplicate anchor** fires (precise localization, unlike the sum\n"
    "case). Worked solution below."
)

code(
    """
# EXERCISE SOLUTION — conflicting duplicate of the same (entity, relation).
dup = clean_triples + [("total", "has_revenue", "350.0")]  # conflicts with 355.0

checker_dup = AnchorChecker.from_enm(enm_from_triples(dup))
dups = checker_dup.check_duplicates(dup)
for v in dups:
    print(v.kind, "|", v.message, "| residual:", v.residual)

assert any(d.kind == "duplicate" for d in dups), "duplicate anchor should fire"
"""
)

# ── Self-check (final cell) ──────────────────────────────────────────────
md(
    "## Self-check\n"
    "\n"
    "The chapter's claim: when a segment figure is corrupted so `Total != Σ parts`,\n"
    "the loop's anchor layer surfaces the break. We assert the injected sum break\n"
    "appears in the anchor violations — exactly the property the brief's self-check\n"
    "names (`anchor_violations` contains the injected sum break)."
)

code(
    """
# Self-check: the injected sum break (Devices 80 -> 88) is detected as a "sum"
# anchor violation. This is the CPU-only anchor path (no GMS training, no Qwen),
# so it runs deterministically in CI.
enm_chk = enm_from_triples(corrupted)
checker_chk = AnchorChecker.from_enm(enm_chk)
viols = checker_chk.check_all(corrupted)

sum_viols = [v for v in viols if v.kind == "sum"]
assert sum_viols, "expected a sum anchor violation for the corrupted segment"
v = sum_viols[0]
assert abs(v.residual - 8.0) < 1e-6, f"expected residual 8.0, got {v.residual}"
print("OK — sum anchor flagged the injected break:", v.message)
"""
)

nb = new_notebook(cells=cells)
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", OUT, "with", len(cells), "cells")
