# SPDX-License-Identifier: Apache-2.0
"""Builder for notebooks/03_provenance_a_inline.ipynb (CPU only; does NOT execute cells).

Run: python scripts/build_nb_03_provenance.py
Emits a valid nbformat-4 notebook. No store/Qwen is loaded here.
"""
from __future__ import annotations

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src))


md(
    "# Chapter 3 — Provenance: span ↔ triple\n"
    "\n"
    "Every fact a GEODE-RAG answer rests on points back to the exact source\n"
    "span it came from — `file:line:char`, plus the raw substring. This\n"
    "notebook reconstructs that span↔triple map for the Northwind FY2025\n"
    "annual report with `ProvenanceLedger`, shows a prose triple resolving as\n"
    "`unaligned`, and proves a resolved span actually supports its triple with\n"
    "`is_consistent`.\n"
    "\n"
    "We do **not** load the trained store or Qwen here; provenance alignment is\n"
    "pure text→span work over the markdown source."
)

# --- Cell 1: KNOWLYTIX_SRC bootstrap (verbatim from global brief) ---
code(
    "import os, sys\n"
    'KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", "/home/user/jupyterlab/GMS-knowlytix")\n'
    "sys.path.insert(0, KNOWLYTIX_SRC)"
)

# --- Cell 2: imports + corpus path ---
code(
    "from knowlytix.knowledge.geode.provenance import (\n"
    "    ProvenanceLedger,\n"
    "    Provenance,\n"
    "    is_consistent,\n"
    "    canon,\n"
    ")\n"
    "\n"
    'REPORT = "data/annual_report.md"\n'
    "ledger = ProvenanceLedger(REPORT)\n"
    'print("ledger built over", REPORT)'
)

md(
    "## Listing 1 — resolve a segment-revenue triple to its table cell\n"
    "\n"
    "The triple `(cloud platform, has_revenue, 120.0)` was extracted from a table\n"
    "cell. `resolve` re-aligns it post-hoc and returns a `Provenance`: the line,\n"
    "the absolute char span, the raw cell text, and the alignment `method`."
)

# --- Cell 3: Listing 1 ---
code(
    'p = ledger.resolve("cloud platform", "has_revenue", "120.0")\n'
    'print("location:", p.location())\n'
    'print("method:  ", p.method)\n'
    'print("raw:     ", repr(p.raw))\n'
    'print("line:    ", repr(ledger.line(p.line_no)))'
)

md(
    "Expected output (grounded in the corpus, line 15 = the Cloud Platform row):\n"
    "\n"
    "```\n"
    "location: data/annual_report.md:15:553-558\n"
    "method:   table_cell\n"
    "raw:      '120.0'\n"
    "line:     '| Cloud Platform | Technology | 120.0 | 340 |'\n"
    "```\n"
    "\n"
    "The `method` is `table_cell`: the figure is anchored to a precise span, not\n"
    "parsed loose from prose. The `raw` substring is the cell text itself — byte‑\n"
    "exact `120.0`, the same value the store's ENM returns (Ch.~\\ref{ch:store})."
)

md(
    "## Listing 2 — a prose triple is `unaligned`\n"
    "\n"
    "The MD&A section says the Cloud Platform segment “continued to lead the\n"
    "company's topline.” That sentence carries no table cell, schema bullet, or\n"
    "section header for a `(cloud platform, leads, topline)` triple. Post-hoc\n"
    "structural alignment cannot find an anchor, so it resolves as `unaligned`."
)

# --- Cell 4: Listing 2 ---
code(
    'prose = ledger.resolve("cloud platform", "leads", "topline")\n'
    'print("location:", prose.location())\n'
    'print("method:  ", prose.method)\n'
    'print("raw:     ", repr(prose.raw))\n'
    'print("aligned? ", prose.method != "unaligned")'
)

md(
    "Expected output:\n"
    "\n"
    "```\n"
    "location: data/annual_report.md:-1:-1--1\n"
    "method:   unaligned\n"
    "raw:      ''\n"
    "aligned?  False\n"
    "```\n"
    "\n"
    "`line_no` and the char offsets are `-1`: the sentinel for *no structural\n"
    "anchor*. This is the honest signal that prose triples need spans captured at\n"
    "**extraction** time — `ProvenanceLedger` re-aligns regex-extracted structured\n"
    "triples (tables, schema bullets, headers) only. It does not invent a span for\n"
    "an LLM-extracted prose claim."
)

md(
    "## Listing 3 — the provenance-span trick: coarse triples cover prose\n"
    "\n"
    "A prose section *can* still get a span — not for a fine-grained fact, but at\n"
    "section granularity via an `in_section` triple keyed on the section slug. The\n"
    "ledger resolves it to the **section header** line, giving the answer layer a\n"
    "bounded region to cite even where no table fact exists."
)

# --- Cell 5: Listing 3 ---
code(
    "# section slugs are derived from headers: '## 6. Management Discussion and\n"
    "# Analysis' -> 'management_discussion_and_analysis'\n"
    'sec = ledger.resolve("md&a note", "in_section",\n'
    '                     "management_discussion_and_analysis")\n'
    'print("location:", sec.location())\n'
    'print("method:  ", sec.method)\n'
    'print("raw:     ", repr(sec.raw))'
)

md(
    "Expected output (the MD&A header is on line 60):\n"
    "\n"
    "```\n"
    "location: data/annual_report.md:60:1527-1567\n"
    "method:   section_header\n"
    "raw:      '## 6. Management Discussion and Analysis'\n"
    "```\n"
    "\n"
    "This is the *provenance-span trick*: a coarse `in_section` triple anchors a\n"
    "prose region to its header span. It does not make the prose ENM-checkable —\n"
    "it only gives a citable location. Fine numeric facts still come from tables\n"
    "(Listing 1)."
)

md(
    "## Listing 4 — verify a resolved span actually supports its triple\n"
    "\n"
    "Resolving to a span is not enough; the span must *support* the triple.\n"
    "`is_consistent` re-reads the raw substring and checks it against the tail —\n"
    "numeric-aware for table cells, slug-aware for headers and schema bullets."
)

# --- Cell 6: Listing 4 ---
code(
    'good = ledger.resolve("total", "has_revenue", "355.0")\n'
    'print("total revenue:", good.location(), "raw=", repr(good.raw),\n'
    '      "consistent=", is_consistent(good))\n'
    "\n"
    "# An unaligned prose triple can never be consistent.\n"
    'print("prose triple:  consistent=", is_consistent(prose))'
)

md(
    "Expected output:\n"
    "\n"
    "```\n"
    "total revenue: data/annual_report.md:19:698-703 raw= '355.0' consistent= True\n"
    "prose triple:  consistent= False\n"
    "```\n"
    "\n"
    "`is_consistent` parses `355.0` out of the raw cell and matches it to the tail\n"
    "within `1e-6`. The unaligned prose triple has no span to read, so it is\n"
    "correctly `False` — the system refuses to claim support it cannot point to."
)

md(
    "## Exercise (worked) — build a ledger over many triples and audit consistency\n"
    "\n"
    "Resolve a batch of segment triples with `ledger.build(...)` and confirm every\n"
    "aligned one is consistent."
)

# --- Cell 7: Exercise worked solution ---
code(
    "triples = [\n"
    '    ("cloud platform", "has_revenue", "120.0"),\n'
    '    ("devices", "has_revenue", "80.0"),\n'
    '    ("logistics", "has_revenue", "95.0"),\n'
    '    ("retail", "has_revenue", "60.0"),\n'
    '    ("total", "has_revenue", "355.0"),\n'
    "]\n"
    "provs = ledger.build(triples)\n"
    "for key, pr in provs.items():\n"
    '    flag = "OK " if is_consistent(pr) else "BAD"\n'
    '    print(flag, key, "->", pr.location(), repr(pr.raw))'
)

md(
    "Expected: every row prints `OK` — each segment-revenue triple aligns to a\n"
    "table cell whose raw value equals the tail. The Total row aligns too, so the\n"
    "sum anchor (Total = Σ segments, Ch.~\\ref{ch:geode}) has a provenance to cite\n"
    "if it ever breaks."
)

md(
    "## Self-check — a resolved triple's `raw` equals the table value\n"
    "\n"
    "The chapter's claim: a resolved segment-revenue triple carries the exact\n"
    "table cell as its provenance, and the cell supports the triple."
)

# --- Cell 8: self-check assert (FINAL) ---
code(
    'p = ledger.resolve("cloud platform", "has_revenue", "120.0")\n'
    "\n"
    "# 1. the raw span IS the table value, byte-exact\n"
    'assert p.raw == "120.0", p.raw\n'
    "# 2. it aligned to a table cell with a real char span\n"
    'assert p.method == "table_cell"\n'
    "assert p.line_no == 15 and p.char_start >= 0 and p.char_end > p.char_start\n"
    "# 3. the span actually supports the triple\n"
    "assert is_consistent(p)\n"
    "# 4. a prose triple with no structural anchor is unaligned + inconsistent\n"
    'u = ledger.resolve("cloud platform", "leads", "topline")\n'
    'assert u.method == "unaligned" and u.line_no == -1\n'
    "assert not is_consistent(u)\n"
    'print("OK: span<->triple provenance verified; prose triple correctly unaligned")'
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

out = "notebooks/03_provenance_a_inline.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
