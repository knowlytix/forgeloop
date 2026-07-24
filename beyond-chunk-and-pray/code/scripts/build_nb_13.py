#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit notebooks/13_abstention_and_coverage_a_inline.ipynb via nbformat (CPU only).

Does NOT execute the notebook: no store load, no Qwen, no GPU. The cells run
real library code (knowlytix.knowledge.rag.coverage) against the shipped corpus
text; GPU/Qwen cells are marked for the lead to execute in CI.
"""
from __future__ import annotations

import os

import nbformat as nbf

NB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "notebooks", "13_abstention_and_coverage_a_inline.ipynb"
)

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src))


# --- Cell 0: bootstrap (verbatim per global brief) ------------------------
code(
    "import os, sys\n"
    'KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", "/path/to/GMS-knowlytix")\n'
    "sys.path.insert(0, KNOWLYTIX_SRC)"
)

md(
    "# Chapter 11 — Honest abstention and coverage\n"
    "\n"
    "A trustworthy RAG system has to know, and *say*, what it does not know.\n"
    "Triple-mediated retrieval gives us a hard, measurable notion of *don't know*:\n"
    "if a question does not bind to a known entity/relation, or no grounded fact\n"
    "matches, the pipeline **abstains** instead of guessing. And because every\n"
    "answerable fact came from a triple with provenance, the regions of the\n"
    "document that carry **body text but no triples** are *measurable blind\n"
    "spots* — not a silent failure mode.\n"
    "\n"
    "This chapter shows three things over the Northwind FY2025 report:\n"
    "1. a prose (MD&A / Risk / Outlook) question **abstains with a notice**;\n"
    "2. `coverage_report(store)` flags exactly the prose sections as blind spots;\n"
    "3. `RagAnswer.audit()` emits a per-query audit trail."
)

# --- Cell: corpus paths ---------------------------------------------------
md(
    "## The corpus and its store\n"
    "\n"
    "We use the shipped Northwind FY2025 report and its trained store. The store\n"
    "was built by `scripts/build_store.py` (see Foundation brief F2); we never\n"
    "rebuild it here."
)
code(
    "DATA = os.path.join(os.path.dirname(os.getcwd()), \"data\") \\\n"
    "    if os.path.basename(os.getcwd()) == \"notebooks\" else \"data\"\n"
    "REPORT_MD = os.path.join(DATA, \"annual_report.md\")\n"
    "STORE_PATH = os.path.join(DATA, \"gms_annual_report_store\")\n"
    "with open(REPORT_MD) as f:\n"
    "    markdown = f.read()\n"
    "print(f\"report: {REPORT_MD}\")\n"
    "print(f\"store:  {STORE_PATH}\")"
)

# --- Listing 2: coverage_report -------------------------------------------
md(
    "## Listing 2 — the coverage monitor\n"
    "\n"
    "`coverage_report` walks the document section by section, buckets each\n"
    "content triple by the source line its provenance resolves to, and reports\n"
    "which sections have body text but zero triples. Those are the blind spots a\n"
    "triple-mediated pipeline cannot reach.\n"
    "\n"
    "We can compute coverage from just the markdown plus the store's triple list\n"
    "— no model forward pass — so this cell runs on CPU. The trained store ships\n"
    "its triple list as `triples.json` on disk; we read it directly and wrap it\n"
    "in a tiny stand-in object carrying `.markdown` and `.triples`, which is all\n"
    "`coverage_report` reads. In production you pass the real `GMSExpertStore`."
)
code(
    "import json\n"
    "from dataclasses import dataclass, field\n"
    "from knowlytix.knowledge.rag import coverage_report\n"
    "\n"
    "# The trained store's triple list, read from disk (CPU-only, no model load).\n"
    "with open(os.path.join(STORE_PATH, \"triples.json\")) as f:\n"
    "    TRIPLES = [tuple(t) for t in json.load(f)]\n"
    "print(f\"{len(TRIPLES)} triples loaded from the store (incl. structural in_section)\")\n"
    "\n"
    "@dataclass\n"
    "class _StoreView:\n"
    "    \"\"\"Minimal view coverage_report needs: .markdown and .triples.\"\"\"\n"
    "    markdown: str\n"
    "    triples: list = field(default_factory=list)\n"
    "    store_path: str = \"\"\n"
    "\n"
    "store_view = _StoreView(markdown=markdown, triples=TRIPLES, store_path=STORE_PATH)\n"
    "report = coverage_report(store_view)\n"
    "\n"
    "print(f\"coverage_ratio = {report.coverage_ratio:.2f}\")\n"
    "for r in report.regions:\n"
    "    mark = \"x\" if r.covered else \" \"\n"
    "    flag = \"  <-- BLIND SPOT\" if r.blind_spot else \"\"\n"
    "    print(f\"[{mark}] {r.title} ({r.triple_count} triples, \"\n"
    "          f\"{r.body_lines} body lines){flag}\")\n"
    "print()\n"
    "print(\"blind spots:\", [r.title for r in report.blind_spots])"
)
md(
    "Expected output (matches `data/corpus_facts.md`):\n"
    "\n"
    "```\n"
    "coverage_ratio = 0.56\n"
    "[ ] Northwind Industries — Annual Report FY2025 (0 triples, 4 body lines)  <-- BLIND SPOT\n"
    "[x] 1. Segment Performance (15 triples, 9 body lines)\n"
    "[x] 2. Divisions (4 triples, 5 body lines)\n"
    "[x] 3. Income Statement (8 triples, 7 body lines)\n"
    "[x] 4. Balance Sheet (3 triples, 6 body lines)\n"
    "[x] 5. Corporate Facts (4 triples, 6 body lines)\n"
    "[ ] 6. Management Discussion and Analysis (0 triples, 9 body lines)  <-- BLIND SPOT\n"
    "[ ] 7. Risk Factors (0 triples, 8 body lines)  <-- BLIND SPOT\n"
    "[ ] 8. Outlook (0 triples, 5 body lines)  <-- BLIND SPOT\n"
    "\n"
    "blind spots: ['Northwind Industries — Annual Report FY2025', "
    "'6. Management Discussion and Analysis', '7. Risk Factors', '8. Outlook']\n"
    "```\n"
    "\n"
    "Roughly half the document's content-bearing regions carry no triple. That is\n"
    "not a bug — those sections are qualitative prose (MD&A, Risk Factors,\n"
    "Outlook) that the report itself defers to the tables. The point is that the\n"
    "gap is **named and counted**, not hidden."
)

# --- Listing 1: abstention via pipeline (Qwen) ----------------------------
md(
    "## Listing 1 — a prose question abstains (real Qwen path)\n"
    "\n"
    "Now the online path. We load the trained store and run a Risk-Factor / MD&A\n"
    "question through `RagPipeline`. The Outlook and Risk Factors sections have no\n"
    "triples, so nothing binds — the bind-check fires and the pipeline abstains\n"
    "with a notice rather than fabricating a paragraph.\n"
    "\n"
    "> **CI-only cell.** This loads the store and runs Qwen2.5-3B-Instruct on the\n"
    "> GPU. Do not run it during authoring; the lead executes it in CI."
)
code(
    "# === CI-only (loads store + Qwen on GPU) ===\n"
    "from knowlytix.knowledge.store import GMSExpertStore\n"
    "from knowlytix.knowledge.config import DocGMSConfig\n"
    "from knowlytix.knowledge.llm_backend import LocalTransformersBackend\n"
    "from knowlytix.knowledge.rag import RagConfig, RagPipeline\n"
    "from knowlytix.knowledge.geode import QWEN_3B\n"
    "\n"
    "store = GMSExpertStore(DocGMSConfig(store_path=STORE_PATH))\n"
    "assert store.load(), \"store failed to load\"\n"
    "\n"
    "qwen = LocalTransformersBackend(QWEN_3B)\n"
    "rag = RagConfig(llm=qwen)            # bank-grade defaults: dense off, abstain-not-guess\n"
    "pipe = RagPipeline.from_store(store, rag)\n"
    "\n"
    "prose_q = \"What are the company's main risk factors?\"\n"
    "ans = pipe.query(prose_q)\n"
    "print(\"decision:\", ans.decision)\n"
    "print(\"route:   \", ans.route)\n"
    "print(\"notice:  \", ans.notice)\n"
    "print(\"answer:  \", ans.answer)\n"
    "assert ans.decision == \"abstain\"\n"
    "assert ans.notice is not None"
)
md(
    "Expected: `decision: abstain`, `route: triple`, and a notice such as\n"
    "*\"Question did not bind to known entities/relations.\"* The answer is the\n"
    "standard refusal string, not a guess. Contrast a chunk-and-pray baseline,\n"
    "which would happily summarize the Risk Factors prose and present it as an\n"
    "answer with no way to verify it."
)

# --- Listing 3: audit trail -----------------------------------------------
md(
    "## Listing 3 — the per-query audit trail\n"
    "\n"
    "Every answer — accepted or abstained — carries a structured audit record:\n"
    "the decision, the route, the query triples, what bound, the source facts and\n"
    "their provenance spans, the verification verdicts, and the notice. This is\n"
    "the bank-grade logging surface; wire it to `RagConfig.audit_sink` to capture\n"
    "every query automatically.\n"
    "\n"
    "> **CI-only cell** (uses the `ans` produced above)."
)
code(
    "# === CI-only (uses `ans` from Listing 1) ===\n"
    "import json\n"
    "audit = ans.audit()\n"
    "print(json.dumps(audit, indent=2, default=str))\n"
    "assert audit[\"decision\"] == \"abstain\"\n"
    "assert audit[\"route\"] == \"triple\"\n"
    "assert audit[\"verified\"] is True   # abstaining IS a verified outcome"
)
md(
    "The audit shows `decision: abstain` with an empty `sources` list — there was\n"
    "no grounded fact to cite, which is exactly why the system declined. Note\n"
    "`verified: true`: an honest abstention is a *verified* outcome, not an\n"
    "error. Compare this trail to a dense-retrieval answer in Chapter 14, which\n"
    "carries `verified=False` and a quarantine notice."
)

# --- Exercise -------------------------------------------------------------
md(
    "## Exercise — clear a blind spot\n"
    "\n"
    "Add a triple to a blind-spot section and watch coverage clear. There is a\n"
    "subtlety the library enforces honestly: `coverage_report` only counts a\n"
    "triple toward a section if its *provenance resolves* into that section. The\n"
    "`ProvenanceLedger` resolves table cells, section headers (`in_section`), and\n"
    "schema bullets — a triple invented out of thin air resolves to line `-1`\n"
    "(`unaligned`) and counts toward nothing. So clearing a blind spot is not a\n"
    "matter of asserting a fact; the fact must be *anchored* to real source text.\n"
    "\n"
    "The Outlook section header is itself a resolvable anchor, so we add an\n"
    "`(outlook, in_section, outlook)` triple. Because `coverage_report` excludes\n"
    "structural relations by default, we pass `exclude_relations=()` to count it."
)
code(
    "from knowlytix.knowledge.geode.provenance import ProvenanceLedger\n"
    "ledger = ProvenanceLedger.from_text(markdown)\n"
    "\n"
    "# An invented content triple does NOT resolve -> it cannot clear a blind spot.\n"
    "fake = (\"cloud platform\", \"has_outlook\", \"continued investment\")\n"
    "print(\"invented triple resolves to line:\", ledger.resolve(*fake).line_no,\n"
    "      \"(unaligned)\")\n"
    "\n"
    "# An anchored triple keyed to the real Outlook header DOES resolve.\n"
    "outlook_triple = (\"outlook\", \"in_section\", \"outlook\")\n"
    "print(\"anchored triple resolves to line:\", ledger.resolve(*outlook_triple).line_no)\n"
    "\n"
    "augmented = _StoreView(markdown=markdown,\n"
    "                       triples=TRIPLES + [outlook_triple],\n"
    "                       store_path=STORE_PATH)\n"
    "# Count structural relations too, so the anchored triple registers.\n"
    "after = coverage_report(augmented, exclude_relations=())\n"
    "titles = [r.title for r in after.blind_spots]\n"
    "print(\"blind spots after:\", titles)\n"
    "print(f\"coverage_ratio: {report.coverage_ratio:.2f} -> {after.coverage_ratio:.2f}\")\n"
    "assert \"8. Outlook\" not in titles"
)
md(
    "`8. Outlook` drops out of `blind_spots` and the coverage ratio rises\n"
    "(0.56 → 0.67). The lesson is twofold. First, the remedy for a blind spot is\n"
    "*more anchored triples* — re-ingest the section, broaden extraction — not\n"
    "turning on distrusted dense retrieval (the opt-in escape hatch in\n"
    "Chapter 14, flagged `verified=False` for a reason). Second, the coverage\n"
    "monitor cannot be gamed: an unanchored claim resolves to `-1` and clears\n"
    "nothing, so the ratio only moves when real evidence backs the new triple."
)

# --- Honest limits --------------------------------------------------------
md(
    "## Honest limits\n"
    "\n"
    "The coverage monitor measures *triple presence per section*, not *answer\n"
    "quality*. A section with triples can still fail to answer a specific\n"
    "question (the triple exists but the asked attribute differs — that is the\n"
    "relevance gate's job, Chapter 10), and a covered section's triples can be\n"
    "wrong if extraction erred (that is GEODE's job, Chapter 5). Coverage also\n"
    "says nothing about whether a blind spot *should* have been triplified —\n"
    "qualitative prose like Risk Factors legitimately carries no authoritative\n"
    "fact, so a 0.56 ratio is not a defect to drive to 1.0. Finally, abstention\n"
    "here is binary: the pipeline does not rank *how close* it came to binding,\n"
    "so a near-miss paraphrase and a wholly off-topic question both simply\n"
    "abstain. Calibrating the bind threshold so paraphrases resolve while\n"
    "off-topic queries still abstain is Chapter 12."
)

# --- Self-check (CPU only) ------------------------------------------------
md(
    "## Self-check\n"
    "\n"
    "The chapter's claim: the prose sections are measurable blind spots, and the\n"
    "coverage monitor names them. This runs CPU-only against the real library\n"
    "(no store, no Qwen)."
)
code(
    "blind = {r.title for r in report.blind_spots}\n"
    "assert \"7. Risk Factors\" in blind\n"
    "assert \"8. Outlook\" in blind\n"
    "assert \"6. Management Discussion and Analysis\" in blind\n"
    "# The fact-bearing tables are NOT blind spots.\n"
    "assert \"1. Segment Performance\" not in blind\n"
    "assert \"3. Income Statement\" not in blind\n"
    "assert abs(report.coverage_ratio - 0.56) < 0.01\n"
    "print(\"OK: prose sections are measurable blind spots; tables are covered.\")"
)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
with open(NB_PATH, "w") as f:
    nbf.write(nb, f)
print(f"wrote {os.path.abspath(NB_PATH)} ({len(cells)} cells)")
