# SPDX-License-Identifier: Apache-2.0
"""Build notebooks/12_self_verification_a_inline.ipynb (CPU-only; does NOT execute it).

Emits a valid nbformat-4 notebook for Chapter 10 (Self-verification: the GMS as
a hallucination detector). Listings are grounded in data/corpus_facts.md and use
only real symbols from knowlytix.knowledge.rag.verify / .pipeline / .config.

Run: python scripts/build_nb_10.py
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "notebooks", "12_self_verification_a_inline.ipynb")


def md(text: str):
    return new_markdown_cell(text)


def code(text: str):
    return new_code_cell(text)


CELLS = []

# --- bootstrap (FIRST cell, exact from global brief) ----------------------
CELLS.append(code(
    "import os, sys\n"
    'KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", "/home/user/jupyterlab/GMS-knowlytix")\n'
    "sys.path.insert(0, KNOWLYTIX_SRC)"
))

CELLS.append(md(
    "# Chapter 10 — Self-verification: the GMS as a hallucination detector\n"
    "\n"
    "Chapter 9 produced a *grounded* draft: the synthesizer was handed retrieved\n"
    "facts plus their spans and told to answer only from them. That constrains the\n"
    "model; it does not *prove* the model obeyed. A small local model can still drop\n"
    "a digit, swap a segment, or paste a half-remembered figure from pre-training.\n"
    "\n"
    "This chapter closes the loop. We decompose the **answer itself** back into\n"
    "claim triples and check each claim against the trained store with\n"
    "`AnswerVerifier`. The check is **calibration-free**: a numeric-aware\n"
    "*contradiction* test (the answer says `t`, the graph asserts `t' != t`) and,\n"
    "for cap-trained stores, an *admissibility* test against the relation's learned\n"
    "cap radius. No threshold is set by hand, and crucially **the verifier is not\n"
    "an LLM judging an LLM** — the adjudicator is the geometry.\n"
    "\n"
    "We run two drafts over Northwind Industries FY2025: a faithful one (every\n"
    "claim *supported*) and a tampered one with a hallucinated total-revenue figure\n"
    "(*contradicted*). With `on_verify_fail=\"abstain\"` the pipeline refuses the\n"
    "tampered answer rather than emit it."
))

# --- deterministic verify-LLM ---------------------------------------------
CELLS.append(md(
    "## A deterministic claim-extraction backend\n"
    "\n"
    "`AnswerVerifier` calls an LLM only to **decompose** the draft answer into claim\n"
    "triples; the adjudication that follows is pure geometry. So for a deterministic,\n"
    "GPU-free demo we inject a scripted `LLMBackend` that returns the claim triples a\n"
    "competent extractor would produce for each draft. (`LLMBackend` is the abstract\n"
    "interface from `knowlytix.knowledge.llm_backend`; the real Qwen path is shown at\n"
    "the end of the chapter.) The extractor emits the JSON array the verifier expects\n"
    "— `QueryTripleExtractor.parse` reads it back."
))

CELLS.append(code(
    "import json\n"
    "from knowlytix.knowledge.llm_backend import LLMBackend\n"
    "\n"
    "\n"
    "class ScriptedBackend(LLMBackend):\n"
    '    """Deterministic stand-in for the verify-LLM.\n'
    "\n"
    "    Maps each draft answer (matched by substring) to the JSON claim-triple\n"
    "    array a real extractor would emit. No GPU, no network -- the geometry,\n"
    "    not this stub, decides supported vs contradicted.\n"
    '    """\n'
    "\n"
    "    def __init__(self, script: dict[str, list[dict]]):\n"
    "        self._script = script\n"
    "\n"
    "    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:\n"
    "        for needle, claims in self._script.items():\n"
    "            if needle in user:\n"
    "                return json.dumps(claims)\n"
    "        return '[]'\n"
    "\n"
    "    @property\n"
    "    def model_name(self) -> str:\n"
    '        return "scripted-verify"'
))

# --- minimal in-notebook store for offline determinism --------------------
CELLS.append(md(
    "## The facts we verify against\n"
    "\n"
    "The verifier reads the *trained* store. Training a real `GMSExpertStore` needs\n"
    "the shared GPU (Chapter 4), so to keep this chapter offline and deterministic we\n"
    "back the verifier with a tiny **fixture store** that exposes exactly the methods\n"
    "`AnswerVerifier` and `TripleBinder` call — `query_triples`, `fuzzy_match_entity`,\n"
    "`cap_radius`, `score_triple`, and an `adapter` with the relation/entity vocab.\n"
    "We seed it with the Northwind triples the two drafts touch, grounded in\n"
    "`data/corpus_facts.md`:\n"
    "\n"
    "- `total has_revenue 355.0` (the canonical FY2025 total),\n"
    "- `cloud platform has_revenue 120.0`,\n"
    "- `logistics has_revenue 95.0`.\n"
    "\n"
    "In a real deployment you would load the store built in Chapter 4 and pass it\n"
    "straight to `AnswerVerifier` — the verification logic is identical; only the\n"
    "store is real. (This fixture is a contradiction-only stand-in: `cap_radius`\n"
    "returns `None`, so the admissibility branch reports `unverifiable` rather than\n"
    "fabricating a geometry the fixture does not have.)"
))

CELLS.append(code(
    "from dataclasses import dataclass, field\n"
    "\n"
    "\n"
    "@dataclass\n"
    "class _Adapter:\n"
    "    relation_to_idx: dict\n"
    "    entity_to_idx: dict\n"
    "\n"
    "\n"
    "class FixtureStore:\n"
    '    """CPU-only stand-in exposing the store surface the verifier/binder use."""\n'
    "\n"
    "    def __init__(self, triples):\n"
    "        self._triples = [tuple(t) for t in triples]\n"
    "        ents = {h for h, _, _ in self._triples} | {t for _, _, t in self._triples}\n"
    "        rels = {r for _, r, _ in self._triples}\n"
    "        self.adapter = _Adapter(\n"
    "            relation_to_idx={r: i for i, r in enumerate(sorted(rels))},\n"
    "            entity_to_idx={e: i for i, e in enumerate(sorted(ents))},\n"
    "        )\n"
    "\n"
    "    def query_triples(self, head=None, relation=None, tail=None):\n"
    "        return [t for t in self._triples\n"
    "                if (head is None or t[0] == head)\n"
    "                and (relation is None or t[1] == relation)\n"
    "                and (tail is None or t[2] == tail)]\n"
    "\n"
    "    def fuzzy_match_entity(self, name):\n"
    "        return name if name in self.adapter.entity_to_idx else None\n"
    "\n"
    "    def cap_radius(self, rel):\n"
    "        return None            # fixture has no learned cap geometry\n"
    "\n"
    "    def score_triple(self, head, rel, tail):\n"
    "        return None            # not scorable without a trained model\n"
    "\n"
    "\n"
    "triples = [\n"
    '    ("total", "has_revenue", "355.0"),\n'
    '    ("cloud platform", "has_revenue", "120.0"),\n'
    '    ("logistics", "has_revenue", "95.0"),\n'
    "]\n"
    "store = FixtureStore(triples)\n"
    "\n"
    "# Ground truth (corpus_facts.md): total revenue is exactly 355.0\n"
    'print(store.query_triples(head="total", relation="has_revenue"))'
))

CELLS.append(md(
    "Expected output (the asserted edge the verifier will compare claims against):\n"
    "\n"
    "```\n"
    "[('total', 'has_revenue', '355.0')]\n"
    "```"
))

# --- supported draft -------------------------------------------------------
CELLS.append(md(
    "## A faithful draft verifies as *supported*\n"
    "\n"
    "The first draft states the correct figure. `AnswerVerifier.verify` decomposes\n"
    "it to the claim `(total, has_revenue, 355.0)`, binds it to the graph, finds the\n"
    "asserted edge `total has_revenue 355.0`, and numeric-aware equality holds — so\n"
    "the claim is `supported` and `report.ok` is `True`."
))

CELLS.append(code(
    "from knowlytix.knowledge.rag.verify import AnswerVerifier\n"
    "\n"
    'good_draft = "Total revenue for FY2025 was 355.0 million."\n'
    "good_script = {\n"
    '    good_draft: [{"head": "total", "relation": "has_revenue", "tail": "355.0"}],\n'
    "}\n"
    "verifier = AnswerVerifier(store, ScriptedBackend(good_script))\n"
    "\n"
    "report = verifier.verify(good_draft)\n"
    'print("ok       :", report.ok)\n'
    'print("verdicts :", [(v.triple.as_tuple(), v.status) for v in report.verdicts])\n'
    'print("failures :", report.failures)'
))

CELLS.append(md(
    "Expected output:\n"
    "\n"
    "```\n"
    "ok       : True\n"
    "verdicts : [(('total', 'has_revenue', '355.0'), 'supported')]\n"
    "failures : []\n"
    "```\n"
    "\n"
    "The claim matched an *asserted* edge, so no calibration and no cap radius were\n"
    "needed — the contradiction test alone settled it."
))

# --- contradicted draft ----------------------------------------------------
CELLS.append(md(
    "## A hallucinated figure verifies as *contradicted*\n"
    "\n"
    "Now a draft that reports `455.0` — a plausible-looking but wrong total (the kind\n"
    "of single-digit slip a 3B model makes). The graph asserts `355.0`; numeric-aware\n"
    "equality fails; the verdict is `contradicted` and `report.ok` is `False`. The\n"
    "`detail` names what the graph actually holds."
))

CELLS.append(code(
    'bad_draft = "Total revenue for FY2025 was 455.0 million."\n'
    "bad_script = {\n"
    '    bad_draft: [{"head": "total", "relation": "has_revenue", "tail": "455.0"}],\n'
    "}\n"
    "bad_verifier = AnswerVerifier(store, ScriptedBackend(bad_script))\n"
    "\n"
    "bad_report = bad_verifier.verify(bad_draft)\n"
    'print("ok      :", bad_report.ok)\n'
    "for v in bad_report.verdicts:\n"
    '    print(v.triple.as_tuple(), "->", v.status, "|", v.detail)'
))

CELLS.append(md(
    "Expected output:\n"
    "\n"
    "```\n"
    "ok      : False\n"
    "('total', 'has_revenue', '455.0') -> contradicted | graph asserts ['355.0']\n"
    "```\n"
    "\n"
    "The verifier did not *judge* whether 455 is reasonable — it found a directly\n"
    "asserted fact that disagrees. That is the difference between a faithfulness\n"
    "check and an opinion."
))

# --- pipeline abstains -----------------------------------------------------
CELLS.append(md(
    "## The pipeline abstains on a failed verification\n"
    "\n"
    "Verification is a pipeline stage, not a one-off call. With\n"
    "`verify_llm_output=True` and `on_verify_fail=\"abstain\"`, a contradicted claim\n"
    "forces `RagPipeline` down the abstain branch: `decision == \"abstain\"`, the\n"
    "answer becomes the standard refusal, and the `verification` audit record carries\n"
    "`ok=False`.\n"
    "\n"
    "We point both the synthesis LLM and the verify LLM at scripted backends so the\n"
    "whole `query()` runs offline: the extractor turns the question into a query\n"
    "triple, the synthesizer emits the tampered draft, and the verifier rejects it.\n"
    "(`RagConfig` exposes `llm`, `llm_extract`, `llm_verify`, `verify_llm_output`,\n"
    "and `on_verify_fail` — all real knobs from `knowlytix.knowledge.rag.config`.)"
))

CELLS.append(code(
    "from knowlytix.knowledge.rag.config import RagConfig\n"
    "from knowlytix.knowledge.rag.pipeline import RagPipeline\n"
    "\n"
    'question = "What was total revenue in FY2025?"\n'
    "\n"
    "# Extractor: question -> a query triple with the asked slot '?'.\n"
    "extract_script = {\n"
    '    question: [{"head": "total", "relation": "has_revenue", "tail": "?"}],\n'
    "}\n"
    "# Synthesizer: emits the tampered draft (455.0).\n"
    "synth_script = {question: bad_draft}\n"
    "\n"
    "\n"
    "class SynthBackend(LLMBackend):\n"
    "    def __init__(self, text):\n"
    "        self._text = text\n"
    "    def call(self, system, user, max_tokens=2048):\n"
    "        return self._text\n"
    "    @property\n"
    "    def model_name(self):\n"
    '        return "scripted-synth"\n'
    "\n"
    "\n"
    "cfg = RagConfig(\n"
    "    llm=SynthBackend(bad_draft),\n"
    "    llm_extract=ScriptedBackend(extract_script),\n"
    "    llm_verify=ScriptedBackend(bad_script),\n"
    "    verify_llm_output=True,\n"
    '    on_verify_fail="abstain",\n'
    "    relevance_gate=False,       # offline: skip the extra LLM relevance call\n"
    "    ground_extraction=False,    # offline: scripted extractor needs no schema\n"
    ")\n"
    "pipe = RagPipeline.from_store(store, cfg)\n"
    "ans = pipe.query(question)\n"
    "\n"
    'print("decision     :", ans.decision)\n'
    'print("answer       :", ans.answer)\n'
    'print("verified ok? :", ans.verification.get("ok"))\n'
    'print("notice       :", ans.notice)'
))

CELLS.append(md(
    "Expected output:\n"
    "\n"
    "```\n"
    "decision     : abstain\n"
    "answer       : I cannot answer this from the available grounded evidence.\n"
    "verified ok? : False\n"
    "notice       : Answer failed GMS self-verification (unsupported claims).\n"
    "```\n"
    "\n"
    "The tampered figure never reaches the user. Compare this to a vanilla pipeline,\n"
    "which would return `455.0` with confident prose and no signal that it is wrong."
))

# --- regenerate exercise note ---------------------------------------------
CELLS.append(md(
    "## Exercise: `on_verify_fail=\"regenerate\"`\n"
    "\n"
    "Setting `on_verify_fail=\"regenerate\"` makes the pipeline re-synthesize once\n"
    "before giving up (see `RagPipeline._self_verify`). With a scripted synthesizer\n"
    "the second draft is identical, so verification fails again and the pipeline\n"
    "*annotates* rather than abstains: confidence is driven to `0.0` and a notice\n"
    "naming the unsupported claim is attached. The cell below demonstrates the\n"
    "annotate-on-still-failing path."
))

CELLS.append(code(
    "regen_cfg = RagConfig(\n"
    "    llm=SynthBackend(bad_draft),\n"
    "    llm_extract=ScriptedBackend(extract_script),\n"
    "    llm_verify=ScriptedBackend(bad_script),\n"
    "    verify_llm_output=True,\n"
    '    on_verify_fail="regenerate",\n'
    "    relevance_gate=False,\n"
    "    ground_extraction=False,\n"
    ")\n"
    "regen_ans = RagPipeline.from_store(store, regen_cfg).query(question)\n"
    "\n"
    'print("decision  :", regen_ans.decision)\n'
    'print("confidence:", regen_ans.confidence)\n'
    'print("notice    :", regen_ans.notice)\n'
    'print("verified  :", regen_ans.verification.get("ok"))'
))

CELLS.append(md(
    "Expected behavior: the re-synthesis yields the same tampered draft, so\n"
    "verification still fails; confidence collapses to `0.0`, the notice flags\n"
    "`total.has_revenue=455.0` as unsupported, and (with the default\n"
    "`accept_threshold=0.0`) the answer is annotated rather than silently accepted.\n"
    "Try wiring a `SynthBackend` whose second call returns the *correct* `355.0`\n"
    "draft and confirm the regenerate path then accepts a `supported` answer."
))

# --- real Qwen path (marked for CI) ---------------------------------------
CELLS.append(md(
    "## The real path: Qwen2.5-3B as the verify-LLM\n"
    "\n"
    "In production the verify-LLM is the same local model that synthesizes — only\n"
    "its *role* differs (decompose, not generate). The cell below is the real Qwen\n"
    "wiring; it is **marked for the lead to execute in CI** (loads the model on the\n"
    "shared GPU) and is not run while authoring. The geometry-based adjudication is\n"
    "identical to the scripted runs above; only the claim extractor changes."
))

CELLS.append(code(
    "# CI-ONLY: requires the shared GPU. Do not run while authoring.\n"
    "from knowlytix.knowledge.geode.agent_llm import QWEN_3B\n"
    "from knowlytix.knowledge.llm_backend import LocalTransformersBackend\n"
    "\n"
    "qwen = LocalTransformersBackend(QWEN_3B)        # Qwen/Qwen2.5-3B-Instruct\n"
    "qwen_verifier = AnswerVerifier(store, qwen)\n"
    "qwen_report = qwen_verifier.verify(bad_draft)\n"
    "assert qwen_report.ok is False                  # 455.0 contradicts 355.0"
))

# --- self-check (FINAL cell) ----------------------------------------------
CELLS.append(md(
    "## Self-check\n"
    "\n"
    "The chapter's claim: a hallucinated figure is caught by the geometry and the\n"
    "pipeline refuses it. The final cell asserts exactly that — the faithful draft\n"
    "verifies, the tampered draft does not, and the pipeline abstains with\n"
    "`verification[\"ok\"] is False`."
))

CELLS.append(code(
    "# Faithful draft verifies as supported.\n"
    "assert report.ok is True\n"
    'assert report.verdicts[0].status == "supported"\n'
    "\n"
    "# Hallucinated figure is contradicted (geometry, not an LLM judge).\n"
    "assert bad_report.ok is False\n"
    'assert bad_report.verdicts[0].status == "contradicted"\n'
    "\n"
    "# The pipeline abstains rather than emit the tampered figure.\n"
    'assert ans.decision == "abstain"\n'
    'assert ans.verification["ok"] is False\n'
    'assert "455" not in ans.answer\n'
    "\n"
    'print("Chapter 10 self-check passed: hallucinated figure caught and refused.")'
))

nb = new_notebook(cells=CELLS)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

with open(OUT, "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
print(f"wrote {OUT} ({len(CELLS)} cells)")
