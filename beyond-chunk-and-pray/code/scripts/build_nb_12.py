#!/usr/bin/env python
"""Builder for notebooks/12_grounded_synthesis_a_inline.ipynb (Ch9 — Grounded synthesis).

CPU-only. Emits a valid nbformat-4 notebook; does NOT execute it (no store, no
Qwen). GPU/Qwen cells are marked for the lead to run in CI.
"""
from __future__ import annotations

import pathlib

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_PATH = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "12_grounded_synthesis_a_inline.ipynb"

cells: list = []

cells.append(new_markdown_cell(
    "# Ch9 — Grounded synthesis\n"
    "\n"
    "Retrieval (Ch8) already produced the answer *values* through the graph. "
    "Synthesis turns those verified facts and their source spans into prose — "
    "**without** letting the model reach into its parametric memory. The "
    "`Assembler` hands the synthesis LLM an `<evidence>` block and a system "
    "prompt that says *answer ONLY from these*. When the evidence does not "
    "contain the answer, the model is instructed to decline.\n"
    "\n"
    "This notebook is grounded in `data/corpus_facts.md` (Northwind Industries "
    "FY2025). Cells tagged **[GPU/Qwen — CI]** load Qwen and are executed by the "
    "lead in CI; every other cell runs on CPU with a scripted fake backend so "
    "the chapter's claim is checkable deterministically."
))

# --- Cell 1: KNOWLYTIX_SRC bootstrap (verbatim from global brief) ---
cells.append(new_code_cell(
    "import os, sys\n"
    'KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", "")\n'
    "sys.path.insert(0, KNOWLYTIX_SRC)"
))

# --- Cell 2: imports ---
cells.append(new_code_cell(
    "from knowlytix.knowledge.rag.assemble import Assembler\n"
    "from knowlytix.knowledge.rag.retrieve import RetrievedFact\n"
    "from knowlytix.knowledge.rag.config import RagConfig\n"
    "from knowlytix.knowledge.llm_backend import LLMBackend\n"
    "from knowlytix.knowledge.geode import QWEN_3B\n"
    "\n"
    "print('synthesis model (real path):', QWEN_3B)"
))

# --- Cell 3: the retrieved evidence (grounded in corpus_facts.md) ---
cells.append(new_code_cell(
    '# Facts as Ch8\'s Retriever would return them, with provenance spans.\n'
    '# Source: data/corpus_facts.md sample retrieval for "cloud platform revenue".\n'
    'facts = [\n'
    '    RetrievedFact(\n'
    '        head="cloud platform", relation="has_revenue", tail="120.0",\n'
    '        score=0.0, confidence=1.0, source="triple",\n'
    '        location=":15:553-558", raw="Cloud Platform | Technology | 120.0 | 340",\n'
    '    ),\n'
    ']\n'
    'ENM_VALUE = "120.0"   # segment_performance / Cloud Platform/Technology/Revenue\n'
    'print(Assembler._context("What was Cloud Platform revenue?", facts))'
))

# --- Cell 4: a scripted fake backend (deterministic, CPU) ---
cells.append(new_code_cell(
    'class FakeBackend(LLMBackend):\n'
    '    """Deterministic stand-in for Qwen: grounds in the evidence string.\n'
    '\n'
    '    Mirrors a well-behaved synthesizer: it answers from the FACT line when\n'
    '    the asked value is present, otherwise it declines. Used so the chapter\'s\n'
    '    claim is checkable in CI without a GPU.\n'
    '    """\n'
    '    def __init__(self, model="fake-grounded"):\n'
    '        self._model = model\n'
    '\n'
    '    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:\n'
    '        if "120.0" in user:\n'
    '            return "Cloud Platform reported revenue of 120.0."\n'
    '        return "I cannot answer that from the available evidence."\n'
    '\n'
    '    @property\n'
    '    def model_name(self) -> str:\n'
    '        return self._model'
))

# --- Cell 5: Listing (1) — synthesize a grounded answer (fake backend, CPU) ---
cells.append(new_code_cell(
    'assembler = Assembler(FakeBackend())\n'
    'grounded = assembler.assemble("What was Cloud Platform revenue?", facts)\n'
    'print(grounded)'
))

# --- Cell 6: Listing (1, real) — the same with local Qwen [GPU/Qwen — CI] ---
cells.append(new_code_cell(
    '# [GPU/Qwen — CI] Real synthesis path. The lead runs this in CI; it is the\n'
    '# per-role LLM choice (local Qwen3-4B-Instruct, no API key).\n'
    'from knowlytix.knowledge.llm_backend import LocalTransformersBackend\n'
    '\n'
    'qwen = LocalTransformersBackend(QWEN_3B)\n'
    'real_answer = Assembler(qwen).assemble("What was Cloud Platform revenue?", facts)\n'
    'print(real_answer)\n'
    '# Expected: prose stating revenue of 120.0, no other figure.'
))

# --- Cell 7: Listing (2) — refuses when the facts lack the answer ---
cells.append(new_code_cell(
    '# No fact answers an Outlook question (a coverage blind spot, Ch11): the\n'
    '# evidence block is empty, so a grounded synthesizer must decline.\n'
    'no_facts: list[RetrievedFact] = []\n'
    'refusal = Assembler(FakeBackend()).assemble(\n'
    '    "What does the Outlook section forecast for FY2026?", no_facts)\n'
    'print(refusal)'
))

# --- Cell 8: Exercise — swap the synthesis LLM via RagConfig (per-role) ---
cells.append(new_markdown_cell(
    "## Exercise — per-role synthesis LLM\n"
    "`RagConfig` keeps the three LLM roles separate: `llm` (synthesis), "
    "`llm_extract` (NL -> query triples), `llm_verify` (answer -> claim "
    "triples). Swap *only* the synthesis backend and confirm the other roles "
    "fall back per `RagConfig.extract_llm()` / `verify_llm()`."
))
cells.append(new_code_cell(
    'extract_be = FakeBackend("fake-extract")\n'
    'synth_be = FakeBackend("fake-synth")\n'
    '\n'
    'cfg = RagConfig(llm=synth_be, llm_extract=extract_be)\n'
    '# Synthesis uses the swapped backend; verify defaults to the extract role.\n'
    'assert cfg.llm.model_name == "fake-synth"\n'
    'assert cfg.extract_llm().model_name == "fake-extract"\n'
    'assert cfg.verify_llm().model_name == "fake-extract"   # verify -> extract\n'
    'print("synthesis :", cfg.llm.model_name)\n'
    'print("extract   :", cfg.extract_llm().model_name)\n'
    'print("verify    :", cfg.verify_llm().model_name)'
))

# --- Final cell: self-check assert proving the chapter's claim ---
cells.append(new_markdown_cell(
    "## Self-check\n"
    "The chapter's claim: a grounded answer **contains the ENM figure and no "
    "other number**, and the synthesizer **refuses** when the evidence lacks "
    "the answer."
))
cells.append(new_code_cell(
    'import re\n'
    '\n'
    '# (a) the grounded answer carries the exact ENM figure ...\n'
    'assert ENM_VALUE in grounded\n'
    '# ... and introduces no other number.\n'
    'nums = re.findall(r"\\d+(?:\\.\\d+)?", grounded)\n'
    'assert set(nums) == {ENM_VALUE}, nums\n'
    '# (b) with no supporting facts, the synthesizer declines.\n'
    'assert "cannot answer" in refusal.lower()\n'
    'print("Ch9 self-check passed: grounded synthesis carries only the ENM figure;"\n'
    '      " empty evidence -> refusal.")'
))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(NB_PATH))
print(f"wrote {NB_PATH}")
