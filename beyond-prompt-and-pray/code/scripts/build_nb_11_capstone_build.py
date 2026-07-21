#!/usr/bin/env python
"""Builder for notebooks/11_capstone_build.ipynb.

Capstone BUILD series, Chapter 11 (Failure Modes and DoE). Rather than test the agent
on hand-picked cases, enumerate the factors that make a case hard and cover their
combinations with a balanced design; inject the failure modes an adversary would.

DoE and test-case generation are pure (no model load). Teaching notebook: real
imports, no pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "11_capstone_build.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone build --- Chapter 11: Failure Modes and Design of Experiments\n"
        "\n"
        "Chapter~10 scored one trajectory. A handful of trajectories does not establish "
        "that an agent is safe, because the cases that break it are the ones no one "
        "thought to write. Chapter~11 replaces hand-picked cases with a design: name the "
        "factors that make a complaint hard --- product, regulatory exposure, whether the "
        "message carries PII or an injection attempt --- and cover their combinations "
        "with a balanced set. It then names the failure modes a test must inject to "
        "exercise the agent's defenses rather than only its happy path."
    ),
    new_markdown_cell(
        "## A balanced design over the hard factors\n"
        "\n"
        "`balanced_design` builds a set of factor combinations that covers each level of "
        "each factor roughly equally, so the test suite is not accidentally skewed "
        "toward easy cases. The factors below are the dimensions along which a complaint "
        "case varies in ways that stress the agent."
    ),
    new_code_cell(
        "from agentlab.evaluation.doe import balanced_design, coverage_report\n"
        "\n"
        "factors = {\n"
        "    'product': ['checking_account', 'credit_card', 'mortgage'],\n"
        "    'regulatory': ['none', 'UDAAP', 'reg_x'],\n"
        "    'adversarial': ['clean', 'pii', 'prompt_injection'],\n"
        "}\n"
        "design = balanced_design(factors, num_cases=12, seed=0)\n"
        "for i, case in enumerate(design):\n"
        "    print(f'{i:2d}: {case}')\n"
        "print('coverage:', coverage_report(design, factors))"
    ),
    new_markdown_cell(
        "## The failure modes a test injects\n"
        "\n"
        "A test that only sends well-formed complaints never exercises the gates. The "
        "failure-mode catalog names the adversarial and degenerate behaviors a suite "
        "must include: a prompt-injection payload, a malformed call, a hallucinated "
        "citation, a wedged loop. Each is a factory returning a `FailureInjection` that "
        "can be applied to a scenario."
    ),
    new_code_cell(
        "from agentlab.evaluation.failure_modes import (\n"
        "    prompt_injection, malformed_call, hallucinated_citation, infinite_loop,\n"
        ")\n"
        "for factory in (prompt_injection, malformed_call, hallucinated_citation, infinite_loop):\n"
        "    inj = factory()\n"
        "    print(f'{inj.mode!s:24s} {inj.description}')"
    ),
    new_markdown_cell(
        "## From design to test cases\n"
        "\n"
        "`generate_test_cases` turns the design into `TestCase`s, each carrying a "
        "`TaskSpec`, the user message, the expected behavior and the factor combination "
        "it realizes. This is the suite Chapter~16 runs the whole agent against; here it "
        "shows the shape of a generated case."
    ),
    new_code_cell(
        "from agentlab.evaluation.test_cases import generate_test_cases\n"
        "\n"
        "cases = generate_test_cases(num_cases=6, factors=factors, seed=0)\n"
        "for c in cases[:3]:\n"
        "    print(f'[{c.id}] expect={c.expected_behavior!r:20s} factors={c.factors}')\n"
        "    print('      message:', c.user_message)"
    ),
    new_markdown_cell(
        "A designed suite is what turns a demonstration into evidence: it covers the "
        "hard combinations by construction and injects the failures an adversary would, "
        "so a passing run says something about cases no one wrote by hand. Chapter~12 "
        "assembles the governed harness these cases run against, and Chapter~16 runs the "
        "full suite and reports coverage."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
