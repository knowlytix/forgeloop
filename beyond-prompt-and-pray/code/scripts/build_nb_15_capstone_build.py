#!/usr/bin/env python
"""Builder for notebooks/15_capstone_build.ipynb.

Capstone BUILD series, Chapter 15 (Capstone). The culmination: every layer built across
Chapters 1-14 is assembled by build_complaint_harness, and this notebook checks that the
assembled wiring -- the five typed tools and the three-gate stack -- is exactly what the
shipped harness returns, then runs routine and adversarial cases and verifies the audit
chain. Runs the real harness (Qwen models + GMS store).

Teaching notebook: real imports, no pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "15_capstone_build.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone build --- Chapter 15: The Assembled Capstone\n"
        "\n"
        "Each chapter of this series added one layer: the typed action (Chapter~1), the "
        "loop (Chapter~1), the reasoning trace (Chapter~3), the task and action types "
        "(Chapter~4), the five-tool action space (Chapter~5), the gate stack (Chapter~6), "
        "the budget (Chapter~7), the plan (Chapter~8), the working memory (Chapter~9), "
        "trajectory evaluation (Chapter~10), the designed suite (Chapter~11), runtime "
        "governance (Chapter~12) and escalation (Chapter~13). `build_complaint_harness` "
        "assembles them. This chapter checks that the assembled object is exactly the "
        "one the layers describe, then runs it."
    ),
    new_markdown_cell(
        "## Assemble the harness\n"
        "\n"
        "One call assembles the whole capstone: it registers the five tools, builds the "
        "policy engine, loads the trained plausibility gate and wraps the executor in "
        "the governance harness. The return is the `(harness, registry)` the shipped "
        "agent runs behind."
    ),
    new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "from agentlab.capstone import build_complaint_harness\n"
        "from agentlab.core import Budget, BudgetTracker, TaskSpec\n"
        "\n"
        "root = next((c for c in (Path('.'), Path('..'), Path('../code'), Path('code'))\n"
        "             if (c / 'data' / 'eval_cases' / 'cases.json').exists()), Path('.'))\n"
        "cases = json.loads((root / 'data' / 'eval_cases' / 'cases.json').read_text())\n"
        "harness, registry = build_complaint_harness(policies_dir=root / 'data' / 'policies')"
    ),
    new_markdown_cell(
        "## The wiring matches what the series built\n"
        "\n"
        "The claim of the series is that the layers add up to the shipped harness. That "
        "is checkable at the level of the wiring: the action space is exactly the five "
        "typed tools of Chapter~5, and the gate stack is exactly the three gates of "
        "Chapters~6 and~12 --- syntax, policy, plausibility --- with the trained gate "
        "last. The assertions below fail if the assembled harness and the described "
        "layers ever drift apart."
    ),
    new_code_cell(
        "from agentlab.governance import SyntaxGate\n"
        "from agentlab.gms_backend import GMSPlausibilityGate\n"
        "\n"
        "tool_names = {t.name for t in registry.all()}\n"
        "expected_tools = {'classify_complaint', 'extract_facts', 'search_policy',\n"
        "                  'flag_regulatory', 'draft_response'}\n"
        "assert tool_names == expected_tools, tool_names\n"
        "print('action space :', sorted(tool_names))\n"
        "\n"
        "gates = harness._executor._gates\n"
        "print('gate stack   :', [g.__class__.__name__ for g in gates])\n"
        "assert len(gates) == 3\n"
        "assert isinstance(gates[0], SyntaxGate)\n"
        "assert isinstance(gates[-1], GMSPlausibilityGate)\n"
        "print('wiring matches the shipped harness: OK')"
    ),
    new_markdown_cell(
        "## A routine case is handled\n"
        "\n"
        "A well-formed complaint runs the full workflow to a drafted reply: classify, "
        "extract, search policy, flag regulatory, draft. The final output carries the "
        "classification, the governing policy evidence and the draft."
    ),
    new_code_cell(
        "case = next(c for c in cases if c['id'] == 'case-002')\n"
        "task = TaskSpec(goal='handle complaint', inputs={'message': case['message']})\n"
        "traj = harness.run(task, max_steps=16, budget_tracker=BudgetTracker(Budget(tool_calls=20)))\n"
        "out = traj.final_state.final_output or {}\n"
        "print('message       :', case['message'])\n"
        "print('status        :', traj.final_state.status)\n"
        "print('classification :', out.get('classification'))\n"
        "print('action        :', out.get('recommended_action'))\n"
        "print('draft         :', (out.get('draft_response') or '')[:120])"
    ),
    new_markdown_cell(
        "## An adversarial case escalates\n"
        "\n"
        "A case that carries regulatory risk does not get a drafted reply: the agent "
        "escalates, and the audit chain records the decision. Verifying the chain "
        "confirms the run was not altered after the fact."
    ),
    new_code_cell(
        "case = next(c for c in cases if c['id'] == 'case-016')\n"
        "task = TaskSpec(goal='handle complaint', inputs={'message': case['message']})\n"
        "traj = harness.run(task, max_steps=16, budget_tracker=BudgetTracker(Budget(tool_calls=20)))\n"
        "esc = next((r for r in traj.records if r.action.kind == 'escalate'), None)\n"
        "print('message   :', case['message'])\n"
        "print('status    :', traj.final_state.status)\n"
        "print('escalation:', esc.action.reason if esc else '(none)')\n"
        "print('audit chain valid:', harness.audit.verify())\n"
        "assert harness.audit.verify()"
    ),
    new_markdown_cell(
        "The capstone is the sum of the layers: a fixed workflow of five typed tools, "
        "run behind a three-gate stack, recorded in a verifiable audit chain, escalating "
        "when it reaches the edge of its authority. What this series built by hand is the "
        "architecture; what it imported is the trained geometry and the language models "
        "inside the tools. The object assembled here is the one "
        "`build_complaint_harness` ships, and Chapter~16 tests it against the designed "
        "suite of Chapter~11."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
