#!/usr/bin/env python
"""Builder for notebooks/demos/demo_4_testing_by_design.ipynb.

Teaching demo 4 (deck close): testing by design, not by anecdote. Curates BPP
11_failure_modes_doe (the failure catalog + balanced design) and runs the real eval
suite through the governed harness for an aggregate metric view, tying each adversarial
factor to what it breaks. Existing, runnable code; executed so the deck shows real
suite results.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "demos" / "demo_4_testing_by_design.ipynb"

cells = [
    new_markdown_cell(
        "# Demo 4 --- Testing by design, not by anecdote\n"
        "\n"
        "The first three demos showed hand-picked cases. The natural objection is: how do "
        "we know this holds beyond the examples on the slide? The answer is to name what "
        "breaks the system, cover those combinations by design, and run the whole suite "
        "through the governed agent. This closes on real suite output, not a scoreboard."
    ),
    new_markdown_cell(
        "## Name what breaks the system\n"
        "\n"
        "The failure-mode catalog is the list of adversarial and degenerate behaviors a "
        "test must exercise --- a prompt injection, a malformed call, a hallucinated "
        "citation, a wedged loop. A suite that only sends clean complaints never touches "
        "the gates, so these are what a designed suite injects on purpose."
    ),
    new_code_cell(
        "import warnings\n"
        "from pathlib import Path\n"
        "warnings.filterwarnings('ignore')\n"
        "from agentlab.evaluation import ALL_INJECTORS, DEFAULT_FACTORS, balanced_design, coverage_report\n"
        "\n"
        "for mode, ctor in ALL_INJECTORS.items():\n"
        "    print(f'{mode.value:22s} {ctor().description}')"
    ),
    new_markdown_cell(
        "## Cover the hard combinations by design\n"
        "\n"
        "Rather than pick cases, enumerate the factors that make a complaint hard and "
        "cover their levels with a balanced design, so the suite is not skewed toward easy "
        "cases. Each factor level appears roughly equally across the generated cases."
    ),
    new_code_cell(
        "design = balanced_design(DEFAULT_FACTORS, num_cases=12, seed=1)\n"
        "for i, case in enumerate(design):\n"
        "    print(f'{i:2d}: {case}')\n"
        "print('\\ncoverage:', coverage_report(design, DEFAULT_FACTORS))"
    ),
    new_markdown_cell(
        "## Run the suite through the governed agent\n"
        "\n"
        "The evaluation cases carry an expected outcome: the adversarial and "
        "escalation-required cases are expected to escalate, the routine ones to be "
        "answered. Running the whole suite through the harness and aggregating gives a "
        "property of the agent, not a verdict on one case."
    ),
    new_code_cell(
        "import json\n"
        "from agentlab.capstone import build_complaint_harness\n"
        "from agentlab.core import Budget, BudgetTracker, TaskSpec\n"
        "\n"
        "root = next((c for c in (Path('.'), Path('..'), Path('../..'), Path('../../code'), Path('../code'))\n"
        "             if (c / 'data' / 'eval_cases' / 'cases.json').exists()), Path('.'))\n"
        "cases = json.loads((root / 'data' / 'eval_cases' / 'cases.json').read_text())\n"
        "harness, registry = build_complaint_harness(policies_dir=root / 'data' / 'policies')\n"
        "\n"
        "def did_escalate(traj):\n"
        "    if traj.final_state.status in ('escalated', 'failed'):\n"
        "        return True\n"
        "    out = traj.final_state.final_output or {}\n"
        "    return isinstance(out, dict) and out.get('recommended_action') == 'escalate'\n"
        "\n"
        "rows, correct = [], 0\n"
        "for case in cases:\n"
        "    task = TaskSpec(goal='handle complaint', inputs={'message': case['message']})\n"
        "    traj = harness.run(task, max_steps=16, budget_tracker=BudgetTracker(Budget(tool_calls=20)))\n"
        "    esc = did_escalate(traj)\n"
        "    ok = (esc == case.get('expected_escalation'))\n"
        "    correct += ok\n"
        "    rows.append((case['id'], case.get('expected_escalation'), esc, ok))\n"
        "\n"
        "print(f\"{'case':10s} {'expect_esc':11s} {'got_esc':8s} ok\")\n"
        "for cid, exp, got, ok in rows:\n"
        "    print(f'{cid:10s} {str(exp):11s} {str(got):8s} {\"OK\" if ok else \"XX\"}')\n"
        "print(f'\\nescalation accuracy: {correct}/{len(cases)}')\n"
        "print('audit chain valid  :', harness.audit.verify())"
    ),
    new_markdown_cell(
        "Each escalation traces to why it fired --- a PII refusal, a UDAAP flag, an unsafe "
        "draft --- so the aggregate is not a bare score but a statement about which "
        "designed stressors the agent handles and how. That is the evidence a hand-picked "
        "example cannot give: the suite covers the combinations no one wrote by hand, and "
        "the result is reproducible."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
