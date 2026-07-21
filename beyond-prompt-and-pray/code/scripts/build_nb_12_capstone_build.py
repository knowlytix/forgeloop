#!/usr/bin/env python
"""Builder for notebooks/12_capstone_build.ipynb.

Capstone BUILD series, Chapter 12 (Runtime Governance). Completes the gate stack begun
in Chapter 6 with the trained GMS plausibility gate, and wraps the executor in a
GovernanceHarness that logs every transition to a tamper-evident audit chain. This
notebook runs the real harness (Qwen models + GMS store).

Teaching notebook: real imports, no pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "12_capstone_build.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone build --- Chapter 12: Runtime Governance\n"
        "\n"
        "Chapter~6 put a two-gate stack in front of the tools: a syntax check and the "
        "banking policies. Chapter~12 adds the third gate and the record. The third gate "
        "is the trained GMS plausibility gate, which scores each workflow transition "
        "against the banking store's learned dependency structure and refuses a step "
        "that does not follow from the last. The record is the audit log, a "
        "tamper-evident chain the harness writes as it runs, so every decision can be "
        "replayed and verified after the fact."
    ),
    new_markdown_cell(
        "## The complete gate stack\n"
        "\n"
        "`build_complaint_harness` assembles the governance layer: the registry, the "
        "policy engine and the plausibility gate, wrapped in a `GovernedToolExecutor` "
        "inside a `GovernanceHarness`. Reading the executor's gates shows the three-gate "
        "stack the shipped agent runs behind --- syntax, policy, plausibility."
    ),
    new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "from agentlab.capstone import build_complaint_harness\n"
        "from agentlab.gms_backend import GMSPlausibilityGate\n"
        "\n"
        "root = next((c for c in (Path('.'), Path('..'), Path('../code'), Path('code'))\n"
        "             if (c / 'data' / 'eval_cases' / 'cases.json').exists()), Path('.'))\n"
        "harness, registry = build_complaint_harness(policies_dir=root / 'data' / 'policies')\n"
        "gates = harness._executor._gates\n"
        "for g in gates:\n"
        "    print(f'{g.__class__.__name__:22s} trained={isinstance(g, GMSPlausibilityGate)}')"
    ),
    new_markdown_cell(
        "The plausibility gate is the one component the reader does not build by hand: "
        "it is loaded from the calibrated GMS banking store, and its threshold is the "
        "one persisted at calibration time. This is the build-not-train boundary of the "
        "series --- the geometry is trained and imported, the stack it sits in is "
        "assembled here."
    ),
    new_markdown_cell(
        "## Governance produces an audit chain\n"
        "\n"
        "Running a case through the harness logs an event per step. The audit log hashes "
        "each state and links the events, so `verify` can confirm the chain was not "
        "altered. This is what makes a run accountable: the sequence of decisions is "
        "recorded, ordered and checkable."
    ),
    new_code_cell(
        "from agentlab.core import Budget, BudgetTracker, TaskSpec\n"
        "\n"
        "cases = json.loads((root / 'data' / 'eval_cases' / 'cases.json').read_text())\n"
        "case = next(c for c in cases if c['id'] == 'case-002')\n"
        "task = TaskSpec(goal='handle complaint', inputs={'message': case['message']})\n"
        "traj = harness.run(task, max_steps=16, budget_tracker=BudgetTracker(Budget(tool_calls=20)))\n"
        "print('final status     :', traj.final_state.status)\n"
        "print('audit events     :', len(harness.audit.events))\n"
        "print('audit chain valid:', harness.audit.verify())"
    ),
    new_markdown_cell(
        "## A gate refusal is recorded, not silent\n"
        "\n"
        "When a gate denies a call the harness records a failed step; the agent reads "
        "that failure and escalates. The governance layer never drops a decision "
        "silently --- a refusal is an auditable event like any other. The audit log below "
        "shows the ordered statuses the run passed through."
    ),
    new_code_cell(
        "for rec in traj.records:\n"
        "    print(f'step {rec.step}: {rec.action.kind:10s} -> {rec.state_after.status}')"
    ),
    new_markdown_cell(
        "Runtime governance is the gate stack and the audit chain together: the stack "
        "decides what may run, and the chain records what did. The harness assembled "
        "here is the object `build_complaint_harness` returns; Chapter~13 traces the "
        "escalation paths out of it, and Chapter~15 assembles the whole capstone and "
        "checks its wiring against the shipped harness."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
