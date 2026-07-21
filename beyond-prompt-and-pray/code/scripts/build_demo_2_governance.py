#!/usr/bin/env python
"""Builder for notebooks/demos/demo_2_governance.ipynb.

Teaching demo 2: governance at the decision point. Curates the mechanisms from BPP
06_safe_tool_execution, 12_runtime_governance and 13_escalation, applied to the real
banking harness: the three-gate stack, a call refused at the gate (PII), an escalation
on a regulatory flag, and the hash-chained audit log. Existing, runnable code; executed
so the deck shows real gate decisions.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "demos" / "demo_2_governance.ipynb"

cells = [
    new_markdown_cell(
        "# Demo 2 --- Governance at the decision point\n"
        "\n"
        "A governed agent is bounded, not just capable. Every tool call passes a stack of "
        "gates before it runs; a call that fails a gate is refused, and a case that "
        "reaches the edge of the agent's authority escalates to a human. Each decision is "
        "written to a hash-chained audit log. This runs on the real harness."
    ),
    new_code_cell(
        "import json, warnings\n"
        "from pathlib import Path\n"
        "warnings.filterwarnings('ignore')\n"
        "from agentlab.capstone import build_complaint_harness\n"
        "from agentlab.core import Budget, BudgetTracker, TaskSpec\n"
        "from agentlab.gms_backend import GMSPlausibilityGate\n"
        "\n"
        "root = next((c for c in (Path('.'), Path('..'), Path('../..'), Path('../../code'), Path('../code'))\n"
        "             if (c / 'data' / 'eval_cases' / 'cases.json').exists()), Path('.'))\n"
        "cases = {c['id']: c for c in json.loads((root / 'data' / 'eval_cases' / 'cases.json').read_text())}\n"
        "harness, registry = build_complaint_harness(policies_dir=root / 'data' / 'policies')\n"
        "\n"
        "def run(cid):\n"
        "    task = TaskSpec(goal='handle complaint', inputs={'message': cases[cid]['message']})\n"
        "    return harness.run(task, max_steps=16, budget_tracker=BudgetTracker(Budget(tool_calls=20)))"
    ),
    new_markdown_cell(
        "## The gate stack runs on every call\n"
        "\n"
        "Three gates guard each tool call: a syntax check that the call is well-formed, "
        "the banking policy engine (PII by regex, prompt-injection and prohibited-advice "
        "by a Qwen guard), and a trained GMS gate that scores the workflow transition "
        "against the policy graph. Only the last is a trained model; the rest is "
        "assembled scaffolding."
    ),
    new_code_cell(
        "gates = harness._executor._gates\n"
        "for g in gates:\n"
        "    print(f'{g.__class__.__name__:22s} trained={isinstance(g, GMSPlausibilityGate)}')"
    ),
    new_markdown_cell(
        "## A call refused at the gate\n"
        "\n"
        "The message in `case-011` carries a Social Security number. The PII policy denies "
        "the very first tool call: the tool body never runs, the failed result names the "
        "gate that refused it, and the agent escalates rather than proceeding. Nothing "
        "unsafe reached the model or the log."
    ),
    new_code_cell(
        "traj = run('case-011')\n"
        "print('message:', cases['case-011']['message'])\n"
        "denied = next((r for r in traj.records if r.action.kind == 'tool_call'\n"
        "               and r.observation and not r.observation.get('success')), None)\n"
        "if denied:\n"
        "    print('refused by gate:', denied.observation['error'])\n"
        "esc = next((r for r in traj.records if r.action.kind == 'escalate'), None)\n"
        "print('outcome        :', traj.final_state.status, '->', esc.action.reason if esc else '')"
    ),
    new_markdown_cell(
        "## An escalation on regulatory risk\n"
        "\n"
        "The \"unfair fee\" message in `case-016` is well-formed and clears the gates, but "
        "`flag_regulatory` marks a UDAAP risk. The agent does not draft a reply that "
        "grants a remedy it cannot authorize; it escalates, naming the flag."
    ),
    new_code_cell(
        "traj = run('case-016')\n"
        "print('message:', cases['case-016']['message'])\n"
        "esc = next((r for r in traj.records if r.action.kind == 'escalate'), None)\n"
        "print('outcome:', traj.final_state.status, '->', esc.action.reason if esc else '(drafted)')"
    ),
    new_markdown_cell(
        "## Every decision is in the audit chain\n"
        "\n"
        "The harness logs one event per step and links them by hash, so the sequence of "
        "decisions can be replayed and verified. A refusal or an escalation is an "
        "auditable event, not a silent drop."
    ),
    new_code_cell(
        "print('audit events     :', len(harness.audit.events))\n"
        "print('audit chain valid:', harness.audit.verify())"
    ),
    new_markdown_cell(
        "The audience sees the agent stopped twice --- once at a gate, once at a flag --- "
        "and both stops are recorded. That is the difference between an agent that is "
        "smart and one that is bounded."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
