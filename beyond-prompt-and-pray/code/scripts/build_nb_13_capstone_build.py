#!/usr/bin/env python
"""Builder for notebooks/13_capstone_build.ipynb.

Capstone BUILD series, Chapter 13 (Escalation). The complaint agent escalates on three
paths: a prior tool failure (a denied gate), a regulatory flag it must not auto-resolve
(UDAAP / Reg X), and an ungrounded claim before drafting. This notebook drives an
adversarial case down each path and shows the human-review interface the escalation
hands off to. Runs the real harness (Qwen models + GMS store).

Teaching notebook: real imports, no pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "13_capstone_build.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone build --- Chapter 13: Escalation\n"
        "\n"
        "A governed agent must know when not to act. The complaint agent escalates to a "
        "human on three paths: when a prior tool call was denied by a gate, when the "
        "regulatory check flags a risk it must not auto-resolve, and when a claim would "
        "reach the draft without evidence. Chapter~13 drives cases down these paths and "
        "shows the human-review interface the escalation hands off to. It runs the real "
        "harness."
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
        "harness, registry = build_complaint_harness(policies_dir=root / 'data' / 'policies')\n"
        "\n"
        "def run(case_id):\n"
        "    case = next(c for c in cases if c['id'] == case_id)\n"
        "    task = TaskSpec(goal='handle complaint', inputs={'message': case['message']})\n"
        "    traj = harness.run(task, max_steps=16, budget_tracker=BudgetTracker(Budget(tool_calls=20)))\n"
        "    return case, traj"
    ),
    new_markdown_cell(
        "## Escalation on regulatory risk\n"
        "\n"
        "When `flag_regulatory` marks a UDAAP or Reg X risk, the agent does not draft a "
        "reply: it raises an `Escalate` naming the flags. The escalation is a step in "
        "the trajectory, so its reason is recorded and auditable."
    ),
    new_code_cell(
        "case, traj = run('case-016')\n"
        "print('message:', case['message'])\n"
        "print('status :', traj.final_state.status)\n"
        "esc = next((r for r in traj.records if r.action.kind == 'escalate'), None)\n"
        "print('reason :', esc.action.reason if esc else '(no escalation)')"
    ),
    new_markdown_cell(
        "## Escalation on a denied tool call\n"
        "\n"
        "A message carrying PII is denied at the policy gate, and the agent reads that "
        "failed result and escalates rather than proceeding. This is the same failed-"
        "result-to-escalation path the plan of Chapter~8 diverted on, now driven by a "
        "real gate refusal."
    ),
    new_code_cell(
        "case, traj = run('case-011')\n"
        "print('message:', case['message'])\n"
        "print('status :', traj.final_state.status)\n"
        "failed = next((r for r in traj.records if r.action.kind == 'tool_call'\n"
        "               and r.observation and not r.observation.get('success')), None)\n"
        "if failed:\n"
        "    print('denied :', failed.observation['error'])\n"
        "esc = next((r for r in traj.records if r.action.kind == 'escalate'), None)\n"
        "print('reason :', esc.action.reason if esc else '(no escalation)')"
    ),
    new_markdown_cell(
        "## The human-review interface\n"
        "\n"
        "An escalation is a hand-off, not a dead end. `harness.run` accepts a "
        "`human_reviewer`; an `EscalationRequest` carries the run id, the step, the "
        "reason and the gate results to the reviewer, who returns a decision. "
        "`ScriptedReviewer` supplies fixed decisions, which is what a test uses in place "
        "of a person at the console."
    ),
    new_code_cell(
        "from agentlab.governance.escalation import (\n"
        "    ScriptedReviewer, HumanResponse, HumanDecision,\n"
        ")\n"
        "reviewer = ScriptedReviewer([HumanResponse(decision=HumanDecision.DEFER, note='needs analyst review')])\n"
        "case, _ = run('case-016')\n"
        "task = TaskSpec(goal='handle complaint', inputs={'message': case['message']})\n"
        "traj = harness.run(task, max_steps=16,\n"
        "                   budget_tracker=BudgetTracker(Budget(tool_calls=20)),\n"
        "                   human_reviewer=reviewer)\n"
        "print('final status with reviewer:', traj.final_state.status)"
    ),
    new_markdown_cell(
        "Escalation is how the agent stays inside its authority: a denied call, a flagged "
        "regulation or an ungrounded claim all route to a human with the context of the "
        "decision, recorded in the audit chain. Chapter~14 generalizes the single agent "
        "into a supervised set of workers, and Chapter~15 assembles the whole capstone."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
