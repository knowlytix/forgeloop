#!/usr/bin/env python
"""Builder for notebooks/10_capstone_build.ipynb.

Capstone BUILD series, Chapter 10 (Trajectory Evaluation and Metrics). Runs the built
harness on one case, turning the loop into a Trajectory, then reads it at several
levels: the decision (did it finish or escalate, correctly), the structure (how many
steps, any tool failures) and a summary. This is the first build notebook that runs the
whole governed agent, so it loads the real Qwen models and the GMS store.

Teaching notebook: real imports, no pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "10_capstone_build.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone build --- Chapter 10: Trajectory Evaluation and Metrics\n"
        "\n"
        "The workflow of Chapter~8 produces a sequence of steps; Chapter~10 reads that "
        "sequence as an object and scores it. A `Trajectory` is the task together with "
        "the recorded steps, and it is inspected at several levels: the decision the "
        "agent reached, the structure of how it got there, and a summary that rolls the "
        "structural metrics together. This is the first build notebook that runs the "
        "whole agent, so it loads the real models and the GMS store."
    ),
    new_markdown_cell(
        "## Build the harness and run one case\n"
        "\n"
        "`build_complaint_harness` returns the assembled governance harness and its "
        "registry; Chapter~12 dissects what it wires. Running it on a routine case "
        "returns a `Trajectory` whose records are the `(state, action, observation)` "
        "transitions of the loop."
    ),
    new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "from agentlab.capstone import build_complaint_harness\n"
        "from agentlab.core import Budget, BudgetTracker, TaskSpec\n"
        "\n"
        "root = next((c for c in (Path('.'), Path('..'), Path('../code'), Path('code'))\n"
        "             if (c / 'data' / 'eval_cases' / 'cases.json').exists()), Path('.'))\n"
        "policies_dir = root / 'data' / 'policies'\n"
        "cases = json.loads((root / 'data' / 'eval_cases' / 'cases.json').read_text())\n"
        "\n"
        "harness, registry = build_complaint_harness(policies_dir=policies_dir)\n"
        "case = next(c for c in cases if c['id'] == 'case-002')\n"
        "task = TaskSpec(goal='handle complaint', inputs={'message': case['message']})\n"
        "traj = harness.run(task, max_steps=16, budget_tracker=BudgetTracker(Budget(tool_calls=20)))\n"
        "print('message:', case['message'])\n"
        "print('steps  :', len(traj.records), '| final status:', traj.final_state.status)"
    ),
    new_markdown_cell(
        "## The decision level\n"
        "\n"
        "The coarsest reading asks what the agent decided and whether that matches what "
        "the case expected. For a routine complaint the expected outcome is a drafted "
        "response rather than an escalation."
    ),
    new_code_cell(
        "out = traj.final_state.final_output or {}\n"
        "print('classification    :', out.get('classification'))\n"
        "print('recommended action:', out.get('recommended_action'))\n"
        "print('expected escalate :', case.get('expected_escalation'))"
    ),
    new_markdown_cell(
        "## The structural level\n"
        "\n"
        "The `metrics` module reads the trajectory's shape: how many steps it took, how "
        "many tool calls it made, whether any tool failed, whether it finished cleanly "
        "or escalated. `summarize` rolls these into one record, which is what a suite "
        "aggregates over many cases in Chapter~16."
    ),
    new_code_cell(
        "from agentlab.evaluation import summarize\n"
        "from agentlab.evaluation.metrics import (\n"
        "    step_count, tool_call_count, tool_failure_count, escalated, finished_cleanly,\n"
        ")\n"
        "print('steps         :', step_count(traj))\n"
        "print('tool calls    :', tool_call_count(traj))\n"
        "print('tool failures :', tool_failure_count(traj))\n"
        "print('escalated     :', escalated(traj))\n"
        "print('finished clean:', finished_cleanly(traj))\n"
        "print('summary       :', summarize(traj))"
    ),
    new_markdown_cell(
        "## The per-step transitions\n"
        "\n"
        "The trajectory is inspectable step by step, which is what makes a decision "
        "auditable: each record names the action proposed and whether its observation "
        "reported success. Reading the records back is how a reviewer sees exactly what "
        "the agent did."
    ),
    new_code_cell(
        "for rec in traj.records:\n"
        "    kind = rec.action.kind\n"
        "    tool = getattr(rec.action, 'tool_name', '')\n"
        "    ok = rec.observation.get('success') if rec.observation else ''\n"
        "    print(f'step {rec.step}: {kind:10s} {tool:18s} success={ok}')"
    ),
    new_markdown_cell(
        "Reading a trajectory at these levels is what separates evaluation from a demo: "
        "the decision says whether the outcome was right, the structure says how the "
        "agent got there, and the records make each step auditable. Chapter~11 designs "
        "the suite of cases these metrics are aggregated over, and Chapter~16 runs it "
        "against the finished agent."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
