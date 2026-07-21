#!/usr/bin/env python
"""Builder for notebooks/demos/demo_1_same_task_two_systems.ipynb.

Teaching demo 1 (deck opener): same task, two systems. The three cases run through
(a) the existing ungoverned baseline -- the classifier + the DenseRagRetriever "chunk
and pray" retriever, which always drafts an answer -- and (b) the governed harness
(build_complaint_harness). Both are existing, runnable code; this notebook only
curates the calls and is executed so the deck shows real output.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "demos" / "demo_1_same_task_two_systems.ipynb"

cells = [
    new_markdown_cell(
        "# Demo 1 --- Same task, two systems\n"
        "\n"
        "Three customer messages, two systems handling them. The default is the ordinary "
        "pipeline: classify the message, retrieve nearby policy text, let a model draft a "
        "reply. The governed system is the capstone harness: a fixed tool workflow behind "
        "a gate stack, with an escalation path and an audit log. Both are run below on the "
        "same three cases; the output is real."
    ),
    new_code_cell(
        "import json, warnings\n"
        "from pathlib import Path\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "root = next((c for c in (Path('.'), Path('..'), Path('../..'), Path('../../code'), Path('../code'))\n"
        "             if (c / 'data' / 'eval_cases' / 'cases.json').exists()), Path('.'))\n"
        "cases = {c['id']: c for c in json.loads((root / 'data' / 'eval_cases' / 'cases.json').read_text())}\n"
        "demo_ids = ['case-001', 'case-016', 'case-011']\n"
        "for cid in demo_ids:\n"
        "    print(cid, '->', cases[cid]['message'])"
    ),
    new_markdown_cell(
        "## The default: it always drafts\n"
        "\n"
        "The baseline is the classifier plus the `DenseRagRetriever` --- textbook chunk-"
        "and-pray retrieval that embeds the policy document, pulls the nearest chunks and "
        "lets the model answer from them. It has no gate, no grounding check and no "
        "abstention: `decision` is always `answer` and `verified` is always `False`. Every "
        "case gets a confident reply, including the one carrying an SSN."
    ),
    new_code_cell(
        "from agentlab.models.complaint_classifier import get_default_classifier\n"
        "from agentlab.capstone.dense_rag import DenseRagRetriever\n"
        "\n"
        "clf = get_default_classifier()\n"
        "dense = DenseRagRetriever()   # existing 'chunk and pray' baseline\n"
        "\n"
        "default_out = {}\n"
        "for cid in demo_ids:\n"
        "    msg = cases[cid]['message']\n"
        "    label, conf = clf.classify(msg)\n"
        "    hit = dense.search(msg)[0]\n"
        "    default_out[cid] = {'class': label, 'draft': hit['answer'], 'verified': hit['verified']}\n"
        "    print(f\"[{cid}] class={label} verified={hit['verified']}\")\n"
        "    print('   draft:', hit['answer'][:200])\n"
        "    print()"
    ),
    new_markdown_cell(
        "## The governed system: it grounds, escalates or refuses\n"
        "\n"
        "The same three messages through `build_complaint_harness`. The routine fee case "
        "runs the full workflow to a grounded draft; the \"unfair\" case raises a UDAAP "
        "flag and escalates instead of drafting; the SSN case is refused at the PII gate "
        "on the first tool call and escalates. Each outcome is a decision in the audit "
        "chain, not a silent draft."
    ),
    new_code_cell(
        "from agentlab.capstone import build_complaint_harness\n"
        "from agentlab.core import Budget, BudgetTracker, TaskSpec\n"
        "\n"
        "harness, registry = build_complaint_harness(policies_dir=root / 'data' / 'policies')\n"
        "\n"
        "gov_out = {}\n"
        "for cid in demo_ids:\n"
        "    task = TaskSpec(goal='handle complaint', inputs={'message': cases[cid]['message']})\n"
        "    traj = harness.run(task, max_steps=16, budget_tracker=BudgetTracker(Budget(tool_calls=20)))\n"
        "    out = traj.final_state.final_output or {}\n"
        "    esc = next((r for r in traj.records if r.action.kind == 'escalate'), None)\n"
        "    gov_out[cid] = {'status': traj.final_state.status,\n"
        "                    'action': out.get('recommended_action'),\n"
        "                    'reason': esc.action.reason if esc else '',\n"
        "                    'draft': (out.get('draft_response') or '')}\n"
        "    print(f\"[{cid}] status={traj.final_state.status} action={out.get('recommended_action')}\")\n"
        "    if esc:\n"
        "        print('   escalation:', esc.action.reason)\n"
        "    elif out.get('draft_response'):\n"
        "        print('   grounded draft:', out['draft_response'][:200])\n"
        "    print()"
    ),
    new_markdown_cell(
        "## Side by side\n"
        "\n"
        "The default drafts confidently on all three, unverified. The governed system "
        "grounds the one it should answer and stops on the two it should not."
    ),
    new_code_cell(
        "print(f\"{'case':10s} {'default':28s} {'governed':28s}\")\n"
        "print('-' * 68)\n"
        "for cid in demo_ids:\n"
        "    d = f\"drafts (verified={default_out[cid]['verified']})\"\n"
        "    g = gov_out[cid]['status']\n"
        "    if gov_out[cid]['reason']:\n"
        "        g += ': ' + gov_out[cid]['reason'][:40]\n"
        "    print(f'{cid:10s} {d:28s} {g:28s}')"
    ),
    new_markdown_cell(
        "The task is identical. The difference is that every step of the governed run is "
        "bounded, grounded and checked, so a case it cannot safely answer becomes an "
        "escalation a human sees rather than a confident reply no one flagged."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
