#!/usr/bin/env python
"""Builder for notebooks/10_capstone_companion.ipynb.

Capstone companion to Chapter 10 (Trajectory Evaluation and Metrics): the concept read
on the running banking complaint agent. Teaching notebook in the style of the main
chapter notebooks -- real capstone imports and the pinned campaign artifact, no
pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "10_capstone_companion.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone companion --- Chapter 10: Trajectory Evaluation and Metrics\n"
        "\n"
        "Chapter~10 argues that a single scalar cannot describe an agent run. A run is "
        "scored at three levels: the *decision* it reached, the *structure* of the "
        "trajectory that produced it, and the *process health* of that trajectory; and a "
        "claim's *groundedness* is measured as a distance from its supporting evidence. "
        "This companion reads those levels on the capstone banking complaint agent, using "
        "the evaluation functions in `agentlab.evaluation` and the pinned campaign "
        "artifact `data/capstone_run.json`."
    ),
    new_markdown_cell(
        "The metric layer is a set of pure functions over a `Trajectory`. Each is "
        "independent and composable, so a report is assembled by applying them rather "
        "than by threading a single accumulator through the run. Reading the module's "
        "exports names the vocabulary the chapter defines."
    ),
    new_code_cell(
        "import agentlab.evaluation as ev\n"
        "\n"
        "level_metrics = ['task_success', 'escalated']            # decision level\n"
        "structure_metrics = ['step_count', 'tool_call_count']    # trajectory structure\n"
        "health_metrics = ['tool_failure_count', 'finished_cleanly']  # process health\n"
        "for group, names in [('decision', level_metrics),\n"
        "                     ('structure', structure_metrics),\n"
        "                     ('process health', health_metrics)]:\n"
        "    present = [n for n in names if hasattr(ev, n)]\n"
        "    print(f'{group:16s}: {present}')"
    ),
    new_markdown_cell(
        "## The pinned campaign artifact\n"
        "\n"
        "A single run yields one trajectory; a *campaign* aggregates many runs into the "
        "metrics the chapter reports. The capstone's campaign is pinned to "
        "`data/capstone_run.json` so its numbers are stable across readings. The top-level "
        "keys separate the decision-level score from the structural and per-component "
        "diagnostics."
    ),
    new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "root = Path('.') if Path('data').exists() else Path('..')\n"
        "run = json.loads((root / 'data' / 'capstone_run.json').read_text())\n"
        "print('n_runs           :', run['n_runs'])\n"
        "print('top-level keys   :', list(run.keys()))"
    ),
    new_markdown_cell(
        "## The decision level\n"
        "\n"
        "The decision-level score is the fraction of test queries decisioned correctly: "
        "the right classification together with the right escalate / do-not-escalate "
        "decision. It is deliberately narrow. The artifact records its definition alongside "
        "the number so the two are never separated."
    ),
    new_code_cell(
        "overall = run['overall']\n"
        "print('decision accuracy:', overall['accuracy'])\n"
        "print('definition       :', overall['definition'])"
    ),
    new_markdown_cell(
        "## The trajectory-structure level\n"
        "\n"
        "The chapter distinguishes reaching the right decision from following the intended "
        "path to it. `workflow_adherence` measures whether the trajectory visited the "
        "tools in the mandated order, and `audit_verifies` records whether the run's audit "
        "trail reconstructs. A run may decide correctly on a malformed trajectory, so these "
        "are reported apart from accuracy rather than folded into it."
    ),
    new_code_cell(
        "print('workflow_adherence:', run['workflow_adherence'])\n"
        "print('audit_verifies    :', run['audit_verifies'])\n"
        "print('weak_link         :', run['weak_link'])"
    ),
    new_markdown_cell(
        "## Per-component attribution\n"
        "\n"
        "A decision accuracy of $0.675$ is a property of the whole workflow. Attribution "
        "asks which tool the errors concentrate in. Each per-tool entry reports how many "
        "outputs were scored, how many were correct, and the resulting accuracy, so the "
        "aggregate is decomposed into the components that produced it."
    ),
    new_code_cell(
        "for name, stats in run['per_tool'].items():\n"
        "    acc = stats['accuracy']\n"
        "    print(f'{name:20s} accuracy={acc:.3f}  '\n"
        "          f'({stats[\"correct\"]}/{stats[\"scored\"]} scored)')"
    ),
    new_markdown_cell(
        "The `weak_link` count locates the errors by component, so a low aggregate is "
        "traced to the tool that produced most of the failed decisions rather than "
        "attributed to the agent as a whole."
    ),
    new_code_cell(
        "weak = run['weak_link']\n"
        "for tool, failures in sorted(weak.items(), key=lambda kv: -kv[1]):\n"
        "    print(f'{tool:20s} failures={failures}')"
    ),
    new_markdown_cell(
        "## Groundedness as a distance\n"
        "\n"
        "Whether a drafted claim is *supported by* the retrieved evidence is a distinct "
        "question from whether the decision was correct. Chapter~10 treats groundedness as "
        "a distance between a claim and its nearest evidence: a claim close enough to some "
        "evidence is `SUPPORTED`, and one too far from all of it is `UNSUPPORTED`. "
        "`check_groundedness` returns that verdict together with the evidence it matched."
    ),
    new_code_cell(
        "from agentlab.evaluation import Claim, check_groundedness, groundedness_report\n"
        "\n"
        "evidence = {\n"
        "    'reg-e': ('Regulation E limits consumer liability for unauthorized electronic '\n"
        "              'fund transfers and requires timely error resolution.'),\n"
        "    'overdraft': ('Overdraft fees may be refunded when the charge was not '\n"
        "                  'authorized by the accountholder.'),\n"
        "}\n"
        "grounded = Claim(text='Regulation E limits liability for unauthorized electronic fund transfers.')\n"
        "fabricated = Claim(text='The bank guarantees a refund within twenty four hours for any dispute.')\n"
        "for r in groundedness_report([grounded, fabricated], evidence):\n"
        "    print(f'{r.verdict.value:12s} evidence={str(r.evidence_id):10s} | {r.claim}')"
    ),
    new_markdown_cell(
        "The supported claim resolves to the evidence it paraphrases; the fabricated "
        "guarantee matches nothing and is `UNSUPPORTED` with no evidence identifier. This "
        "is the distance made operational: the verdict is a function of how near the claim "
        "sits to the closest supporting passage, not of how fluent the draft reads."
    ),
    new_markdown_cell(
        "## Reading the levels together\n"
        "\n"
        "This is the capstone's realization of Chapter~10. The three levels answer "
        "different questions and do not substitute for one another: decision accuracy is "
        "$0.675$, workflow adherence is $1.0$, and the errors concentrate in "
        "`classify_complaint`, while groundedness is scored per claim as a distance from "
        "evidence. Chapter~15 assembles the five tools into the governed workflow these "
        "metrics score, and Chapter~16 turns the campaign into a designed test suite that "
        "attributes the aggregate to the factors that move it."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
