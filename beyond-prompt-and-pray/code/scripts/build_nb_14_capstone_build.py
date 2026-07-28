#!/usr/bin/env python
"""Builder for notebooks/14_capstone_build.ipynb.

Capstone BUILD series, Chapter 14 (Multi-Agent). The capstone ships as a single
governed agent; multi-agent is the generalization for when one workflow is not enough.
This notebook wraps the complaint harness as a Worker under a Supervisor and delegates
a case over a message bus, showing the same governed agent as one node in a larger
system. Runs the real harness (Qwen models + GMS store).

Teaching notebook: real imports, no pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "14_capstone_build.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone build --- Chapter 14: Multi-Agent\n"
        "\n"
        "The complaint agent is a single governed workflow, and for the capstone that is "
        "enough. Chapter~14 is about what changes when one workflow is not: when a task "
        "must be decomposed and routed to specialists that report back. The structure is "
        "a supervisor that delegates to named workers over a message bus. The complaint "
        "harness built in the previous chapters becomes one such worker, unchanged --- "
        "the multi-agent layer wraps governed agents, it does not replace their "
        "governance."
    ),
    new_markdown_cell(
        "## A worker wraps the governed harness\n"
        "\n"
        "A `Worker` pairs a name and a capability with a `GovernanceHarness`. Wrapping "
        "the complaint harness in a worker exposes it as a delegable unit: the worker "
        "handles a message by running its harness on the delegated task, so the gates, "
        "the audit chain and the escalation paths all still apply."
    ),
    new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "from agentlab.capstone import build_complaint_harness\n"
        "from agentlab.multiagent.worker import Worker\n"
        "from agentlab.multiagent.supervisor import Supervisor\n"
        "\n"
        "root = next((c for c in (Path('.'), Path('..'), Path('../code'), Path('code'))\n"
        "             if (c / 'data' / 'eval_cases' / 'cases.json').exists()), Path('.'))\n"
        "cases = json.loads((root / 'data' / 'eval_cases' / 'cases.json').read_text())\n"
        "harness, registry = build_complaint_harness(policies_dir=root / 'data' / 'policies')\n"
        "\n"
        "complaint_worker = Worker(\n"
        "    name='complaint_handler',\n"
        "    capability='handle a banking complaint under policy and regulation',\n"
        "    harness=harness,\n"
        ")\n"
        "print('worker     :', complaint_worker.name)\n"
        "print('capability :', complaint_worker.capability)"
    ),
    new_markdown_cell(
        "## A supervisor delegates over a bus\n"
        "\n"
        "A `Supervisor` holds the workers and a `MessageBus`. `delegate` sends a typed "
        "delegation message to a named worker, which runs its governed harness and "
        "returns a response message. The bus records the exchange, so a multi-agent run "
        "is as inspectable as a single trajectory."
    ),
    new_code_cell(
        "supervisor = Supervisor(name='triage', workers=[complaint_worker])\n"
        "print('workers:', supervisor.workers())\n"
        "\n"
        "case = next(c for c in cases if c['id'] == 'case-002')\n"
        "response = supervisor.delegate(\n"
        "    worker_name='complaint_handler',\n"
        "    goal='handle complaint',\n"
        "    inputs={'message': case['message']},\n"
        "    max_steps=16,\n"
        ")\n"
        "print('response from :', response.sender)\n"
        "print('message type  :', response.message_type)\n"
        "print('payload keys  :', list(response.payload))"
    ),
    new_markdown_cell(
        "## The exchange is on the bus\n"
        "\n"
        "The delegation and its response are both messages on the supervisor's bus, so "
        "the routing is auditable in the same way the single agent's steps are. The "
        "thread below is the record of who asked what of whom."
    ),
    new_code_cell(
        "for msg in supervisor.bus.all():\n"
        "    print(f'{msg.sender:16s} -> {msg.receiver:16s} [{msg.message_type}]')"
    ),
    new_markdown_cell(
        "Multi-agent structure is a way to compose governed agents, not a way around "
        "their governance: the complaint worker runs the same harness, gates and audit "
        "chain whether it is called directly or delegated to. The capstone uses the "
        "single agent because its task is one fixed workflow; the supervisor here shows "
        "the seam at which it would become a specialist among several. Chapter~15 "
        "assembles the single-agent capstone in full."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
