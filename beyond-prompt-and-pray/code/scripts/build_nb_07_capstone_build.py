#!/usr/bin/env python
"""Builder for notebooks/07_capstone_build.ipynb.

Capstone BUILD series, Chapter 7 (Cost and Latency Budgets). Adds a BudgetTracker to
the loop: per-axis upper bounds on tokens, seconds and tool calls. When an axis is
exhausted the loop stops and escalates rather than running unbounded. The capstone
passes BudgetTracker(Budget(tool_calls=20)) on every case.

Budget mechanics are shown directly (no model load needed). Teaching notebook: real
imports, no pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "07_capstone_build.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone build --- Chapter 7: Cost and Latency Budgets\n"
        "\n"
        "A loop that runs until the agent chooses to finish can run without bound if the "
        "agent never does. Chapter~7 puts a ceiling on the loop. A `Budget` declares "
        "per-axis upper bounds --- tokens, seconds, tool calls, dollars --- and a "
        "`BudgetTracker` accumulates consumption against them. The loop checks the "
        "tracker before each step and stops with an escalation once any axis is "
        "exhausted, so a runaway or wedged case ends in a bounded, auditable way."
    ),
    new_markdown_cell(
        "## Declaring a budget\n"
        "\n"
        "Each axis of a `Budget` is an upper bound, and `None` means unlimited on that "
        "axis. The complaint agent bounds the number of tool calls, which caps a case at "
        "its fixed five-step workflow plus a margin; it leaves tokens and dollars "
        "unbounded because the local models carry no per-call charge."
    ),
    new_code_cell(
        "from agentlab.core import Budget, BudgetTracker\n"
        "\n"
        "budget = Budget(tool_calls=20)\n"
        "tracker = BudgetTracker(budget)\n"
        "print('bounds     :', budget)\n"
        "print('exhausted? :', tracker.exhausted())"
    ),
    new_markdown_cell(
        "## Accumulating against the bound\n"
        "\n"
        "The tracker records consumption as the loop runs. Recording tool calls up to "
        "the bound leaves the tracker exhausted, and it reports which axis ran out. This "
        "is the check `run_loop` performs before each step."
    ),
    new_code_cell(
        "small = BudgetTracker(Budget(tool_calls=3))\n"
        "for i in range(3):\n"
        "    small.record_tool_call()\n"
        "    print(f'after {i+1} calls: exhausted={small.exhausted()}')\n"
        "print('reason      :', small.reason_exhausted())\n"
        "print('consumption :', small.consumption())"
    ),
    new_markdown_cell(
        "## The loop stops on an exhausted budget\n"
        "\n"
        "`run_loop` accepts a `budget_tracker`. Before proposing an action it checks "
        "whether the budget is exhausted; if so it yields a final `Escalate` step with "
        "the exhaustion reason and stops. Starting the loop with an already-exhausted "
        "tracker shows the mechanism without needing to run a tool: the very first step "
        "is the budget escalation."
    ),
    new_code_cell(
        "from agentlab.core.agent import BaseAgent\n"
        "from agentlab.core.action import Finish\n"
        "from agentlab.core.state import AgentState\n"
        "from agentlab.core.task import TaskSpec\n"
        "from agentlab.core.loop import run_loop\n"
        "\n"
        "class NeverFinishes(BaseAgent):\n"
        "    def propose_action(self, state):\n"
        "        return Finish(output=None)  # never reached: the budget check fires first\n"
        "    def update(self, state, action, observation):\n"
        "        return state\n"
        "\n"
        "state = AgentState(task=TaskSpec(goal='demo', inputs={}))\n"
        "exhausted = BudgetTracker(Budget(tool_calls=0))  # nothing allowed\n"
        "for rec in run_loop(NeverFinishes(), None, state, max_steps=4, budget_tracker=exhausted):\n"
        "    print(f'step {rec.step}: {rec.action.kind} -> {rec.state_after.status}')\n"
        "    if rec.action.kind == 'escalate':\n"
        "        print('   reason:', rec.action.reason)"
    ),
    new_markdown_cell(
        "The budget makes termination a property of the harness, not a hope about the "
        "agent. In the capstone every case runs under `BudgetTracker(Budget(tool_calls="
        "20))`, so a case that fails to converge escalates on the budget axis and is "
        "logged like any other escalation. Chapter~8 gives the agent the fixed plan that "
        "keeps a well-behaved case well inside this bound."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
