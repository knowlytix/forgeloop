#!/usr/bin/env python
"""Builder for notebooks/07_capstone_companion.ipynb.

Capstone companion to Chapter 7 (Cost, Latency and Budgets): the concept read on
the running banking complaint agent. A run is bounded by an explicit Budget and a
BudgetTracker that accumulates consumption and reports when any axis is exhausted.
Teaching notebook in the style of the main chapter notebooks -- real capstone
imports, cheap deterministic demos, no pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "07_capstone_companion.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone companion --- Chapter 7: Cost, Latency and Budgets\n"
        "\n"
        "An agent that reasons in a loop will, if left unbounded, consume tokens, "
        "wall-clock time, tool calls and dollars without limit; a run that never "
        "terminates is a failure of resource discipline before it is a failure of "
        "correctness. Chapter~7 addresses this by giving every run an explicit "
        "budget and a tracker that accumulates consumption against it, so the loop "
        "terminates when any axis is exhausted. This companion reads that principle "
        "on the capstone banking complaint agent."
    ),
    new_markdown_cell(
        "The capstone represents a budget as two objects in `agentlab.core`. A "
        "`Budget` declares per-axis upper bounds; a `BudgetTracker` accumulates a "
        "`Consumption` against one budget and reports whether any axis is spent. The "
        "run loop consults the tracker before each step, so the concern is enforced "
        "at the loop boundary rather than inside any single tool."
    ),
    new_code_cell(
        "from agentlab.core import Budget, BudgetTracker, Consumption\n"
        "import inspect\n"
        "\n"
        "print('Budget fields      :', list(Budget.__dataclass_fields__))\n"
        "print('Consumption fields :', list(Consumption.__dataclass_fields__))\n"
        "print('BudgetTracker API  :',\n"
        "      [m for m in dir(BudgetTracker) if not m.startswith('_')])"
    ),
    new_markdown_cell(
        "## Declaring a budget for a complaint run\n"
        "\n"
        "A `Budget` caps four axes: `tokens`, `seconds`, `tool_calls` and `dollars`. "
        "A value of `None` leaves an axis unbounded. A single complaint passes "
        "through five tools --- classify, extract, search, flag and draft --- so a "
        "healthy run makes on the order of five tool calls. The budget below sets a "
        "tool-call ceiling that admits that path with a small margin, together with "
        "token and dollar ceilings sized for one short response."
    ),
    new_code_cell(
        "budget = Budget(tokens=4000, seconds=30.0, tool_calls=6, dollars=0.05)\n"
        "budget"
    ),
    new_markdown_cell(
        "## Tracking consumption across the run\n"
        "\n"
        "A `BudgetTracker` is constructed against one budget and starts a wall-clock "
        "timer. As the run proceeds the tracker records what each step consumes: "
        "`record_tool_call` on every tool invocation, `record_tokens` and "
        "`record_dollars` on every model call. The simulation below records the five "
        "tool calls of a nominal complaint run together with their token and dollar "
        "cost, then reads back the accumulated consumption."
    ),
    new_code_cell(
        "tracker = BudgetTracker(budget)\n"
        "\n"
        "# A nominal five-tool complaint run: (tool name, tokens, dollars).\n"
        "steps = [\n"
        "    ('classify_complaint', 320, 0.0032),\n"
        "    ('extract_facts',      540, 0.0054),\n"
        "    ('search_policy',      410, 0.0041),\n"
        "    ('flag_regulatory',    280, 0.0028),\n"
        "    ('draft_response',     900, 0.0090),\n"
        "]\n"
        "for name, toks, cost in steps:\n"
        "    tracker.record_tool_call()\n"
        "    tracker.record_tokens(toks)\n"
        "    tracker.record_dollars(cost)\n"
        "\n"
        "c = tracker.consumption()\n"
        "print(f'tool calls : {c.tool_calls}/{budget.tool_calls}')\n"
        "print(f'tokens     : {c.tokens}/{budget.tokens}')\n"
        "print(f'dollars    : {c.dollars:.4f}/{budget.dollars:.4f}')\n"
        "print(f'elapsed    : {c.seconds:.4f}s (budget {budget.seconds}s)')\n"
        "print('exhausted  :', tracker.exhausted())"
    ),
    new_markdown_cell(
        "The nominal run stays inside every axis, so `exhausted` is false and the "
        "loop would continue to a normal `Finish`. The tracker reports elapsed "
        "seconds from its own timer rather than from a recorded value, because "
        "wall-clock time accrues whether or not a step chooses to spend it."
    ),
    new_markdown_cell(
        "## Reaching the ceiling\n"
        "\n"
        "The purpose of the tracker is to name the axis that terminates a run. A "
        "complaint that loops --- re-searching policy after each partial answer --- "
        "spends tool calls without converging. `reason_exhausted` returns a "
        "human-readable string identifying the first axis at or above its bound, or "
        "`None` while the budget holds. The loop below records tool calls until the "
        "tracker reports exhaustion."
    ),
    new_code_cell(
        "looping = BudgetTracker(Budget(tool_calls=6))\n"
        "for i in range(1, 11):\n"
        "    looping.record_tool_call()\n"
        "    reason = looping.reason_exhausted()\n"
        "    print(f'after call {i:2d}: exhausted={looping.exhausted()!s:5s}  reason={reason}')\n"
        "    if reason is not None:\n"
        "        break"
    ),
    new_markdown_cell(
        "## What the run loop does at the ceiling\n"
        "\n"
        "The capstone run loop in `agentlab.core.loop` consults the tracker at the "
        "top of every step. When `exhausted` becomes true it does not silently stop: "
        "it synthesizes a terminal `Escalate` action whose reason is the string from "
        "`reason_exhausted` and whose context marks the source as the budget, then "
        "sets the run status to `failed`. A run that overspends therefore ends in a "
        "recorded, attributable escalation rather than an open-ended consumption of "
        "resources."
    ),
    new_code_cell(
        "from agentlab.core import Escalate\n"
        "\n"
        "# Reconstruct the terminal record the loop emits when the budget is spent.\n"
        "reason = looping.reason_exhausted()\n"
        "terminal = Escalate(reason=reason, context={'source': 'budget'})\n"
        "print('action  :', terminal.kind)\n"
        "print('reason  :', terminal.reason)\n"
        "print('source  :', terminal.context['source'])"
    ),
    new_markdown_cell(
        "This is the capstone's realization of Chapter~7: a complaint run carries an "
        "explicit `Budget`, a `BudgetTracker` accumulates consumption across its "
        "tools, and the loop terminates the run with an attributable budget "
        "escalation the moment any axis is exhausted. Chapter~15 assembles the five "
        "tools into the governed workflow, where the budget tracker bounds the same "
        "loop that the earlier chapters typed, guarded and sequenced."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
