#!/usr/bin/env python
"""Builder for notebooks/03_capstone_build.ipynb.

Capstone BUILD series, Chapter 3 (Reasoning Traces). Adds a Scratchpad to the loop:
each step's model output enters as a typed entry with a trust level and an evidence
pointer, and a claim without evidence fails a structural check. This is the record the
capstone's ComplaintAgent reconstructs as a pure function of its trajectory.

Teaching notebook: real imports, no pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "03_capstone_build.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone build --- Chapter 3: Reasoning Traces\n"
        "\n"
        "The loop in Chapter~1 recorded what the agent did --- the action and the "
        "observation at each step. Chapter~3 adds a record of what the agent may "
        "conclude: a reasoning trace in which every entry is typed, carries a trust "
        "level and points at its evidence. The capstone builds this trace with the "
        "`Scratchpad`, and enforces one rule on it before any customer-facing output "
        "leaves the system --- every claim must carry evidence."
    ),
    new_markdown_cell(
        "## A typed reasoning entry\n"
        "\n"
        "The `Scratchpad` is an append-only collection of typed entries. An observation "
        "records something a tool reported and names its source; a claim asserts "
        "something the reply may depend on and must carry evidence; an assumption "
        "records something inferred without textual support, which is precisely what "
        "must not be allowed to pass for a fact. Each entry carries a trust level."
    ),
    new_code_cell(
        "from agentlab.reasoning import Scratchpad, TrustLevel\n"
        "\n"
        "pad = Scratchpad()\n"
        "pad.add_observation(\n"
        "    'message classified as complaint',\n"
        "    source='classify_complaint',\n"
        "    evidence='classifier confidence=1.00',\n"
        "    trust=TrustLevel.MEDIUM,\n"
        ")\n"
        "pad.add_claim(\n"
        "    'issue is unauthorized_fee',\n"
        "    evidence='extract_facts grounded to policy entity [overdraft_fee]',\n"
        "    trust=TrustLevel.HIGH,\n"
        ")\n"
        "pad.add_assumption('urgency high, sentiment negative')\n"
        "print(pad.render_table(as_string=True))"
    ),
    new_markdown_cell(
        "The trace separates what was observed, what is claimed and what was merely "
        "assumed. The assumption about urgency and sentiment is recorded honestly as an "
        "assumption: it was inferred without a citation, so it can never pass a check "
        "for evidence it has not earned."
    ),
    new_markdown_cell(
        "## The evidence check\n"
        "\n"
        "The discipline the capstone enforces is that no claim reaches an output step "
        "without evidence. `assert_all_claims_have_evidence` is the structural check the "
        "harness runs; a claim added without an evidence pointer makes it raise, and the "
        "agent escalates to a human rather than letting an ungrounded statement through."
    ),
    new_code_cell(
        "pad.assert_all_claims_have_evidence()   # passes: the only claim has evidence\n"
        "print('all claims grounded:', not pad.unsupported_claims())\n"
        "\n"
        "pad.add_claim('the fee was a bank error')   # no evidence pointer\n"
        "try:\n"
        "    pad.assert_all_claims_have_evidence()\n"
        "except AssertionError as exc:\n"
        "    print('blocked before output:', exc)\n"
        "    print('unsupported:', [e.text for e in pad.unsupported_claims()])"
    ),
    new_markdown_cell(
        "In the capstone this trace is not written by hand. The `ComplaintAgent` "
        "reconstructs the scratchpad from its tool results as a pure function of the "
        "trajectory: the same trajectory always yields the same trace, so it replays "
        "from the audit log without rerunning the agent. Each tool output enters as a "
        "typed entry --- the classifier result as an observation, a grounded issue as a "
        "high-trust claim, an implicated regulation as a highest-trust claim checked "
        "against the graph --- and the evidence check runs before the draft step. "
        "Chapter~4 formalizes the task, state and action types this trace hangs on; the "
        "governance gates of Chapter~6 and the escalation path of Chapter~13 are what "
        "act on a failed check."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
