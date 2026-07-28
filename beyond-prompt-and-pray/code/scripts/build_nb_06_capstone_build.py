#!/usr/bin/env python
"""Builder for notebooks/06_capstone_build.ipynb.

Capstone BUILD series, Chapter 6 (Safe Tool Execution). Replaces the bare environment
of Chapter 1 with a GovernedToolExecutor whose gate stack runs before any tool body.
A SyntaxGate checks the call is well-formed; a PolicyEngine gate runs the banking
policies. A call carrying PII is denied at the boundary and never executes.

Uses the deterministic pii_policy (regex) so the notebook stays fast and offline; the
shipped harness adds the Qwen-backed semantic policies and the GMS gate (Chapter 12).
Teaching notebook: real imports, no pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "06_capstone_build.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone build --- Chapter 6: Safe Tool Execution\n"
        "\n"
        "In Chapter~1 the environment ran a tool as soon as the arguments validated "
        "against its schema. A regulated agent cannot execute every well-formed call: a "
        "message carrying a Social Security number must not be forwarded to a model, and "
        "a prompt-injection attempt must not reach a tool at all. Chapter~6 inserts a "
        "gate stack between the proposal and the tool body. The `GovernedToolExecutor` "
        "runs each gate in turn and refuses the call if any gate denies it, so an unsafe "
        "action is stopped before it acts rather than after."
    ),
    new_markdown_cell(
        "## The action space, as before\n"
        "\n"
        "The registry is the five-tool action space from Chapter~5. Here it is built "
        "with the classify tool alone, which is enough to show the gate stack refusing "
        "a call; the full registry behaves identically because the gates act on the "
        "proposed action, not on which tool it names."
    ),
    new_code_cell(
        "from agentlab.tools import ToolRegistry\n"
        "from agentlab.capstone.banking_tools import classify_complaint\n"
        "\n"
        "registry = ToolRegistry()\n"
        "registry.register(classify_complaint)"
    ),
    new_markdown_cell(
        "## The gate stack\n"
        "\n"
        "The executor is configured with an ordered list of gates. The `SyntaxGate` "
        "rejects a call that does not match the tool's schema. The policy gate is built "
        "from a `PolicyEngine`, which aggregates the banking policies; the deterministic "
        "PII policy is used here, a regex format check that is the right tool for "
        "detecting an SSN, card number or email. The shipped harness adds two "
        "Qwen-backed semantic policies (prompt injection and prohibited advice) and the "
        "trained GMS plausibility gate of Chapter~12 to this same list."
    ),
    new_code_cell(
        "from agentlab.tools import GovernedToolExecutor\n"
        "from agentlab.governance import PolicyEngine, SyntaxGate, pii_policy\n"
        "\n"
        "engine = PolicyEngine([pii_policy])\n"
        "gates = [SyntaxGate(), engine.as_gate()]\n"
        "executor = GovernedToolExecutor(registry, gates=gates)\n"
        "print('gate stack:', [g.__class__.__name__ for g in gates])"
    ),
    new_markdown_cell(
        "## A safe call runs; an unsafe call is denied\n"
        "\n"
        "The executor returns a `ToolResult` either way. A well-formed, policy-clean call "
        "runs the tool and reports `success=True`. A call whose message carries an SSN is "
        "denied by the policy gate: the result is `success=False`, its error names the "
        "gate that refused it, and the tool body never ran."
    ),
    new_code_cell(
        "from agentlab.core.action import ToolCall\n"
        "\n"
        "clean = executor.execute(ToolCall(\n"
        "    tool_name='classify_complaint',\n"
        "    arguments={'message': 'I was charged a $35 overdraft fee I did not authorize.'},\n"
        "))\n"
        "print('clean call : success=', clean.success, '| output=', clean.output)\n"
        "\n"
        "unsafe = executor.execute(ToolCall(\n"
        "    tool_name='classify_complaint',\n"
        "    arguments={'message': 'My SSN is 123-45-6789 and I want a refund.'},\n"
        "))\n"
        "print('unsafe call: success=', unsafe.success, '| error=', unsafe.error)\n"
        "for r in unsafe.gate_results:\n"
        "    print('   gate', r.gate_name, '->', r.decision)"
    ),
    new_markdown_cell(
        "This governed executor replaces the bare environment of Chapter~1. Every tool "
        "call the agent proposes now passes the gate stack first, and a denied call "
        "returns a failed result the agent can act on --- in the capstone, a failed "
        "result at any step drives an escalation (Chapter~13). Chapter~7 bounds the loop "
        "in cost and latency, and Chapter~12 completes the gate stack with the trained "
        "GMS plausibility gate and wraps the executor in the governance harness."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
