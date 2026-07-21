#!/usr/bin/env python
"""Builder for notebooks/06_capstone_companion.ipynb.

Capstone companion to Chapter 6 (Safe Tool Execution): the concept read on the
running banking complaint agent. Teaching notebook in the style of the main
chapter notebooks -- real capstone imports, no pre-embedded outputs (the reader
runs it). The live demos are deliberately cheap: they construct the executor and
its gates and feed them malformed or high-risk calls, so no GPU-backed tool body
runs.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "06_capstone_companion.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone companion --- Chapter 6: Safe Tool Execution\n"
        "\n"
        "Chapter~6 addresses the problem that a typed tool call, valid in form, may still "
        "be one that should not run: it may name a tool that does not exist, carry "
        "arguments the tool cannot accept, or invoke a high-risk action without "
        "authorization. The remedy is a governed executor that interposes a fixed sequence "
        "of gates between a proposed call and the tool body, so a bad call is caught before "
        "it acts. This companion reads that principle on the capstone banking complaint "
        "agent, whose tool calls all pass through a single `GovernedToolExecutor`."
    ),
    new_markdown_cell(
        "The executor holds a `ToolRegistry` and an ordered list of gates. Each gate "
        "receives the proposed `ToolCall`, the current `AgentState` and the registry, and "
        "returns a `GateResult` carrying one of three decisions: `ALLOW`, `DENY` or "
        "`ESCALATE`. The executor runs the gates in order and stops at the first "
        "non-`ALLOW` verdict, returning a failed `ToolResult` that records which gate "
        "objected and why. Only a call that clears every gate reaches the tool."
    ),
    new_code_cell(
        "from agentlab.tools import (\n"
        "    GovernedToolExecutor, ToolRegistry,\n"
        "    SyntaxGate, PolicyGate, PlausibilityGate,\n"
        "    GateResult, GateDecision, RiskLevel,\n"
        ")\n"
        "from agentlab.core.action import ToolCall\n"
        "import agentlab.capstone.banking_tools as banking_tools"
    ),
    new_markdown_cell(
        "## Registering the capstone tools\n"
        "\n"
        "The four module-level capstone tools are registered directly; each is a `Tool` "
        "carrying its typed input and output schema and a declared risk level. The "
        "policy-search tool is omitted here because its retrieval reads the deployed store "
        "on the GPU, and the gates below act on the call rather than on the tool body, so "
        "the demonstration needs no tool execution. The registry is the action space the "
        "gates validate against."
    ),
    new_code_cell(
        "registry = ToolRegistry()\n"
        "for tool in (banking_tools.classify_complaint,\n"
        "             banking_tools.extract_facts,\n"
        "             banking_tools.flag_regulatory,\n"
        "             banking_tools.draft_response):\n"
        "    registry.register(tool)\n"
        "for tool in registry.all():\n"
        "    print(f'{tool.name:20s} risk={tool.risk.value:6s} in -> {tool.input_schema.__name__}')"
    ),
    new_markdown_cell(
        "## Assembling the governed executor\n"
        "\n"
        "The executor is constructed with an explicit gate list. The syntax gate checks "
        "that the named tool exists and that its arguments validate against the input "
        "schema; the policy gate aggregates a list of pluggable rules; the plausibility "
        "gate rejects arguments whose serialized size is implausibly large. A single "
        "authorization rule is installed in the policy gate: any call to a tool declared "
        "`HIGH` risk is escalated rather than executed."
    ),
    new_code_cell(
        "def escalate_high_risk(action: ToolCall, state) -> GateResult:\n"
        "    tool = registry.get(action.tool_name)\n"
        "    if tool.risk is RiskLevel.HIGH:\n"
        "        return GateResult(GateDecision.ESCALATE, 'authorization',\n"
        "                          f'{action.tool_name} is HIGH risk; requires review')\n"
        "    return GateResult(GateDecision.ALLOW, 'authorization')\n"
        "\n"
        "executor = GovernedToolExecutor(\n"
        "    registry,\n"
        "    gates=[SyntaxGate(), PolicyGate([escalate_high_risk]), PlausibilityGate()],\n"
        ")\n"
        "print('gates:', [g.name for g in executor._gates])"
    ),
    new_markdown_cell(
        "## A well-formed call clears the gates\n"
        "\n"
        "A call to `classify_complaint` carrying the single required `message` field is "
        "well-typed, names a `LOW`-risk tool and is small, so it passes all three gates. "
        "The classifier body itself loads a model, so the call is not executed here; the "
        "syntax gate alone is exercised to confirm the proposal is admissible. A `GateResult` "
        "of `ALLOW` from every gate is the precondition for the executor to proceed."
    ),
    new_code_cell(
        "good = ToolCall(tool_name='classify_complaint',\n"
        "                arguments={'message': 'I was charged a $35 overdraft fee I did not authorize.'})\n"
        "for gate in (SyntaxGate(), PolicyGate([escalate_high_risk]), PlausibilityGate()):\n"
        "    r = gate.check(good, None, registry)\n"
        "    print(f'{r.gate_name:14s} -> {r.decision.value}')"
    ),
    new_markdown_cell(
        "## A malformed call is denied by the syntax gate\n"
        "\n"
        "The first failure mode is a call whose arguments do not validate against the "
        "tool's input schema. Here the required `message` field is replaced by an "
        "unrecognized `msg` field. The executor runs the syntax gate first, which attempts "
        "to validate the arguments against `ClassifyInput`, fails, and returns `DENY`. The "
        "executor stops and returns a failed `ToolResult`; the tool body never runs."
    ),
    new_code_cell(
        "malformed = ToolCall(tool_name='classify_complaint',\n"
        "                     arguments={'msg': 'wrong field name'})\n"
        "result = executor.execute(malformed, state=None)\n"
        "print('success :', result.success)\n"
        "print('error   :', result.error)\n"
        "print('gates   :', [(g.gate_name, g.decision.value) for g in result.gate_results])"
    ),
    new_markdown_cell(
        "## A call to an unknown tool is denied\n"
        "\n"
        "A proposal may also name a tool outside the registry, whether through a model "
        "error or an attempt to reach an action the agent was never granted. The syntax "
        "gate looks the tool up first and returns `DENY` when the lookup fails, so an "
        "action outside the declared space is refused at the boundary."
    ),
    new_code_cell(
        "unknown = ToolCall(tool_name='delete_account', arguments={'id': 1})\n"
        "result = executor.execute(unknown, state=None)\n"
        "print('success :', result.success)\n"
        "print('error   :', result.error)"
    ),
    new_markdown_cell(
        "## A high-risk call is escalated by the policy gate\n"
        "\n"
        "The third failure mode is a call that is well-typed but carries more authority "
        "than the agent may exercise unsupervised. To read this on the capstone without a "
        "genuinely destructive tool, a variant registry registers `draft_response` at "
        "`HIGH` risk and a matching executor is built over it. A well-formed call to that "
        "tool passes the syntax gate, but the authorization rule in the policy gate returns "
        "`ESCALATE`, and the executor stops with a failed result that names the escalating "
        "gate. In deployment such a call is routed to a human reviewer rather than silently "
        "dropped."
    ),
    new_code_cell(
        "from dataclasses import replace\n"
        "\n"
        "hi_registry = ToolRegistry()\n"
        "hi_registry.register(replace(banking_tools.draft_response, risk=RiskLevel.HIGH))\n"
        "\n"
        "def escalate_high_risk_hi(action: ToolCall, state) -> GateResult:\n"
        "    if hi_registry.get(action.tool_name).risk is RiskLevel.HIGH:\n"
        "        return GateResult(GateDecision.ESCALATE, 'authorization',\n"
        "                          f'{action.tool_name} is HIGH risk; requires review')\n"
        "    return GateResult(GateDecision.ALLOW, 'authorization')\n"
        "\n"
        "hi_executor = GovernedToolExecutor(\n"
        "    hi_registry,\n"
        "    gates=[SyntaxGate(), PolicyGate([escalate_high_risk_hi]), PlausibilityGate()],\n"
        ")\n"
        "\n"
        "call = ToolCall(tool_name='draft_response',\n"
        "                arguments={'category': 'billing_dispute', 'issue': 'unauthorized fee',\n"
        "                           'policy_evidence': [{'text': 'Reg E: provisional credit within 10 days.'}],\n"
        "                           'message': 'I want the $35 fee reversed.'})\n"
        "result = hi_executor.execute(call, state=None)\n"
        "print('success :', result.success)\n"
        "print('error   :', result.error)\n"
        "print('gates   :', [(g.gate_name, g.decision.value) for g in result.gate_results])"
    ),
    new_markdown_cell(
        "The executor stops at the first gate to escalate and never invokes the tool, so "
        "the record shows the syntax gate allowing the call and the policy gate escalating "
        "it. A reviewer with authority to override may re-run the call through "
        "`force_execute`, which bypasses the gates and marks the result as a human "
        "override, leaving an audit trail that the gates were skipped and why."
    ),
    new_markdown_cell(
        "This is the capstone's realization of Chapter~6: every tool call the agent "
        "proposes is routed through one `GovernedToolExecutor`, and a fixed sequence of "
        "gates validates the call's form, its authorization and its plausibility before the "
        "tool acts. Chapter~5 established that the action space is a registry of typed "
        "tools; Chapter~6 adds the guard that runs those typed calls safely. The capstone "
        "chapter (Chapter~15) fills the policy gate with the banking rules the agent must "
        "obey and assembles the five tools and the executor into the governed workflow."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
