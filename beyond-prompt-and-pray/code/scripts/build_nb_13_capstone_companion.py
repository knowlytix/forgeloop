#!/usr/bin/env python
"""Builder for notebooks/13_capstone_companion.ipynb.

Capstone companion to Chapter 13 (Human-in-the-Loop and Escalation): the concept
read on the running banking complaint agent. Teaching notebook in the style of the
main chapter notebooks -- real capstone imports, no pre-embedded outputs (the
reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "13_capstone_companion.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone companion --- Chapter 13: Human-in-the-Loop and Escalation\n"
        "\n"
        "Chapter~13 concerns the point at which an autonomous agent stops acting and "
        "hands off to a human: the conditions that warrant escalation, and the record "
        "the agent produces so that a reviewer can decide without re-running the case. "
        "This companion reads that principle on the capstone banking complaint agent, "
        "which escalates along two distinct routes and carries, in each, enough context "
        "for a human decision."
    ),
    new_markdown_cell(
        "The capstone escalates on two kinds of condition. The first is a *gate refusal*: "
        "a governance gate on the executor rejects a proposed tool call --- personally "
        "identifiable information detected in an argument, or a prompt-injection pattern "
        "in the message. The second is *regulatory risk*: the `flag_regulatory` tool marks "
        "a complaint as implicating UDAAP or Regulation~X, and the agent hands the case to "
        "a human rather than draft an automated reply. The two routes reach a human through "
        "different mechanisms, examined in turn below."
    ),
    new_code_cell(
        "from agentlab.governance.escalation import (\n"
        "    EscalationRequest,\n"
        "    HumanResponse,\n"
        "    HumanDecision,\n"
        "    ScriptedReviewer,\n"
        "    HumanReviewer,\n"
        ")\n"
        "from agentlab.governance.policies import pii_policy, prompt_injection_policy\n"
        "from agentlab.tools.executor import GateDecision, GateResult\n"
        "from agentlab.core.action import ToolCall, Escalate"
    ),
    new_markdown_cell(
        "## The escalation record\n"
        "\n"
        "An escalation is not a bare signal; it is a record. `EscalationRequest` carries the "
        "`run_id` and `step` that locate the case in the audit log, a `reason` in plain "
        "language, the `proposed_action` that triggered the handoff, and the `gate_results` "
        "that decided it. The record serializes to JSON, so it survives being written to a "
        "queue and read back by a reviewer who was not present when the agent ran."
    ),
    new_code_cell(
        "request = EscalationRequest(\n"
        "    run_id='case-4417',\n"
        "    step=0,\n"
        "    reason='pii_policy: detected PII: ssn',\n"
        "    proposed_action={'kind': 'tool_call', 'tool_name': 'classify_complaint',\n"
        "                     'arguments': {'message': 'my SSN is 123-45-6789 and I was overcharged'}},\n"
        "    gate_results=[{'gate': 'pii_policy', 'decision': 'escalate',\n"
        "                   'reason': 'detected PII: ssn'}],\n"
        ")\n"
        "print(request.to_json(indent=2))"
    ),
    new_markdown_cell(
        "The record round-trips: `from_json` reconstructs an equal request from the serialized "
        "form, which is the property a durable review queue requires."
    ),
    new_code_cell(
        "restored = EscalationRequest.from_json(request.to_json())\n"
        "print('round-trips equal:', restored == request)"
    ),
    new_markdown_cell(
        "## Route one: a gate refusal escalates\n"
        "\n"
        "The governance gates on the executor decide, per proposed tool call, whether the call "
        "may proceed. A gate returns a `GateResult` carrying a `GateDecision`: `ALLOW` lets the "
        "call run, `DENY` blocks it, and `ESCALATE` requests a human. The PII gate escalates "
        "when an argument contains a social-security or card number; the prompt-injection gate "
        "denies a call whose arguments carry an override instruction. Both are read below on a "
        "message that carries a social-security number."
    ),
    new_code_cell(
        "call = ToolCall(\n"
        "    tool_name='classify_complaint',\n"
        "    arguments={'message': 'my SSN is 123-45-6789 and I was overcharged a $35 fee'},\n"
        ")\n"
        "pii = pii_policy(call, None)\n"
        "inj = prompt_injection_policy(call, None)\n"
        "print(f'{pii.gate_name:24s} {pii.decision.value:9s} {pii.reason}')\n"
        "print(f'{inj.gate_name:24s} {inj.decision.value:9s} {inj.reason or \"(clean)\"}')"
    ),
    new_markdown_cell(
        "A gate that returns `ESCALATE` does not itself contact a human. The governance harness "
        "translates the escalating gate result into an `EscalationRequest` and routes it to a "
        "`HumanReviewer` --- the boundary between the automated system and the person. The "
        "harness builds exactly the record shown below from the gate's decision and the proposed "
        "action."
    ),
    new_code_cell(
        "escalating = [g for g in (pii, inj) if g.decision == GateDecision.ESCALATE]\n"
        "reason = '; '.join(f'{g.gate_name}: {g.reason}' for g in escalating)\n"
        "gate_refusal = EscalationRequest(\n"
        "    run_id='case-4417',\n"
        "    step=0,\n"
        "    reason=reason,\n"
        "    proposed_action=call.model_dump(),\n"
        "    gate_results=[{'gate': g.gate_name, 'decision': g.decision.value, 'reason': g.reason}\n"
        "                  for g in (pii, inj)],\n"
        ")\n"
        "print(gate_refusal.to_json(indent=2))"
    ),
    new_markdown_cell(
        "## The reviewer returns a decision\n"
        "\n"
        "A `HumanReviewer` maps an `EscalationRequest` to a `HumanResponse`, whose `decision` is "
        "one of `APPROVE`, `DENY` or `DEFER`. `ScriptedReviewer` supplies a fixed response and "
        "captures every request it sees, which is what makes an escalation path testable without "
        "a person in the loop; `CLIReviewer` reads the decision from standard input for "
        "interactive use. The scripted reviewer below denies the PII case, and the captured "
        "request confirms the reviewer saw the full record."
    ),
    new_code_cell(
        "reviewer = ScriptedReviewer(HumanResponse(HumanDecision.DENY, note='redact SSN before reprocessing'))\n"
        "print('is a HumanReviewer:', isinstance(reviewer, HumanReviewer))\n"
        "response = reviewer.review(gate_refusal)\n"
        "print('decision :', response.decision.value)\n"
        "print('note     :', response.note)\n"
        "print('captured :', len(reviewer.requests), 'request(s);',\n"
        "      'reason =', reviewer.requests[0].reason)"
    ),
    new_markdown_cell(
        "The harness acts on the decision. `APPROVE` overrides the gate and forces the call; "
        "`DEFER` ends the run in escalated status, leaving the case for a human; `DENY` keeps the "
        "refusal in place. The decision is recorded in the audit log alongside the request, so "
        "the disposition of every escalated case is reconstructable after the fact."
    ),
    new_markdown_cell(
        "## Route two: regulatory risk escalates from within the workflow\n"
        "\n"
        "The second route does not pass through the executor's gates. When `flag_regulatory` "
        "marks a complaint as implicating UDAAP or Regulation~X, the complaint agent emits an "
        "`Escalate` action directly, in place of proposing the `draft_response` call. A "
        "regulated matter is handed to a human rather than answered by the model. The `Escalate` "
        "action carries the same two elements as the gate route --- a plain-language `reason` and "
        "a structured `context` --- so the two routes converge on one auditable handoff shape."
    ),
    new_code_cell(
        "# What the agent constructs when flag_regulatory reports escalate=True.\n"
        "flags = {'flags': ['UDAAP'], 'escalate': True}\n"
        "regulatory = Escalate(\n"
        "    reason=f\"regulatory risk flagged: {flags['flags']}\",\n"
        "    context={'flags': flags['flags']},\n"
        ")\n"
        "print('kind    :', regulatory.kind)\n"
        "print('reason  :', regulatory.reason)\n"
        "print('context :', regulatory.context)"
    ),
    new_markdown_cell(
        "## The two routes side by side\n"
        "\n"
        "The gate route escalates *before* a tool acts, on a property of the proposed call (PII, "
        "injection); the regulatory route escalates *within* the workflow, on the result of a "
        "tool (`flag_regulatory`). Both stop autonomous action and produce a record a human can "
        "act on. The table below summarizes the routing the capstone implements."
    ),
    new_code_cell(
        "routes = [\n"
        "    ('PII in argument',        'pii_policy',              'ESCALATE (gate refusal)', 'EscalationRequest -> HumanReviewer'),\n"
        "    ('prompt-injection intent','prompt_injection_policy', 'DENY (gate refusal)',     'call blocked; no autonomous action'),\n"
        "    ('UDAAP / Reg_X flagged',  'flag_regulatory',         'Escalate action',          'agent hands off in place of draft'),\n"
        "]\n"
        "print(f'{\"condition\":26s} {\"decided by\":24s} {\"outcome\":26s} handoff')\n"
        "for cond, by, outcome, handoff in routes:\n"
        "    print(f'{cond:26s} {by:24s} {outcome:26s} {handoff}')"
    ),
    new_markdown_cell(
        "This is the capstone's realization of Chapter~13: escalation is a defined boundary, not "
        "an exception. A gate refusal or a regulatory flag stops the agent, and the handoff is a "
        "structured record --- an `EscalationRequest` routed to a `HumanReviewer`, or an "
        "`Escalate` action carrying its reason and context --- that a person can decide on and an "
        "auditor can reconstruct. Chapter~15 assembles these routes with the five tools into the "
        "governed workflow the capstone runs end to end."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
