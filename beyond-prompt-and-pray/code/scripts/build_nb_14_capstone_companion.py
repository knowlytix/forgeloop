#!/usr/bin/env python
"""Builder for notebooks/14_capstone_companion.ipynb.

Capstone companion to Chapter 14 (Multi-Agent: when, how, and when not to). The
chapter's decision framework -- the three-question test -- is read on the running
banking complaint agent, which is deliberately a SINGLE governed agent with a
fixed tool workflow. Teaching notebook in the style of the main chapter notebooks:
real capstone imports, no pre-embedded outputs (the reader runs it). The
inspection is structural and cheap; the agent is not run and no GPU is required.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "14_capstone_companion.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone companion --- Chapter 14: Multi-Agent, When and When Not\n"
        "\n"
        "Chapter~14 treats the multi-agent question as a design decision to be argued, "
        "not a default to be adopted. Before a system is decomposed into cooperating "
        "agents, three questions are put to it. First, can a single governed agent "
        "equipped with planning and tools already perform the task? Second, do the "
        "proposed workers genuinely differ in capability, in the policy they operate "
        "under, or in the audit boundary they sit behind? Third, are the additional "
        "costs of coordination, latency and failure surface justified by what "
        "decomposition returns? This companion applies that test to the capstone "
        "banking complaint agent, which is built as a single governed agent over a "
        "fixed tool workflow, and reads why that is the right choice here."
    ),
    new_markdown_cell(
        "The capstone processes one customer complaint through a fixed sequence: "
        "classify the message, extract its grounded facts, search the governing "
        "policy, flag any regulatory risk, then draft a reply or escalate. A natural "
        "first instinct is to make each of these a separate agent. The three-question "
        "test rejects that instinct, and the code shows why: the workflow is a single "
        "`ComplaintAgent` whose `propose_action` method selects the next tool by step, "
        "under one governance harness. The tools differ; the agents do not, because "
        "there is only one."
    ),
    new_code_cell(
        "from agentlab.capstone import complaint_agent as ca\n"
        "from agentlab.capstone.complaint_agent import ComplaintAgent, build_complaint_harness\n"
        "from agentlab.core import Escalate, Finish, ToolCall\n"
        "\n"
        "# The workflow the single agent walks, as the GMS store's has_enables DAG names it.\n"
        "print('workflow nodes :', ca._WORKFLOW_NODES)\n"
        "print('tool -> node   :', ca._TOOL_NODE_MAP)\n"
        "print('agent class    :', ComplaintAgent.__mro__[0].__name__,\n"
        "      '(single BaseAgent subclass)')"
    ),
    new_markdown_cell(
        "## Question one: can one governed agent with planning and tools do it?\n"
        "\n"
        "The first question asks whether decomposition is necessary at all. In the "
        "capstone the planning is deterministic and local: the agent inspects the "
        "state it has accumulated and returns the next action. Reading the body of "
        "`propose_action` makes the plan explicit --- it is a step-indexed dispatch "
        "over the five tools, with escalation branches where a prior result forces a "
        "human. There is no need for a second agent to hold any part of this plan, "
        "because the plan is small, sequential and fully observable from the state."
    ),
    new_code_cell(
        "import inspect\n"
        "\n"
        "src = inspect.getsource(ComplaintAgent.propose_action)\n"
        "# Show the plan as branches: each step selects one tool or an escalation.\n"
        "for line in src.splitlines():\n"
        "    s = line.strip()\n"
        "    if s.startswith('if state.step') or 'tool_name=' in s or s.startswith('return Escalate') \\\n"
        "       or s.startswith('return Finish'):\n"
        "        print(s)"
    ),
    new_markdown_cell(
        "The dispatch above is the whole planner. Five steps, each a typed `ToolCall`, "
        "with `Escalate` on a forced hand-off and `Finish` at the end. Handing any one "
        "step to a separate agent would add a message boundary without adding a "
        "decision --- the branch is already resolved by the state the single agent "
        "holds. Question one is answered in the affirmative, so the remaining two "
        "questions must clear a high bar to overturn it."
    ),
    new_markdown_cell(
        "## Question two: do the workers really differ?\n"
        "\n"
        "The second question asks whether the candidate workers differ in a way that "
        "warrants separation --- in capability, in the policy they run under, or in "
        "the audit boundary that contains them. The capstone's tools differ in "
        "capability: a classifier, a fact extractor, a retriever, a regulatory checker "
        "and a drafter. Difference in *capability* is served by giving one agent "
        "different tools. It is difference in *policy* or in *audit boundary* that "
        "would justify separate agents, because those are properties of the actor, not "
        "of the action. The capstone applies a single policy set and a single audit "
        "boundary across all five steps, which is visible in the shared harness."
    ),
    new_code_cell(
        "hsrc = inspect.getsource(build_complaint_harness)\n"
        "# One PolicyEngine, one set of gates, one executor -- one audit boundary.\n"
        "for line in hsrc.splitlines():\n"
        "    s = line.strip()\n"
        "    if any(k in s for k in ('pii_policy', 'semantic_', 'PolicyEngine(',\n"
        "                            'gates =', 'GovernedToolExecutor', 'GovernanceHarness(')):\n"
        "        print(s)"
    ),
    new_markdown_cell(
        "Every tool call in the workflow passes through the same `PolicyEngine`, the "
        "same gate list --- syntax, policy, GMS plausibility --- and the same "
        "`GovernedToolExecutor`, all wrapped by one `GovernanceHarness`. The audit log "
        "that results is a single, ordered trajectory. Were the steps split across "
        "agents, this one boundary would fracture into several, each needing its own "
        "policy attachment and its own audit stitching, with no compensating gain: the "
        "steps do not require different policy, and a customer complaint must be "
        "reconstructable as one record. The workers differ in what they do but not in "
        "how they are governed, so question two argues against decomposition."
    ),
    new_markdown_cell(
        "## Question three: are the costs justified?\n"
        "\n"
        "The third question weighs the coordination cost of a multi-agent design "
        "against its return. A key property of the capstone is that failure is "
        "handled by short-circuiting the *same* trajectory to a human, not by "
        "negotiating across agents. The very first lines of `propose_action` scan the "
        "accumulated tool results and escalate on any failure, which is how an "
        "injected-PII or prompt-injection case is stopped. Cross-agent coordination "
        "would have to reproduce this escalation contract at every boundary, "
        "multiplying the failure surface it is meant to contain."
    ),
    new_code_cell(
        "# The failure-to-human contract lives in one place: the head of propose_action.\n"
        "head = src.splitlines()[:8]\n"
        "print(chr(10).join(head))\n"
        "print()\n"
        "print('escalation is a first-class Action:', Escalate.model_fields['kind'].default,\n"
        "      '| finish:', Finish.model_fields['kind'].default,\n"
        "      '| tool call:', ToolCall.model_fields['kind'].default)"
    ),
    new_markdown_cell(
        "Escalation, finishing and calling a tool are the three actions the single "
        "agent chooses among, each a typed member of the action union. The escalation "
        "path is a single conditional at the top of the loop, so any tool failure "
        "anywhere in the workflow routes to one human hand-off with one context "
        "record. A multi-agent version would pay coordination and latency cost to "
        "distribute a plan that is already sequential and a failure contract that is "
        "already centralized. Under the third question the costs are not justified."
    ),
    new_markdown_cell(
        "## Reading the verdict\n"
        "\n"
        "The three questions converge. A single governed agent with a step-indexed "
        "planner and five typed tools performs the task; the steps differ in "
        "capability but share one policy set and one audit boundary; and the "
        "coordination cost of splitting them returns nothing the single agent does not "
        "already provide, while enlarging the failure surface. The capstone is "
        "therefore a single agent by design, and Chapter~14's framework names the "
        "conditions under which that verdict would change --- a step that needed a "
        "genuinely different policy, a different trust boundary, or a capability no "
        "single agent could hold --- none of which the complaint workflow presents."
    ),
    new_markdown_cell(
        "This is the capstone's realization of Chapter~14: multi-agent structure is "
        "adopted only when the three-question test demands it, and here it does not, so "
        "the system stays a single governed agent whose plan and audit trail are one. "
        "Chapter~15 turns from this design question to evaluation, asking how the "
        "assembled agent is tested against the behaviors the earlier chapters "
        "specified, before the capstone chapter puts the whole workflow to work."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
