#!/usr/bin/env python
"""Builder for notebooks/04_capstone_companion.ipynb.

Capstone companion to Chapter 4 (Tasks, State and Actions): the typed objects
that make an agent auditable, read on the running banking complaint agent. A
TaskSpec fixes what is being solved, an AgentState carries how far the agent
has got, and the Action union names what it may emit at each step. Teaching
notebook in the style of the main chapter notebooks -- real capstone imports,
no pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "04_capstone_companion.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone companion --- Chapter 4: Tasks, State and Actions\n"
        "\n"
        "Chapter~4 introduces the three typed objects that make an agent auditable. A "
        "`TaskSpec` states what is to be solved, together with its inputs, expected "
        "outputs, constraints and validation rules. An `AgentState` records how far the "
        "agent has progressed --- the step count, the messages, the tool results, the "
        "status. An `Action` is the single thing the agent emits on each step, drawn "
        "from a small closed union. Because each object is a validated schema, a "
        "trajectory can be serialized, replayed and inspected after the fact. This "
        "companion reads those objects on the capstone banking complaint agent."
    ),
    new_markdown_cell(
        "The capstone agent handles a customer complaint through a fixed workflow: "
        "classify, extract facts, search policy, flag regulatory risk, then draft or "
        "escalate. Each of the three Chapter~4 objects has a concrete role in that "
        "workflow. The `TaskSpec` names the complaint to be handled; the `AgentState` "
        "accumulates the result of each tool as the workflow advances; and the `Action` "
        "the agent proposes at a given step is a `ToolCall`, an `Escalate` or a "
        "`Finish`, according to what the state so far permits."
    ),
    new_code_cell(
        "from agentlab.core import (\n"
        "    TaskSpec, ValidationRule,\n"
        "    AgentState,\n"
        "    Action, ActionKind, ToolCall, AskUser, Finish, Escalate,\n"
        "    parse_action,\n"
        ")"
    ),
    new_markdown_cell(
        "## The TaskSpec for a complaint\n"
        "\n"
        "A task is more than a prompt string. The `TaskSpec` names the goal, carries the "
        "customer message as a typed input, and declares what a valid result must "
        "contain and what it must not do. The constraints and validation rules are the "
        "contract the finished trajectory is checked against; they are recorded on the "
        "task itself rather than left implicit in the agent's code."
    ),
    new_code_cell(
        "message = (\n"
        "    'I was charged a $35 overdraft fee on my checking account that I never '\n"
        "    'authorized, and the bank refuses to reverse it.'\n"
        ")\n"
        "task = TaskSpec(\n"
        "    goal='handle complaint',\n"
        "    inputs={'message': message},\n"
        "    expected_outputs=['classification', 'issue', 'risk_flags', 'draft_response'],\n"
        "    constraints=[\n"
        "        'cite a policy for any factual claim in the reply',\n"
        "        'do not promise a fee reversal without authorization',\n"
        "    ],\n"
        "    validation=[\n"
        "        ValidationRule(name='grounded', description='every claim carries an evidence pointer'),\n"
        "        ValidationRule(name='no_pii_leak', description='reply contains no account or card numbers'),\n"
        "    ],\n"
        ")\n"
        "print('goal        :', task.goal)\n"
        "print('inputs      :', list(task.inputs))\n"
        "print('expected    :', task.expected_outputs)\n"
        "print('constraints :', task.constraints)\n"
        "print('validation  :', [r.name for r in task.validation])"
    ),
    new_markdown_cell(
        "The goal field is validated on construction: it may not be empty. This is the "
        "smallest example of the chapter's discipline --- a malformed task is rejected "
        "at the boundary rather than carried silently into the loop."
    ),
    new_code_cell(
        "from pydantic import ValidationError\n"
        "\n"
        "try:\n"
        "    TaskSpec(goal='   ', inputs={'message': message})\n"
        "except ValidationError as e:\n"
        "    print('rejected empty goal:', e.errors()[0]['msg'])"
    ),
    new_markdown_cell(
        "## The AgentState the agent evolves\n"
        "\n"
        "The state records where the agent is, not where a conversation is. It is "
        "initialized from the task with an empty history, and each executed step appends "
        "a tool result and advances the step counter. Because every field is a plain "
        "serializable value, the whole state can be written to disk and reloaded without "
        "loss --- the property that makes replay and audit possible."
    ),
    new_code_cell(
        "state = AgentState(task=task)\n"
        "print('step        :', state.step)\n"
        "print('status      :', state.status)\n"
        "print('tool_results:', state.tool_results)\n"
        "\n"
        "# The workflow advances by appending tool results and stepping. Here we simulate\n"
        "# the first two nodes (classify, extract) to show the shape the agent reads.\n"
        "state.tool_results.append({'success': True, 'output': {'category': 'billing_dispute', 'confidence': 0.91}})\n"
        "state.step = 1\n"
        "state.tool_results.append({'success': True, 'output': {'issue': 'unauthorized_fee', 'product': 'checking'}})\n"
        "state.step = 2\n"
        "print('after two steps -> step', state.step, 'results', len(state.tool_results))"
    ),
    new_markdown_cell(
        "The state round-trips through a dictionary without loss. `to_dict` and "
        "`from_dict` are the serialization boundary: an audit log stores the dictionary, "
        "and a later process reconstructs the exact `AgentState` to inspect or resume."
    ),
    new_code_cell(
        "restored = AgentState.from_dict(state.to_dict())\n"
        "print('round-trip equal    :', restored == state)\n"
        "print('restored goal       :', restored.task.goal)\n"
        "print('restored step       :', restored.step)\n"
        "print('restored result[0]  :', restored.tool_results[0]['output'])"
    ),
    new_markdown_cell(
        "## The Action union the agent emits\n"
        "\n"
        "On each step the agent emits exactly one `Action`. The union is closed: a "
        "`ToolCall` invokes a named tool with arguments, an `AskUser` requests missing "
        "information, a `Finish` returns the compiled result, and an `Escalate` hands "
        "the case to a human with a reason. Each variant is discriminated by a `kind` "
        "field constrained to a single literal, so an action deserialized from an audit "
        "log parses back to the correct type."
    ),
    new_code_cell(
        "actions = [\n"
        "    ToolCall(tool_name='classify_complaint', arguments={'message': message}),\n"
        "    AskUser(question='Which account was the fee charged to?'),\n"
        "    Escalate(reason='regulatory risk flagged: UDAAP', context={'flags': ['UDAAP']}),\n"
        "    Finish(output={'recommended_action': 'respond'}),\n"
        "]\n"
        "for a in actions:\n"
        "    print(f'{a.kind:12s} -> {type(a).__name__}')\n"
        "print()\n"
        "print('ActionKind members:', [k.value for k in ActionKind])"
    ),
    new_markdown_cell(
        "The `kind` literal is what makes the union safe across the serialization "
        "boundary. `parse_action` dispatches on that field to reconstruct the concrete "
        "type from a plain dictionary, and the actions are frozen, so an action recorded "
        "in the trajectory cannot be mutated after it is emitted."
    ),
    new_code_cell(
        "logged = ToolCall(tool_name='search_policy', arguments={'query': message}).model_dump()\n"
        "print('logged dict :', logged)\n"
        "reparsed = parse_action(logged)\n"
        "print('reparsed    :', type(reparsed).__name__, '- tool', reparsed.tool_name)\n"
        "\n"
        "try:\n"
        "    reparsed.tool_name = 'other'      # actions are frozen\n"
        "except ValidationError as e:\n"
        "    print('frozen      :', e.errors()[0]['type'])"
    ),
    new_markdown_cell(
        "## The three objects in the capstone loop\n"
        "\n"
        "The complaint agent's decision rule is a pure function of the state: it reads "
        "the step count and the tool results so far, and returns the next `Action`. The "
        "cell below runs that rule directly on the two-step state assembled above, "
        "without executing any tool, to show which action the agent proposes given how "
        "far the workflow has advanced."
    ),
    new_code_cell(
        "from agentlab.capstone.complaint_agent import ComplaintAgent\n"
        "\n"
        "agent = ComplaintAgent()\n"
        "\n"
        "s0 = AgentState(task=task)\n"
        "print('step 0 ->', repr(agent.propose_action(s0)))\n"
        "\n"
        "s1 = AgentState(task=task, step=1,\n"
        "                tool_results=[{'success': True, 'output': {'category': 'billing_dispute', 'confidence': 0.91}}])\n"
        "print('step 1 ->', repr(agent.propose_action(s1)))\n"
        "\n"
        "# A recorded tool failure short-circuits the same rule to an Escalate.\n"
        "s_fail = AgentState(task=task, step=1,\n"
        "                    tool_results=[{'success': False, 'error': 'PII detected in message'}])\n"
        "print('failure->', repr(agent.propose_action(s_fail)))"
    ),
    new_markdown_cell(
        "This is the capstone's realization of Chapter~4. The `TaskSpec` fixes the "
        "problem and the contract it will be judged against; the `AgentState` carries the "
        "trajectory in a form that serializes, replays and audits without loss; and the "
        "`Action` union names the closed set of things the agent may do at each step, "
        "each variant typed and discriminated so it survives the round-trip through the "
        "log. Chapter~5 gives the tools that a `ToolCall` invokes their own typed input "
        "and output schemas, and the capstone chapter (Chapter~15) assembles the full "
        "governed workflow over these objects."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
