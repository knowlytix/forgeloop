#!/usr/bin/env python
"""Builder for notebooks/04_capstone_build.ipynb.

Capstone BUILD series, Chapter 4 (Tasks, State, Actions). Formalizes the three types
the loop moves between: the TaskSpec the agent is given, the AgentState it carries, and
the discriminated-union Action space it may propose from (ToolCall, AskUser, Finish,
Escalate). Escalate joins the action space here as the mechanism later governance uses.

Teaching notebook: real imports, no pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "04_capstone_build.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone build --- Chapter 4: Tasks, State, and Actions\n"
        "\n"
        "The loop in Chapter~1 passed three types between the agent and the environment "
        "without naming them: the task the agent was given, the state it carried and the "
        "actions it proposed. Chapter~4 makes those types explicit, because the "
        "governance, evaluation and audit layers added later all read them. A task is "
        "more than a prompt, state is where the agent is rather than what was said, and "
        "an action is one of a fixed set of typed possibilities."
    ),
    new_markdown_cell(
        "## The task is more than a prompt\n"
        "\n"
        "A `TaskSpec` carries the goal, the inputs the agent acts on, the outputs it is "
        "expected to produce and the constraints and validation rules it must satisfy. "
        "The complaint agent's task names the customer message as its input and the "
        "classification and draft as expected outputs, and can carry validation rules "
        "the harness checks a result against."
    ),
    new_code_cell(
        "from agentlab.core.task import TaskSpec, ValidationRule\n"
        "\n"
        "task = TaskSpec(\n"
        "    goal='handle a customer complaint under banking policy',\n"
        "    inputs={'message': 'I was charged a $35 overdraft fee I did not authorize.'},\n"
        "    expected_outputs=['classification', 'draft_response'],\n"
        "    constraints=['no unauthorized fee-waiver promises', 'cite governing policy'],\n"
        "    validation=[ValidationRule(name='grounded',\n"
        "                               description='every claim carries evidence')],\n"
        ")\n"
        "print('goal      :', task.goal)\n"
        "print('inputs    :', task.inputs)\n"
        "print('constraints:', task.constraints)\n"
        "print('validation:', [r.name for r in task.validation])"
    ),
    new_markdown_cell(
        "## State is where the agent is\n"
        "\n"
        "`AgentState` is serializable so it can be persisted, replayed and audited. It "
        "carries the task, the step counter, the accumulated tool results and a status "
        "the loop reads. Because it round-trips through a dict, the audit log can store "
        "a hash of the state before each step and the replay machinery can reconstruct "
        "the case exactly."
    ),
    new_code_cell(
        "from agentlab.core.state import AgentState\n"
        "\n"
        "state = AgentState(task=task)\n"
        "print('initial status:', state.status, '| step:', state.step)\n"
        "restored = AgentState.from_dict(state.to_dict())\n"
        "print('round-trips   :', restored.to_dict() == state.to_dict())"
    ),
    new_markdown_cell(
        "## Actions are a closed, typed set\n"
        "\n"
        "An agent may only ever return one of four actions: a `ToolCall`, an `AskUser`, "
        "a `Finish` or an `Escalate`. The set is a discriminated union keyed on `kind`, "
        "so a proposed action can be parsed and dispatched without guessing its shape. "
        "Chapter~1 used `ToolCall` and `Finish`; `Escalate` is the action the governance "
        "layer and the reasoning check raise when a case must go to a human, and "
        "`parse_action` reconstructs any of them from a serialized record."
    ),
    new_code_cell(
        "from agentlab.core.action import ToolCall, AskUser, Finish, Escalate, parse_action\n"
        "\n"
        "actions = [\n"
        "    ToolCall(tool_name='classify_complaint', arguments={'message': task.inputs['message']}),\n"
        "    AskUser(question='Which account was charged?'),\n"
        "    Escalate(reason='regulatory risk flagged: UDAAP', context={'flags': ['UDAAP']}),\n"
        "    Finish(output={'classification': 'complaint'}),\n"
        "]\n"
        "for a in actions:\n"
        "    print(f'{a.kind:11s} -> {parse_action(a.model_dump()).__class__.__name__}')"
    ),
    new_markdown_cell(
        "These three types are the vocabulary the rest of the capstone is written in. "
        "The `ComplaintAgent` of Chapter~8 is a function from `AgentState` to one of "
        "these `Action` types; the governance harness of Chapter~12 logs each "
        "`(state, action, observation)` transition; the trajectory of Chapter~10 is the "
        "recorded sequence of them. Chapter~5 fixes the concrete `ToolCall` targets by "
        "defining the full typed action space."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
