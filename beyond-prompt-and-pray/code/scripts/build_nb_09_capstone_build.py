#!/usr/bin/env python
"""Builder for notebooks/09_capstone_build.ipynb.

Capstone BUILD series, Chapter 9 (Memory). The complaint agent's working memory is the
accumulated tool_results carried in AgentState: a later step reads an earlier step's
output rather than recomputing it. This notebook shows the memory abstractions
(WorkingMemory, ShortTermMemory) and then the capstone's concrete pattern -- the
extraction from step 1 is reused by steps 2, 3 and 4, so the message is parsed once.

Pure state manipulation (no model load). Teaching notebook: real imports, no
pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "09_capstone_build.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone build --- Chapter 9: Memory\n"
        "\n"
        "Each step of the workflow produces something a later step needs: the extraction "
        "names the product and issue the policy search queries on, and the classification "
        "and policy evidence feed the draft. If the agent recomputed those each time it "
        "would parse the message repeatedly and risk two steps disagreeing. Chapter~9 is "
        "about the memory that lets a later step read an earlier step's result. In the "
        "capstone that memory is the accumulated tool results carried in the agent state."
    ),
    new_markdown_cell(
        "## The memory abstraction\n"
        "\n"
        "`WorkingMemory` holds the facts and open questions for a single task, and "
        "`ShortTermMemory` is a bounded store that keeps the most recent items. These "
        "are the general-purpose structures; a task that accumulates named facts as it "
        "runs writes them to a working memory keyed by the task."
    ),
    new_code_cell(
        "from agentlab.memory.short_term import WorkingMemory, ShortTermMemory\n"
        "from agentlab.memory.base import MemoryItem, MemoryKind\n"
        "\n"
        "wm = WorkingMemory(task_id='case-002')\n"
        "wm.facts['product'] = 'checking_account'\n"
        "wm.facts['issue'] = 'unauthorized_fee'\n"
        "print('working facts :', wm.facts)\n"
        "\n"
        "stm = ShortTermMemory(maxlen=8)\n"
        "stm.add(MemoryItem(content='classified as complaint', kind=MemoryKind.TOOL))\n"
        "print('recent items  :', [i.content for i in stm.query('classification', k=3)])"
    ),
    new_markdown_cell(
        "## The capstone's working memory is the state\n"
        "\n"
        "The `ComplaintAgent` does not reach for a separate store: its working memory is "
        "`state.tool_results`, the list of outputs accumulated as the workflow runs. A "
        "later step reads an earlier result by index through the agent's `_output` "
        "helper. This is what lets step~2 reuse the step~1 extraction instead of parsing "
        "the message again."
    ),
    new_code_cell(
        "from agentlab.capstone.complaint_agent import ComplaintAgent\n"
        "from agentlab.core.state import AgentState\n"
        "from agentlab.core.task import TaskSpec\n"
        "\n"
        "agent = ComplaintAgent()\n"
        "task = TaskSpec(goal='handle a complaint',\n"
        "                inputs={'message': 'I was charged a $35 overdraft fee I did not authorize.'})\n"
        "state = AgentState(task=task, step=2)\n"
        "# Simulate the memory after classify (step 0) and extract (step 1) have run.\n"
        "state.tool_results.append({'success': True, 'output': {'category': 'complaint', 'confidence': 1.0}})\n"
        "state.tool_results.append({'success': True, 'output': {\n"
        "    'product': 'checking_account', 'issue': 'unauthorized_fee',\n"
        "    'extraction': {'bound': True}, 'query_facts': [('overdraft_fee', 'has_issue', 'unauthorized_fee')]}})\n"
        "\n"
        "# At step 2 the agent proposes search_policy, reading the extraction from memory.\n"
        "action = agent.propose_action(state)\n"
        "print('proposes     :', action.kind, '->', action.tool_name)\n"
        "print('reuses step-1 extraction:', action.arguments.get('extraction'))"
    ),
    new_markdown_cell(
        "The extraction computed once at step~1 is carried in the state and read again "
        "at step~2, so the message is parsed exactly once and every downstream step "
        "grounds on the same facts. The tool results are also what the reasoning trace "
        "of Chapter~3 is reconstructed from and what the audit log of Chapter~12 records. "
        "Chapter~10 turns the completed sequence of these steps into a trajectory and "
        "scores it."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
