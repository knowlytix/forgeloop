#!/usr/bin/env python
"""Builder for notebooks/05_capstone_build.ipynb.

Capstone BUILD series, Chapter 5 (Tools as Typed Actions). Widens the Chapter 1
registry from the single classify tool to the full five-tool action space via
banking_tools.register_all, and shows that a proposal is validated against the input
schema at the registry boundary before it can run.

This is the progressive-build counterpart to 05_capstone_companion.ipynb: the
companion reads the typed toolset on the finished agent, whereas this notebook widens
the loop's action space as the next build step. Teaching notebook: real imports, no
pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "05_capstone_build.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone build --- Chapter 5: Tools as Typed Actions\n"
        "\n"
        "The loop in Chapter~1 had an action space of one tool. Chapter~5 makes the "
        "action space the full set the complaint agent draws from: five typed tools, "
        "each a named action over a Pydantic input and output model. This notebook "
        "widens the registry to all five and shows the property that makes the action "
        "space safe --- a malformed proposal is rejected against the input schema at the "
        "registry boundary, before any tool body runs."
    ),
    new_markdown_cell(
        "## The full typed action space\n"
        "\n"
        "`register_all` binds the five tools into a registry. Four are module-level tool "
        "objects; the policy-search tool is constructed against the policy directory "
        "because its retrieval reads the deployed store. The path is resolved so the "
        "notebook runs whether the working directory is the notebooks folder or the "
        "project root."
    ),
    new_code_cell(
        "from agentlab.tools import ToolRegistry\n"
        "from agentlab.capstone import banking_tools\n"
        "from pathlib import Path\n"
        "\n"
        "root = next((c for c in (Path('.'), Path('..'), Path('../code'), Path('code'))\n"
        "             if (c / 'data' / 'policies').exists()), Path('.'))\n"
        "registry = ToolRegistry()\n"
        "banking_tools.register_all(registry, policies_dir=root / 'data' / 'policies')\n"
        "for tool in registry.all():\n"
        "    print(f'{tool.name:20s} {tool.input_schema.__name__:14s} '\n"
        "          f'-> {tool.output_schema.__name__:14s} risk={tool.risk.value}')"
    ),
    new_markdown_cell(
        "The registry is now the agent's action space: the set of typed actions it may "
        "propose. Each entry pairs an input contract with an output contract, so the "
        "harness can check a call on the way in and a result on the way out. The risk "
        "column records how much scrutiny each action warrants, which the gate stack in "
        "Chapter~6 reads when it decides whether to authorize a call."
    ),
    new_markdown_cell(
        "## A malformed call is rejected at the boundary\n"
        "\n"
        "The registry validates a proposed call against the tool's input schema before "
        "the tool runs. A well-formed proposal is parsed into the typed input model; a "
        "proposal missing a required field raises a validation error, so an ill-formed "
        "action never reaches the tool body. This is the same `registry.validate` the "
        "environment called in Chapter~1, seen now as the boundary check it is."
    ),
    new_code_cell(
        "from pydantic import ValidationError\n"
        "\n"
        "ok = registry.validate(\n"
        "    'classify_complaint',\n"
        "    {'message': 'I was charged a $35 overdraft fee I did not authorize.'},\n"
        ")\n"
        "print('accepted   :', type(ok).__name__, '->', ok)\n"
        "\n"
        "try:\n"
        "    registry.validate('classify_complaint', {})   # missing the message field\n"
        "except ValidationError as e:\n"
        "    err = e.errors()[0]\n"
        "    print('rejected    :', err['type'], '-', err['loc'])"
    ),
    new_markdown_cell(
        "This is the capstone's realization of Chapter~5: the agent's action space is a "
        "registry of typed tools, and every proposal is validated against a declared "
        "schema before it can act. The loop from Chapter~1 now proposes from these five "
        "actions instead of one. Chapter~6 wraps the registry in the governed executor "
        "that adds the gate stack around this boundary check, and Chapter~15 assembles "
        "the five tools and the gates into the shipped harness, at which point the "
        "action space here is exactly the one `build_complaint_harness` registers."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
