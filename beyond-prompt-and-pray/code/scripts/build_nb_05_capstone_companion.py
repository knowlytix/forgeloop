#!/usr/bin/env python
"""Builder for notebooks/05_capstone_companion.ipynb.

Capstone companion to Chapter 5 (Tools as Typed Actions): the concept read on the
running banking complaint agent. Teaching notebook in the style of the main chapter
notebooks -- real capstone imports, no pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "05_capstone_companion.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone companion --- Chapter 5: Tools as Typed Actions\n"
        "\n"
        "Chapter~5 argues that an agent's actions must be typed: each tool declares a "
        "schema for its input and its output, so a malformed call is rejected at the "
        "boundary rather than after it has acted. This companion reads that principle on "
        "the capstone banking complaint agent, whose five tools are each defined as a "
        "typed action over a Pydantic input and output model."
    ),
    new_markdown_cell(
        "The five tools are `classify_complaint`, `extract_facts`, `search_policy`, "
        "`flag_regulatory` and `draft_response`. Each carries an input model and an "
        "output model; the harness validates a proposed call against the input model "
        "before the tool runs, and validates the return against the output model before "
        "any downstream tool consumes it."
    ),
    new_code_cell(
        "from agentlab.capstone import banking_tools\n"
        "from agentlab.capstone.banking_tools import (\n"
        "    ClassifyInput, ClassifyOutput,\n"
        "    ExtractInput, ExtractOutput,\n"
        "    FlagInput, FlagOutput,\n"
        "    DraftInput, DraftOutput,\n"
        ")"
    ),
    new_markdown_cell(
        "## The input and output schema of one tool\n"
        "\n"
        "The classifier maps a customer message to one of three labels. Its typed "
        "contract is the pair of models below: the input names the single `message` "
        "field, the output names the `category` label and a confidence. Reading the JSON "
        "schema shows exactly what the harness checks."
    ),
    new_code_cell(
        "import json\n"
        "print('ClassifyInput  :', json.dumps(ClassifyInput.model_json_schema()['properties'], indent=2))\n"
        "print('ClassifyOutput :', json.dumps(ClassifyOutput.model_json_schema()['properties'], indent=2))"
    ),
    new_markdown_cell(
        "## Assembling the typed toolset\n"
        "\n"
        "`register_all` binds the five tools into a registry the agent draws from. The "
        "policy-search tool is constructed against the policy directory, because its "
        "retrieval reads the deployed store; the other four are module-level tool "
        "objects. The registry is the typed action space the agent may propose from."
    ),
    new_code_cell(
        "from agentlab.tools import ToolRegistry\n"
        "from pathlib import Path\n"
        "\n"
        "root = Path('.') if Path('data').exists() else Path('..')\n"
        "registry = ToolRegistry()\n"
        "banking_tools.register_all(registry, policies_dir=root / 'data' / 'policies')\n"
        "for name in ('classify_complaint', 'extract_facts', 'search_policy',\n"
        "             'flag_regulatory', 'draft_response'):\n"
        "    tool = registry.get(name)\n"
        "    print(f'{tool.name:20s} in -> {tool.input_schema.__name__:16s} '\n"
        "          f'out -> {tool.output_schema.__name__:16s} risk={tool.risk}')"
    ),
    new_markdown_cell(
        "## A malformed call is rejected at the boundary\n"
        "\n"
        "The point of a typed action is that an ill-formed proposal never reaches the "
        "tool body. Constructing an input model with a missing or wrong-typed field "
        "raises a validation error, so the harness can refuse the call before it acts."
    ),
    new_code_cell(
        "from pydantic import ValidationError\n"
        "\n"
        "ok = ClassifyInput(message='I was charged a $35 overdraft fee I did not authorize.')\n"
        "print('valid input :', ok)\n"
        "try:\n"
        "    ClassifyInput()          # missing the required message field\n"
        "except ValidationError as e:\n"
        "    print('rejected    :', e.errors()[0]['type'], '-', e.errors()[0]['loc'])"
    ),
    new_markdown_cell(
        "This is the capstone's realization of Chapter~5: the agent's action space is a "
        "registry of typed tools, and every call is checked against a declared schema on "
        "the way in and on the way out. Chapter~6 adds the execution guard that runs "
        "these validated calls safely, and the capstone chapter (Chapter~15) assembles "
        "the five tools into the governed workflow."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
