#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Builder for notebooks/00_setup.ipynb (CPU only; does NOT execute cells).

00_setup regenerates the artifacts the beyond-ship-and-pray chapters read.
Unlike the other topics, most of what this book *tests* is built elsewhere: the
System Under Test is the governed complaint agent from **beyond-prompt-and-pray**,
so this topic depends on that topic's GMS stores and on a recorded capstone run.

Stages:
  0. Dependency check — beyond-prompt-and-pray's Tier-1 stores must exist
     (built by that topic's notebooks/00_setup.ipynb).
  1. Capstone campaign — scripts/capstone_run.py drives the agent through the
     DoE framework and writes data/capstone_{run,rows,testset}.json, which the
     analysis/attribution chapters read. Needs an LLM.

Run: python scripts/build_nb_00_setup.py
Emits a valid nbformat-4 notebook.
"""
from __future__ import annotations

import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
NBDIR = os.path.join(HERE, "..", "..", "notebooks")

nb = nbf.v4.new_notebook()
cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src))


md(
    "# 00 — Set up the test artifacts\n"
    "\n"
    "This book *tests* an agent it does not build. The System Under Test is the\n"
    "governed complaint agent from **beyond-prompt-and-pray**, so the chapters\n"
    "here read two kinds of artifact, neither committed (both git-ignored):\n"
    "\n"
    "1. **beyond-prompt-and-pray's GMS stores** — built by *that* topic's\n"
    "   `notebooks/00_setup.ipynb`.\n"
    "2. **A recorded capstone run** — `data/capstone_{run,rows,testset}.json`,\n"
    "   produced by `scripts/capstone_run.py`, which drives the agent through the\n"
    "   DoE framework. The analysis, attribution and resilience chapters read it.\n"
    "\n"
    "**What you need**\n"
    "\n"
    "- The licensed **`knowlytix`** substrate (`pip install knowlytix` + a key at\n"
    "  `~/.knowlytix/license.key`; https://knowlytix.ai/signup/).\n"
    "- `pip install -e \".[notebooks]\"` for matplotlib/pandas used by the plots.\n"
    "- The capstone campaign runs the agent end to end and **needs an LLM**\n"
    "  (Qwen + a GPU); the model-free chapters do not.\n"
    "\n"
    "Stages are **idempotent** — skipped when their output already exists."
)

md("## 1. Bootstrap + dependency check")

code(
    "import importlib.util, os, sys\n"
    "\n"
    "# Locate this topic's code/ dir, robust to the working directory.\n"
    "_cwd = os.getcwd()\n"
    'for REPO in (_cwd, os.path.join(os.path.dirname(_cwd), "code"), os.path.join(_cwd, "code")):\n'
    '    if os.path.isdir(os.path.join(REPO, "gmstest")):\n'
    "        break\n"
    "else:\n"
    '    REPO = os.environ.get("GMSTEST_REPO", _cwd)\n'
    "for p in (REPO, os.path.join(REPO, 'scripts')):\n"
    "    if p not in sys.path:\n"
    "        sys.path.insert(0, p)\n"
    "print('repo:', REPO)\n"
    "\n"
    "if importlib.util.find_spec('knowlytix') is None:\n"
    "    raise ModuleNotFoundError(\n"
    "        'knowlytix not found. Install it (`pip install knowlytix`, licensed) '\n"
    "        'and put your key at ~/.knowlytix/license.key.')\n"
    "\n"
    "# The SUT lives in the sibling topic; its stores must be built first.\n"
    "BPP = os.path.normpath(os.path.join(REPO, '..', '..', 'beyond-prompt-and-pray', 'code'))\n"
    "needed = ['gms_banking_store', 'gms_policy_store', 'gms_regulatory_store']\n"
    "missing = [n for n in needed if not os.path.isdir(os.path.join(BPP, 'data', n))]\n"
    "for n in needed:\n"
    "    print(f\"  beyond-prompt-and-pray/{n:24} \"\n"
    "          f\"{'OK' if n not in missing else 'MISSING'}\")\n"
    "if missing:\n"
    "    print('\\n-> Run beyond-prompt-and-pray/notebooks/00_setup.ipynb first '\n"
    "          '(Tier 1 is CPU-only).')"
)

md("## 2. Idempotent stage runner")

code(
    "import subprocess, sys, os\n"
    "\n"
    "def stage(title, script, args=(), produces=()):\n"
    "    \"\"\"Run scripts/<script> unless every path in `produces` already exists.\"\"\"\n"
    "    produces = list(produces)\n"
    "    if produces and all(os.path.exists(os.path.join(REPO, p)) for p in produces):\n"
    "        print(f'\\u2713 {title}: already built \\u2014 skipping')\n"
    "        return\n"
    "    cmd = [sys.executable, os.path.join(REPO, 'scripts', script), *map(str, args)]\n"
    "    print(f'\\u25b6 {title}: python scripts/{script} ' + ' '.join(map(str, args)))\n"
    "    env = os.environ.copy()\n"
    "    env['PYTHONPATH'] = REPO + os.pathsep + env.get('PYTHONPATH', '')\n"
    "    r = subprocess.run(cmd, cwd=REPO, env=env)\n"
    "    if r.returncode:\n"
    "        raise RuntimeError(f'{script} failed (exit {r.returncode})')\n"
    "    print(f'\\u2713 {title}: done')"
)

md(
    "## 3. Capstone campaign (needs the SUT + an LLM)\n"
    "\n"
    "Drives the governed complaint agent through the DoE framework and records\n"
    "the run the analysis chapters read. This exercises the real agent, so it\n"
    "needs beyond-prompt-and-pray's stores (above) and an LLM. Set\n"
    "`RUN_CAPSTONE = True` to build it."
)

code(
    "RUN_CAPSTONE = False  # set True to record a capstone run (needs the SUT + an LLM)\n"
    "\n"
    "if RUN_CAPSTONE:\n"
    "    stage('Capstone campaign', 'capstone_run.py',\n"
    "          produces=['data/capstone_run.json'])\n"
    "    print('\\nCapstone artifacts ready.')\n"
    "else:\n"
    "    print('RUN_CAPSTONE is False. The model-free chapters (taxonomy, design '\n"
    "          'space, enrichment, resilience) run without it; the analysis and '\n"
    "          'attribution chapters need data/capstone_run.json.')"
)

md("## 4. What's present")

code(
    "import os\n"
    "for f in ('capstone_run.json', 'capstone_rows.json', 'capstone_testset.json',\n"
    "          'capstone_companions.json'):\n"
    "    p = os.path.join(REPO, 'data', f)\n"
    "    print(f\"  {f:28} {'OK' if os.path.isfile(p) else 'missing'}\")"
)

md(
    "## Done\n"
    "\n"
    "Artifacts live under `code/data/` (git-ignored, never packaged). Re-run this\n"
    "notebook any time; existing artifacts are skipped."
)

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out = os.path.abspath(os.path.join(NBDIR, "00_setup.ipynb"))
with open(out, "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
print(f"wrote {out} ({len(cells)} cells)")
