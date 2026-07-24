#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Builder for notebooks/00_setup.ipynb (CPU only; does NOT execute cells).

00_setup is the notebook that regenerates the trained GMS stores the chapter
notebooks load. Those stores are build artifacts — git-ignored, never shipped in
the package — so a fresh checkout has none of them and most chapters fail with
"failed to load GMS ... store".

It drives the existing build scripts through an idempotent stage runner: each
stage is skipped when its output already exists, so re-running is cheap and
safe. Stages are split by what they need:

  Tier 1 (CPU, no LLM)  retrain_gms_banking.py -> calibrate_gms_thresholds.py
                        build_policy_rag_store.py, build_regulatory_guard_store.py
  Tier 2 (GPU + Qwen)   build_geode_rag_store.py (the GEODE policy-RAG store)

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
    "# 00 — Set up the GMS stores\n"
    "\n"
    "The chapter notebooks load **trained GMS stores** from `code/data/`\n"
    "(`gms_banking_store`, `gms_policy_store`, `gms_regulatory_store`, and the\n"
    "GEODE policy-RAG store). Those are build artifacts: **not committed and not\n"
    "shipped in the package** (git-ignored, like all trained weights). A fresh\n"
    "checkout has none of them, so most chapters fail with *\"failed to load GMS\n"
    "... store\"* until this notebook has been run.\n"
    "\n"
    "**What you need**\n"
    "\n"
    "- The licensed **`knowlytix`** substrate: `pip install knowlytix` plus a\n"
    "  developer license from https://knowlytix.ai/signup/ (key at\n"
    "  `~/.knowlytix/license.key`). Without it no store can be built.\n"
    "- `torch`.\n"
    "\n"
    "**Two tiers**\n"
    "\n"
    "1. **Tier 1 — CPU, no LLM.** The banking substrate store, its calibrated\n"
    "   gate thresholds, the policy entity index and the regulatory guard. This\n"
    "   is what most chapters (and the capstone tests) need. A few minutes.\n"
    "2. **Tier 2 — GPU + Qwen.** The GEODE self-corrected policy-RAG store, used\n"
    "   by the retrieval/answering chapters. Downloads Qwen2.5-3B-Instruct and\n"
    "   expects a CUDA GPU.\n"
    "\n"
    "Every stage is **idempotent** — skipped when its output already exists — so\n"
    "you can re-run this notebook freely."
)

md("## 1. Bootstrap: locate `knowlytix` and this topic's code/ dir")

code(
    "import importlib.util, os, sys\n"
    "\n"
    "# Locate this topic's code/ dir, robust to the working directory.\n"
    "_cwd = os.getcwd()\n"
    'for REPO in (_cwd, os.path.join(os.path.dirname(_cwd), "code"), os.path.join(_cwd, "code")):\n'
    '    if os.path.isdir(os.path.join(REPO, "agentlab")):\n'
    "        break\n"
    "else:\n"
    '    REPO = os.environ.get("AGENTLAB_REPO", _cwd)\n'
    "for p in (REPO, os.path.join(REPO, 'scripts')):\n"
    "    if p not in sys.path:\n"
    "        sys.path.insert(0, p)\n"
    "\n"
    "if importlib.util.find_spec('knowlytix') is None:\n"
    "    raise ModuleNotFoundError(\n"
    "        'knowlytix not found. Install it (`pip install knowlytix`, licensed) '\n"
    "        'and put your key at ~/.knowlytix/license.key — see '\n"
    "        'https://knowlytix.ai/signup/.')\n"
    "\n"
    "import torch\n"
    "print('torch:', torch.__version__, '| device:',\n"
    "      'cuda' if torch.cuda.is_available() else 'cpu')\n"
    "print('repo :', REPO)"
)

md("## 2. Idempotent stage runner")

code(
    "import subprocess, sys, os\n"
    "\n"
    "def stage(title, script, args=(), produces=()):\n"
    "    \"\"\"Run scripts/<script> unless every path in `produces` already exists.\n"
    "    Streams the script's output; raises on non-zero exit.\"\"\"\n"
    "    produces = list(produces)\n"
    "    if produces and all(os.path.exists(os.path.join(REPO, p)) for p in produces):\n"
    "        print(f'\\u2713 {title}: already built \\u2014 skipping')\n"
    "        return\n"
    "    cmd = [sys.executable, os.path.join(REPO, 'scripts', script), *map(str, args)]\n"
    "    print(f'\\u25b6 {title}: python scripts/{script} ' + ' '.join(map(str, args)))\n"
    "    r = subprocess.run(cmd, cwd=REPO, env=os.environ.copy())\n"
    "    if r.returncode:\n"
    "        raise RuntimeError(f'{script} failed (exit {r.returncode})')\n"
    "    print(f'\\u2713 {title}: done')"
)

md(
    "## 3. Tier 1 — CPU stores (required)\n"
    "\n"
    "No LLM and no GPU. The banking store is the Chapter-16 substrate (ENM,\n"
    "tension, plausibility gate); calibrating it writes the gate thresholds the\n"
    "capstone reads. The policy and regulatory stores back `search_policy` and\n"
    "`flag_regulatory`."
)

code(
    "stage('Banking substrate store', 'retrain_gms_banking.py',\n"
    "      produces=['data/gms_banking_store/model.pt'])\n"
    "\n"
    "stage('Calibrate plausibility/contradiction thresholds', 'calibrate_gms_thresholds.py',\n"
    "      produces=['data/gms_banking_store/calibration.json'])\n"
    "\n"
    "stage('Policy entity index (Graph-RAG)', 'build_policy_rag_store.py',\n"
    "      produces=['data/gms_policy_store/model.pt'])\n"
    "\n"
    "stage('Regulatory guard store', 'build_regulatory_guard_store.py',\n"
    "      produces=['data/gms_regulatory_store/model.pt'])"
)

md("## 4. Verify the Tier-1 stores load")

code(
    "import json, os\n"
    "\n"
    "for name in ('gms_banking_store', 'gms_policy_store', 'gms_regulatory_store'):\n"
    "    p = os.path.join(REPO, 'data', name)\n"
    "    print(f\"{name:24} {'OK ' if os.path.isdir(p) else 'MISSING'}\")\n"
    "\n"
    "cal = os.path.join(REPO, 'data', 'gms_banking_store', 'calibration.json')\n"
    "if os.path.isfile(cal):\n"
    "    theta = json.load(open(cal))['plausibility_gate']['threshold']\n"
    "    print('calibrated plausibility threshold:', theta)"
)

md(
    "## 5. Tier 2 — GEODE policy-RAG store (optional; GPU + Qwen)\n"
    "\n"
    "The retrieval/answering chapters bind queries through a GEODE\n"
    "self-corrected policy graph. Building it runs Qwen2.5-3B-Instruct and\n"
    "**expects a CUDA GPU**; it also downloads the model (~6 GB). Set\n"
    "`RUN_TIER2 = True` to build it."
)

code(
    "RUN_TIER2 = False  # set True to build the GEODE policy-RAG store (GPU + Qwen)\n"
    "\n"
    "if RUN_TIER2:\n"
    "    stage('GEODE policy-RAG store (loads Qwen)', 'build_geode_rag_store.py',\n"
    "          produces=['data/gms_policy_store_geode/model.pt'])\n"
    "    print('\\nTier 2 complete.')\n"
    "else:\n"
    "    print('RUN_TIER2 is False \\u2014 Tier-1 stores only. The retrieval/answering '\n"
    "          'chapters and the end-to-end capstone tests need Tier 2.')"
)

md(
    "## Done\n"
    "\n"
    "The stores live under `code/data/` (git-ignored, never packaged). Open the\n"
    "chapter notebooks — they load these directly. Re-run this notebook any time;\n"
    "existing stores are skipped."
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
