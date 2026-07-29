# SPDX-License-Identifier: Apache-2.0
"""Builder for notebooks/00_setup.ipynb (CPU only; does NOT execute cells).

00_setup is the ONE notebook that regenerates the trained GMS store data every
other chapter loads. The store artifacts are never committed to the repo or
shipped in the package (they are git-ignored, licensed-substrate outputs), so a
fresh checkout runs this first. It drives the existing build scripts
(build_store.py -> enrich_data.py -> finetune_encoders.py -> calibrate_*.py),
skipping any stage whose output already exists.

Run: python scripts/build_nb_00_setup.py
Emits a valid nbformat-4 notebook. No store/Qwen is loaded here.
"""
from __future__ import annotations

import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
# notebooks/ is a sibling of code/ (this file lives in code/scripts/).
NBDIR = os.path.join(HERE, "..", "..", "notebooks")

nb = nbf.v4.new_notebook()
cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src))


md(
    "# 00 — Set up the store data\n"
    "\n"
    "Every other notebook in this topic loads a **trained GMS store** from\n"
    "`data/gms_annual_report_store/`. That store is a build artifact: it is\n"
    "**not committed to the repo and not shipped in the package** (it is\n"
    "git-ignored, like all trained weights and GMS stores). This notebook is the\n"
    "one place that regenerates it, from the committed corpus\n"
    "(`data/annual_report.md`) using the scripts in `code/scripts/`.\n"
    "\n"
    "**What you need**\n"
    "\n"
    "- The licensed **`knowlytix`** substrate: `pip install knowlytix` and a\n"
    "  developer license from https://knowlytix.ai/signup/ (key at\n"
    "  `~/.knowlytix/license.key`). If you use a source checkout instead, set\n"
    "  `KNOWLYTIX_SRC` to it. Without the substrate, the stores cannot be built.\n"
    "- `torch`. **Tier 1** — the base store below — is CPU-only (the corpus is\n"
    "  small), no GPU needed. **Tier 2** — the optional Qwen stages (enrichment,\n"
    "  encoder fine-tuning, gate calibration) — expects a **CUDA GPU** and\n"
    "  downloads Qwen3-4B-Instruct.\n"
    "\n"
    "**Two tiers**\n"
    "\n"
    "1. **Base store** (required by most chapters) — the GEODE-corrected graph +\n"
    "   cap geometry + ENM. Fast; no LLM.\n"
    "2. **Full pipeline** (optional; needed only by the advanced chapters —\n"
    "   enrichment, fine-tuned encoders, calibrated gates). Heavier: loads Qwen.\n"
    "\n"
    "Each stage is **idempotent** — it is skipped when its output already exists,\n"
    "so you can re-run this notebook freely."
)

md("## 1. Bootstrap: locate `knowlytix` and this repo")

code(
    "# Self-contained bootstrap (mirrors book_kit.resolve_knowlytix so this cell\n"
    "# works before book_kit — which imports knowlytix — can be imported).\n"
    "import importlib.util, os, sys\n"
    "\n"
    "REPO = os.environ.get('GMS_RAG_TUTORIAL')\n"
    "if not REPO:\n"
    "    # notebooks/ is a sibling of code/; resolve code/ from the cwd.\n"
    "    cwd = os.getcwd()\n"
    "    REPO = os.path.join(os.path.dirname(cwd), 'code') if os.path.basename(cwd) == 'notebooks' else cwd\n"
    "os.environ['GMS_RAG_TUTORIAL'] = REPO\n"
    "for p in (REPO, os.path.join(REPO, 'scripts')):\n"
    "    if p not in sys.path:\n"
    "        sys.path.insert(0, p)\n"
    "\n"
    "def resolve_knowlytix():\n"
    "    if importlib.util.find_spec('knowlytix') is not None:\n"
    "        return 'already importable'\n"
    "    for c in [os.environ.get('KNOWLYTIX_SRC'),\n"
    "              os.path.expanduser('~/source/GMS-knowlytix'),\n"
    "              os.path.expanduser('~/GMS-knowlytix'),\n"
    "              os.path.normpath(os.path.join(REPO, '..', '..', 'GMS-knowlytix'))]:\n"
    "        if c and os.path.isdir(os.path.join(c, 'knowlytix')):\n"
    "            sys.path.insert(0, c); os.environ['KNOWLYTIX_SRC'] = c\n"
    "            return c\n"
    "    raise ModuleNotFoundError(\n"
    "        'knowlytix not found. Install it (`pip install knowlytix`, licensed) or, '\n"
    "        \"for a source checkout, set os.environ['KNOWLYTIX_SRC']='/path/to/GMS-knowlytix', \"\n"
    "        'then restart the kernel.')\n"
    "\n"
    "print('knowlytix:', resolve_knowlytix())\n"
    "import torch\n"
    "print('torch:', torch.__version__, '| device:',\n"
    "      'cuda' if torch.cuda.is_available() else 'cpu')\n"
    "print('repo:', REPO)"
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
    "## 3. Stage 1 — base store (required)\n"
    "\n"
    "GEODE self-corrects `data/annual_report.md` into a trained store and writes\n"
    "the ground-truth facts sheet. Skipped if the store already exists."
)

code(
    "stage('Build base store', 'build_store.py',\n"
    "      produces=['data/gms_annual_report_store/model.pt',\n"
    "                'data/corpus_facts.md'])"
)

md("## 4. Verify the base store loads")

code(
    "from book_kit import load_store\n"
    "\n"
    "store = load_store()\n"
    "print('entities:', store.adapter.num_entities,\n"
    "      '| relations:', store.adapter.num_relations)"
)

md(
    "## 5. Stage 2–4 — full pipeline (optional; loads Qwen on GPU)\n"
    "\n"
    "The advanced chapters (calibration, pluggable LLMs, the capstone accept\n"
    "gate) also need the DoE-enriched training data, the fine-tuned v/u encoders,\n"
    "and the calibrated gates. These are **heavier** — enrichment and the accept\n"
    "gate load Qwen — so they are opt-in. Set `RUN_FULL = True` to build them.\n"
    "Each stage is idempotent."
)

code(
    "RUN_FULL = False  # set True to build enrichment + tuned encoders + gates (needs Qwen/GPU)\n"
    "\n"
    "if RUN_FULL:\n"
    "    stage('Enrich (DoE over the store; loads Qwen)', 'enrich_data.py',\n"
    "          args=['--per-seed', 6],\n"
    "          produces=['data/enrichment/embedding_sft.jsonl'])\n"
    "    stage('Fine-tune encoders + relevance gate', 'finetune_encoders.py',\n"
    "          produces=['data/gms_annual_report_store/tuned_encoder'])\n"
    "    stage('Calibrate groundedness + relevance gates', 'calibrate_gates.py',\n"
    "          produces=['data/gms_annual_report_store/calibration.json'])\n"
    "    stage('Calibrate accept/abstain gate (loads Qwen)', 'calibrate_accept_gate.py',\n"
    "          produces=['data/gms_annual_report_store/rag_gate_calibration.json'])\n"
    "    print('\\nfull pipeline complete.')\n"
    "else:\n"
    "    print('RUN_FULL is False \\u2014 base store only. '\n"
    "          'Set it True for the advanced chapters.')"
)

md(
    "## Done\n"
    "\n"
    "The store data now lives under `data/` (git-ignored, never packaged). Open\n"
    "the chapter notebooks (`01_`, `02_`, …) — they load it through\n"
    "`book_kit.load_store()`. Re-run this notebook any time to rebuild; existing\n"
    "artifacts are skipped."
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
