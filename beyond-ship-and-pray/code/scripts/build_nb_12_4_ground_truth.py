#!/usr/bin/env python
"""Builder for notebook 12_4_ground_truth.ipynb (Chapter 12, section 12.4).

Dependency-light: the executable cells only load the pinned companion artifact,
print the precondition baseline, and run assert-based self-checks. The real
GEODE-store build + precondition is shown as reference only (not executed).
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbformat import write
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[2] / "notebooks" / "12_4_ground_truth.ipynb"

cells = []

# 1. Title + intro (adapted from sec:capstone-groundtruth opening prose)
cells.append(new_markdown_cell(
    "# 12.4 Generating the Ground Truth\n"
    "\n"
    "The enriched scenarios are inputs; scoring them needs an answer key whose own "
    "correctness is not in question. That key is *generated*, not annotated -- read "
    "from the same geometric memory substrate (the GEODE store) that the agent runs "
    "on. The same store is both what the agent consults and what the test scores it "
    "against, which is how the ground truth stays provably correct rather than a "
    "second opinion.\n"
    "\n"
    "Before any score built on this key is trusted, there is a **precondition**: the "
    "graph must answer its own generated questions perfectly -- a baseline of `1.0`. "
    "A perfect baseline confirms the ground truth is sound; only then is a failure on "
    "the real run chargeable to the agent rather than to a noisy key."
))

# 2. Reference API (do not execute)
cells.append(new_markdown_cell(
    "**Reference (run separately):** the GEODE self-correction loop ingests the "
    "policy, proposes facts, diagnoses the ones the geometry finds implausible or "
    "contradictory, repairs them, then trains and calibrates the result. The build "
    "below produces the store the answer key is read from; the precondition is that "
    "the graph then answers its own generated questions at a baseline of `1.0`.\n"
    "\n"
    "```python\n"
    "# scripts/build_geode_rag_store.py -> data/gms_policy_store_geode\n"
    "from knowlytix.geode import GeodeBuilder\n"
    "\n"
    "builder = GeodeBuilder.from_policy('data/banking_policy_full.md')\n"
    "store = builder.ingest().diagnose().repair().train().calibrate()\n"
    "store.save('data/gms_policy_store_geode')   # the GEODE oracle\n"
    "\n"
    "# The precondition: the graph answers its own generated questions perfectly.\n"
    "from knowlytix.harness.testing import CapstoneTestHarness\n"
    "h = CapstoneTestHarness(store='data/gms_policy_store_geode')\n"
    "baseline = h.substrate_test().baseline      # must be 1.0 before any score is trusted\n"
    "assert baseline == 1.0, 'ground truth is not sound; do not score the agent'\n"
    "```"
))

# 3. Load artifact (robust path)
cells.append(new_markdown_cell(
    "**What the next cell does** — finds the project root and loads the pinned companion artifact:\n"
    "\n"
    "1. **Walk up to the root.** Starts at `Path.cwd()` and climbs parent directories until `data/capstone_companions.json` exists, so the cell runs from anywhere in the tree.\n"
    "2. **Load the companions.** Reads `capstone_companions.json` into `C` and pulls `C['substrate']` into `substrate`, the block holding the ground-truth precondition numbers."
))
cells.append(new_code_cell(
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "root = Path.cwd()\n"
    "while not (root / 'data' / 'capstone_companions.json').exists() and root != root.parent:\n"
    "    root = root.parent\n"
    "\n"
    "C = json.loads((root / 'data' / 'capstone_companions.json').read_text())\n"
    "substrate = C['substrate']\n"
    "print('loaded capstone_companions.json from', root / 'data')"
))

# 4. Print the numbers
cells.append(new_markdown_cell(
    "**What the next cell does** — prints the ground-truth precondition and renders a verdict:\n"
    "\n"
    "1. **Read the numbers.** Pulls `substrate['baseline']` and `substrate['n_questions']` into `baseline` and `n_questions`.\n"
    "2. **Print the baseline table.** Reports how many questions were reversed from the policy graph and the precondition baseline, formatted to two decimals.\n"
    "3. **Render the verdict.** Sets `verdict` to `SOUND` when `baseline == 1.0` and `NOISY KEY` otherwise, then prints it."
))
cells.append(new_code_cell(
    "baseline = substrate['baseline']\n"
    "n_questions = substrate['n_questions']\n"
    "\n"
    "print('Ground-truth precondition (graph answers its own questions)')\n"
    "print('-' * 56)\n"
    "print(f'{\"questions reversed from the policy graph\":42s} {n_questions}')\n"
    "print(f'{\"precondition baseline (must equal 1.0)\":42s} {baseline:.2f}')\n"
    "print()\n"
    "verdict = 'SOUND -> agent failures are chargeable' if baseline == 1.0 else 'NOISY KEY -> do not score'\n"
    "print(f'ground truth: {verdict}')"
))

# 5. Interpretation (adapted from the .tex)
cells.append(new_markdown_cell(
    "A perfect baseline of `1.0` confirms the answer key is sound: the governing "
    "policy and its accept-set are graph-connected, the regulation's severity is a "
    "graph traversal, the authoritative fee is the exact-memory value, and the "
    "substrate questions are reversed wholesale out of the policy graph with provable "
    "answers. The seed cases' classification labels are the one authored input; "
    "everything fact-grounded is read from the graph.\n"
    "\n"
    "This is the condition that makes the capstone a *designed experiment* rather than "
    "a benchmark: only once the key answers its own questions perfectly is a failure "
    "on the real run attributable to the agent rather than to a noisy ground truth."
))

# 6. Self-check
cells.append(new_markdown_cell(
    "**What the next cell does** — asserts the answer key is sound before any scoring is trusted:\n"
    "\n"
    "1. **Check the block exists.** Asserts `'substrate'` is present in `C`.\n"
    "2. **Check the baseline.** Asserts `substrate['baseline']` is exactly `1.0`.\n"
    "3. **Check the question count.** Asserts `n_questions > 0`, then prints `self-check passed`."
))
cells.append(new_code_cell(
    "assert 'substrate' in C, 'companions artifact must carry the substrate block'\n"
    "assert substrate['baseline'] == 1.0, 'precondition baseline must be exactly 1.0'\n"
    "assert n_questions > 0, 'questions must be generated from the graph'\n"
    "print('self-check passed')"
))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
})

ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": str(NB_PATH.parent)}})

with open(NB_PATH, "w") as f:
    write(nb, f)
print("wrote", NB_PATH)
