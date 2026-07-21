#!/usr/bin/env python
"""Builder for notebook 12.3 Test enrichment (dependency-light).

Constructs the notebook with nbformat, executes it with ExecutePreprocessor,
and writes it to notebooks/12_3_test_enrichment.ipynb.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbformat import NO_CONVERT
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "12_3_test_enrichment.ipynb"

nb = new_notebook()
cells = []

# 1. Title + intro
cells.append(new_markdown_cell(
    "# 12.3 Test enrichment\n"
    "\n"
    "The test inputs are built by the base-then-enrich pipeline run once on this "
    "agent's domain. Twenty labeled complaint cases are the base; three presentation "
    "factors (clarity, entity_aliasing, reasoning_cue) vary *how* a complaint is "
    "presented without changing its correct handling. The product is the scenario "
    "suite the rest of the chapter runs on: one balanced run, scored many ways."
))

# 2. Reference API (not executed)
cells.append(new_markdown_cell(
    "**Reference (run separately):** the suite is composed by `gt.compose`, with the "
    "seed case entering as a balanced blocking factor and the three presentation "
    "factors space-filled by the Sobol-plus-refine generator. This is the heavy "
    "construction that produced the numbers below; it is shown for reference and is "
    "not executed here.\n"
    "\n"
    "```python\n"
    "import gmstest as gt\n"
    "from functools import partial\n"
    "from gmstest.compose import graphdoe_design\n"
    "\n"
    "cat   = gt.Catalog.load()\n"
    "items = gt.SeedCaseSource(cases).items()                  # 20 labeled cases\n"
    "suite = gt.resolve(cat, ['conditional_rule'],\n"
    "                   ['clarity', 'entity_aliasing', 'reasoning_cue'], mode='embedded')\n"
    "scns  = gt.compose(suite, items, n_runs=120, seed=42,\n"
    "                   design_fn=partial(graphdoe_design, method='sobol+refine'),\n"
    "                   balance_base=True, level_overrides=overrides)\n"
    "```"
))

# 3. Load artifacts (robust path)
cells.append(new_markdown_cell(
    "**What the next cell does** — load the pinned suite artifacts:\n"
    "\n"
    "1. **Find the repo root.** Walk up from `Path.cwd()` until `data/capstone_run.json` exists, so the notebook runs from any directory.\n"
    "2. **Read the JSON.** Load `data/capstone_run.json` into `D` and `data/capstone_companions.json` into `C` with `json.loads`."
))
cells.append(new_code_cell(
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "root = Path.cwd()\n"
    "while not (root / 'data' / 'capstone_run.json').exists() and root != root.parent:\n"
    "    root = root.parent\n"
    "D = json.loads((root / 'data' / 'capstone_run.json').read_text())\n"
    "C = json.loads((root / 'data' / 'capstone_companions.json').read_text())\n"
    "print('loaded capstone_run.json and capstone_companions.json')"
))

# 4. Print the numbers
cells.append(new_markdown_cell(
    "**What the next cell does** — report the suite's balance:\n"
    "\n"
    "1. **Pull the counts.** Read `D['n_runs']`, `D['seed_balance']` (runs per base case) and `D['clarity_balance']` (runs per clarity level).\n"
    "2. **Print the size and blocking.** Show `n_runs`, then the number of base cases in `seed_balance` with the distinct counts present and their min and max.\n"
    "3. **Print clarity coverage.** Loop over `clear`, `ambiguous`, `misleading`, print each `clarity_balance[level]` and the total."
))
cells.append(new_code_cell(
    "n_runs = D['n_runs']\n"
    "seed_balance = D['seed_balance']\n"
    "clarity_balance = D['clarity_balance']\n"
    "\n"
    "print(f'scenarios in the suite: {n_runs}')\n"
    "print()\n"
    "print('seed-case blocking (runs per base case):')\n"
    "counts = sorted(set(seed_balance.values()))\n"
    "print(f'  {len(seed_balance)} cases, counts present = {counts}  '\n"
    "      f'(min {min(seed_balance.values())}, max {max(seed_balance.values())})')\n"
    "print()\n"
    "print('clarity factor coverage:')\n"
    "for level in ('clear', 'ambiguous', 'misleading'):\n"
    "    print(f'  {level:11s} {clarity_balance[level]:3d}')\n"
    "print(f'  {\"total\":11s} {sum(clarity_balance.values()):3d}')"
))

# 5. Interpretation
cells.append(new_markdown_cell(
    "The result is 120 scenarios: the blocking holds the seed case exactly balanced "
    "(six runs each across all twenty cases), while the presentation factors are "
    "near-uniformly covered by the space-filling generator. Because every case is "
    "exercised equally and the factors are spread by design rather than hand-picked, "
    "the factor attribution later in the chapter reads a controlled design. This one "
    "balanced suite drives the outcome, trajectory, component and groundedness checks."
))

# 6. Self-check
cells.append(new_markdown_cell(
    "**What the next cell does** — assert the suite matches the design:\n"
    "\n"
    "1. **Check the size.** Assert `n_runs == 120`.\n"
    "2. **Check the blocking.** Assert `seed_balance` has 20 cases, every count is exactly 6, and the counts sum to `n_runs`.\n"
    "3. **Check the factor.** Assert `clarity_balance` covers `clear`, `ambiguous`, `misleading` and sums to `n_runs`."
))
cells.append(new_code_cell(
    "# the suite has the expected size\n"
    "assert n_runs == 120, n_runs\n"
    "\n"
    "# seed-case blocking is exactly balanced: every base case appears the same number of times\n"
    "assert len(seed_balance) == 20, len(seed_balance)\n"
    "assert set(seed_balance.values()) == {6}, seed_balance\n"
    "assert sum(seed_balance.values()) == n_runs\n"
    "\n"
    "# the presentation factor covers all three levels and sums to the suite size\n"
    "assert set(clarity_balance) == {'clear', 'ambiguous', 'misleading'}, clarity_balance\n"
    "assert sum(clarity_balance.values()) == n_runs, clarity_balance\n"
    "\n"
    "print('self-check passed')"
))

nb["cells"] = cells

# Execute
client = NotebookClient(nb, timeout=120, kernel_name="python3",
                        resources={"metadata": {"path": str(REPO / "notebooks")}})
client.execute()

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f, version=NO_CONVERT)
print("wrote", OUT)
