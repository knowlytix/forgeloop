"""Builder for notebook 12.7 Operational testing: trajectory and process health.

Dependency-light: the executable cells only import json/pathlib, load the pinned
artifact, print the numbers, and run assert-based self-checks. The real evaluate.run
/ evaluate.summary API is shown as reference only (markdown fenced block).
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbformat import NO_CONVERT
from nbclient import NotebookClient
from pathlib import Path

NB_PATH = "/path/to/forgeloop/beyond-ship-and-pray/notebooks/12_7_operational.ipynb"

cells = []

cells.append(new_markdown_cell(
    "# 12.7 Operational testing: trajectory and process health\n\n"
    "Component and system testing ask whether the agent is right; operational testing "
    "asks whether it ran the way a governed workflow must -- in order, terminating cleanly "
    "and leaving a verifiable record. Trajectory structure reads the step record (did the "
    "five-step workflow run in the prescribed order, and on escalation did it take the path "
    "its category implies); process health counts steps, tool calls and failures and whether "
    "the run terminated within budget."
))

cells.append(new_markdown_cell(
    "**Reference (run separately):** the operational metrics below were produced by running "
    "the scenario campaign against the system under test with the prescribed workflow order, "
    "then summarizing the rows.\n\n"
    "```python\n"
    "# Reference (run separately): heavy path that produced the numbers.\n"
    "rows = evaluate.run(scns, sut, workflow_order=workflow)\n"
    "print(evaluate.summary(rows))\n"
    "```"
))

cells.append(new_markdown_cell(
    "**What the next cell does** -- loads the pinned operational artifact:\n\n"
    "1. **Locate the file.** Start at `Path.cwd()` and walk up parent directories until "
    "`data/capstone_run.json` is found, so the cell runs from `notebooks/` or the repo root.\n"
    "2. **Load it.** Read the file and `json.loads` it into the dict `D`, then print the resolved path."
))

cells.append(new_code_cell(
    "import json\n"
    "from pathlib import Path\n\n"
    "# Resolve the artifact whether we run from notebooks/ or the repo root.\n"
    "root = Path.cwd()\n"
    "while not (root / 'data' / 'capstone_run.json').exists() and root != root.parent:\n"
    "    root = root.parent\n"
    "D = json.loads((root / 'data' / 'capstone_run.json').read_text())\n"
    "print('loaded capstone_run.json from', root / 'data' / 'capstone_run.json')"
))

cells.append(new_markdown_cell(
    "**What the next cell does** -- reads the campaign-level operational signals:\n\n"
    "1. **Pull the metrics.** Read `D['n_runs']` into `n_runs`, `D['workflow_adherence']` "
    "into `adherence` and `D['audit_verifies']` into `audit`.\n"
    "2. **Report them.** Print the scenario count, the workflow-adherence fraction and the "
    "audit-verifies flag -- trajectory and process-health signals, not bare outcome accuracy."
))

cells.append(new_code_cell(
    "# Operational health across the campaign. We report trajectory and process-health\n"
    "# signals, NOT the bare outcome accuracy (which would conflate a clean escalation\n"
    "# with a run that terminated without an answer).\n"
    "n_runs = D['n_runs']\n"
    "adherence = D['workflow_adherence']\n"
    "audit = D['audit_verifies']\n\n"
    "print(f'Scenarios            : {n_runs}')\n"
    "print(f'Workflow adherence   : {adherence:.3f}')\n"
    "print(f'Audit verifies       : {audit}')"
))

cells.append(new_markdown_cell(
    "Workflow adherence is perfect: the plausibility gate held the five-step sequence in "
    "order on every phrasing, so no failure is the agent firing tools out of turn. The audit "
    "log verifies across the whole campaign, so any failure surfaced elsewhere is behavioral, "
    "not an integrity failure of the record. Reducing a run to one bit would conflate a clean "
    "escalation at the first gate with a workflow that terminated without an answer -- reading "
    "the trajectory and process columns keeps them distinct."
))

cells.append(new_markdown_cell(
    "**What the next cell does** -- asserts the pinned operational metrics are intact:\n\n"
    "1. **Keys present.** Assert `D` contains `workflow_adherence` and `audit_verifies`.\n"
    "2. **Values hold.** Assert `D['workflow_adherence'] == 1.0`, `D['audit_verifies'] is True` "
    "and `D['n_runs'] == 120`, then print that the self-check passed."
))

cells.append(new_code_cell(
    "# Self-check on the pinned operational metrics.\n"
    "assert 'workflow_adherence' in D and 'audit_verifies' in D\n"
    "assert D['workflow_adherence'] == 1.0, 'workflow adherence must be perfect'\n"
    "assert D['audit_verifies'] is True, 'audit log must verify'\n"
    "assert D['n_runs'] == 120\n"
    "print('self-check passed')"
))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})

client = NotebookClient(nb, timeout=120, kernel_name="python3",
                        resources={"metadata": {"path": str(Path(NB_PATH).parent)}})
client.execute()

# Verify no cell errors.
errs = []
for i, c in enumerate(nb.cells):
    if c.get("cell_type") == "code":
        for out in c.get("outputs", []):
            if out.get("output_type") == "error":
                errs.append((i, out.get("ename"), out.get("evalue")))
if errs:
    raise SystemExit(f"cell errors: {errs}")

nbformat.write(nb, NB_PATH, version=NO_CONVERT)
print("wrote", NB_PATH, "cells", len(nb.cells))
