"""Builder for notebook 12_8_resilience.ipynb (Chapter 12, section 12.8).

Dependency-light: the executable cells only load pinned artifacts and run
assert-based self-checks. The real fault-injection API is shown as reference
markdown only.
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbformat import write
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "12_8_resilience.ipynb"

cells = []

# 1. Title + intro
cells.append(new_markdown_cell(
    "# 12.8 Resilience: fault injection and the detection rate\n"
    "\n"
    "Component testing asks whether each tool is correct; the resilience probe "
    "asks what happens when a tool fails. A gateway is installed on the agent's "
    "executor and one tool at a time is made to error; the measured quantity is "
    "the **detection rate** -- the fraction of runs in which the agent escalates "
    "rather than silently proceeding on a corrupted result. This notebook reads "
    "the pinned resilience numbers and checks them."
))

# 2. Reference API (markdown, not executed)
cells.append(new_markdown_cell(
    "**Reference (run separately):** the real fault-injection probe installs a "
    "`ToolGateway` on the capstone agent's executor, errors one tool per run, "
    "and records whether the agent escalated (detected) or proceeded (silent).\n"
    "\n"
    "```python\n"
    "from agentlab.harness.capstone import CapstoneTestHarness\n"
    "\n"
    "harness = CapstoneTestHarness(agent=capstone_agent, cases=eval_cases)\n"
    "\n"
    "# fault_injection() installs a ToolGateway on the executor that forces one\n"
    "# tool at a time to raise; the agent's run is observed for escalation.\n"
    "report = harness.fault_injection(fault='error')\n"
    "\n"
    "print(report.detection_rate)      # fraction of faulted runs that escalate\n"
    "print(report.silent_failures)     # runs that proceeded on a corrupted result\n"
    "print(report.per_tool_detection)  # detection rate broken out per tool\n"
    "```\n"
    "\n"
    "This call is GPU-heavy and not run here; its outputs are pinned in "
    "`capstone_companions.json`."
))

cells.append(new_markdown_cell(
    "## What the probe injects, and what counts as detection\n"
    "\n"
    "The probe targets one tool at a time. For each of the five tools it installs a "
    "`ToolGateway` carrying a single `FaultProfile` matched to that tool, runs the "
    "scenarios, then uninstalls it and moves on, so the other four tools behave normally "
    "throughout. The fault itself is one of three kinds:\n"
    "\n"
    "- **hard error** (the default, and the one reported here) -- `error_rate=1.0`, so the "
    "gateway blocks every call to the tool and it fails whenever it is reached;\n"
    "- **latency** -- the call is delayed by a fixed budget, then allowed;\n"
    "- **stale** -- the call is let through but its result is replaced by an out-of-date "
    "response, the case a syntactic check cannot catch.\n"
    "\n"
    "```python\n"
    "from knowlytix.harness.testing import FaultProfile, ToolGateway\n"
    "# the default fault: force one tool to fail on every call it receives\n"
    "profile = FaultProfile(tool_pattern='search_policy', error_rate=1.0,\n"
    "                       error_message='search_policy service unavailable (injected)')\n"
    "```\n"
    "\n"
    "A faulted run counts as **detected** only when the agent both *reached* the faulted "
    "tool and then *failed loud*, ending escalated or failed rather than finishing on a "
    "corrupted result. A run that finishes normally after hitting the broken tool is a "
    "**silent** failure, the outcome a governed agent must never produce. The detection "
    "rate is measured over the runs that reached the faulted tool, so a run that escalated "
    "earlier never counts against that tool."
))

# 3. Load artifact (robust path)
cells.append(new_markdown_cell(
    "**What the next cell does** — loads the pinned resilience block from disk:\n"
    "\n"
    "1. **Find the data root.** Walk up from `Path.cwd()` until a directory "
    "containing `data/capstone_companions.json` is found.\n"
    "2. **Read the artifact.** `json.loads(...)` parses that file into `C`, then "
    "`res = C['resilience']` pulls the resilience sub-dict (`fault`, "
    "`detection_rate`, `silent_failures`, `per_tool_detection`).\n"
    "3. **Echo it.** The bare `res` displays the dict so the raw numbers are visible."
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
    "res = C['resilience']\n"
    "res"
))

# 4. Print the numbers
cells.append(new_markdown_cell(
    "**What the next cell does** — prints the resilience numbers in a readable "
    "layout:\n"
    "\n"
    "1. **Headline figures.** Prints `res['fault']`, `res['detection_rate']` and "
    "`res['silent_failures']`.\n"
    "2. **Per-tool breakdown.** Loops over `res['per_tool_detection'].items()` and "
    "prints the detection rate for each of the five tools."
))
cells.append(new_code_cell(
    "print(f\"fault injected        : {res['fault']}\")\n"
    "print(f\"detection rate (all)  : {res['detection_rate']:.2f}\")\n"
    "print(f\"silent failures       : {res['silent_failures']}\")\n"
    "print()\n"
    "print('per-tool detection rate:')\n"
    "for tool, rate in res['per_tool_detection'].items():\n"
    "    print(f\"  {tool:18s} {rate:.2f}\")"
))

# 5. Interpretation
cells.append(new_markdown_cell(
    "On this agent the detection rate is **1.0 on every tool** and the "
    "silent-failure count is **zero**: a hard fault on any step makes the agent "
    "fail loud and escalate to a human. The gateway establishes that an outright "
    "failure is caught; the verifiers of the earlier chapters -- the exact-register "
    "check and the groundedness distance -- are what catch the harder case of a "
    "plausible-but-wrong structured result."
))

# 5b. The hardest fault: a stance reversal (value-polarity)
cells.append(new_markdown_cell(
    "## The hardest fault: a stance reversal\n"
    "\n"
    "The purest plausible-but-wrong result is a draft that **inverts** a policy's "
    "stance -- naming the right entity and relation but asserting *optional* where the "
    "policy makes it required, or *permitted* where it is forbidden. The type and shape "
    "are correct, so a syntactic check passes; only a comparison against the stored "
    "stance catches it. The value-polarity verifier makes that comparison and returns "
    "`supported`, `contradicted` or `uncertain`."
))
cells.append(new_markdown_cell(
    "**What the next cell does** — reads the pinned value-polarity result "
    "(`C['value_polarity']`) and prints the reversal-detection and synonym-acceptance "
    "rates."
))
cells.append(new_code_cell(
    "vp = C['value_polarity']\n"
    "print(f\"stance reversals flagged contradicted : {vp['reversal_detected']}/{vp['reversal_n']}\"\n"
    "      f\"  (rate {vp['reversal_rate']:.2f})\")\n"
    "print(f\"synonym stances accepted as supported : {vp['synonym_accepted']}/{vp['synonym_n']}\"\n"
    "      f\"  (rate {vp['synonym_rate']:.2f})\")"
))
cells.append(new_markdown_cell(
    "On a four-item reversal battery the verifier flags three of the four reversals as "
    "contradicted; the miss is a required-to-optional assertion scored `uncertain`, a "
    "deferral to review rather than a false pass. On the matching synonym battery, where "
    "the asserted stance paraphrases the stored one, it accepts three of four. The stance "
    "reversal is the plausible-but-wrong fault in its purest form, and it is caught "
    "downstream, not at the tool boundary."
))

# 6. Self-check
cells.append(new_markdown_cell(
    "**What the next cell does** — asserts the resilience contract holds:\n"
    "\n"
    "1. **Headline checks.** Asserts `res['fault'] == 'error'`, "
    "`res['detection_rate'] == 1.0` and `res['silent_failures'] == 0`.\n"
    "2. **Coverage check.** Asserts the keys of `res['per_tool_detection']` are "
    "exactly the five capstone tools.\n"
    "3. **Per-tool check.** Asserts every value in `res['per_tool_detection']` is "
    "`1.0`, then prints `self-check passed`."
))
cells.append(new_code_cell(
    "assert res['fault'] == 'error'\n"
    "assert res['detection_rate'] == 1.0, 'agent must escalate on every faulted run'\n"
    "assert res['silent_failures'] == 0, 'no fault may propagate silently'\n"
    "assert set(res['per_tool_detection']) == {\n"
    "    'classify_complaint', 'extract_facts', 'search_policy',\n"
    "    'flag_regulatory', 'draft_response',\n"
    "}\n"
    "assert all(r == 1.0 for r in res['per_tool_detection'].values()), \\\n"
    "    'every tool fault must be detected'\n"
    "vp = C['value_polarity']\n"
    "assert vp['reversal_detected'] == 3 and vp['reversal_n'] == 4   # 3/4 reversals caught\n"
    "assert vp['synonym_accepted'] == 3 and vp['synonym_n'] == 4     # 3/4 synonyms accepted\n"
    "print('self-check passed')"
))

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": str(REPO / "notebooks")}})

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    write(nb, f)
print(f"wrote {OUT}")
