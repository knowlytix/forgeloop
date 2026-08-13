#!/usr/bin/env python
"""Builder for notebooks/08_capstone_companion.ipynb.

Capstone companion to Chapter 8 (Planning, Decomposition and Replanning): the
concept read on the running banking complaint agent. Teaching notebook in the
style of the main chapter notebooks -- real capstone imports, no pre-embedded
outputs (the reader runs it). Structural demos are cheap and run offline; the
store-backed plausibility gate is shown as reader-runnable (it loads the GMS
banking store on GPU/CPU).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "08_capstone_companion.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone companion --- Chapter 8: Planning, Decomposition and Replanning\n"
        "\n"
        "Chapter~8 treats a plan as a work list: an ordered decomposition of a task "
        "into steps, subject to replanning when a step fails. The chapter distinguishes "
        "a free planner, which selects the next step at run time, from a fixed workflow, "
        "whose step order is decided in advance and enforced structurally. The capstone "
        "banking complaint agent takes the second position. Its plan is a fixed five-step "
        "workflow --- classify, extract, search policy, flag regulatory, draft --- and "
        "the ordering is not left to the model to rediscover on each call. It is encoded "
        "as a directed acyclic graph in the GMS banking store and enforced by a "
        "plausibility gate that rejects any transition the graph does not admit."
    ),
    new_markdown_cell(
        "This companion reads that principle on the real capstone code. The workflow, the "
        "node order and the enabling relation are inspected offline; the store-backed gate "
        "is shown as reader-runnable, because scoring a transition loads the trained GMS "
        "store. The distinction the chapter draws --- a plan that is a work list versus a "
        "planner that improvises --- appears here as a fixed node sequence checked against "
        "a `has_enables` DAG."
    ),
    new_markdown_cell(
        "## The plan as a fixed work list\n"
        "\n"
        "The agent's plan is the module-level constant `_WORKFLOW_NODES`, an ordered list "
        "of the five workflow nodes. Two tool names differ from their node names, so the "
        "map `_TOOL_NODE_MAP` records the correspondence: the tools `classify_complaint` "
        "and `extract_facts` occupy the nodes `classify` and `extract`. Reading these two "
        "constants gives the entire plan; there is no run-time step selection to inspect."
    ),
    new_code_cell(
        "from agentlab.capstone.complaint_agent import _WORKFLOW_NODES, _TOOL_NODE_MAP\n"
        "\n"
        "print('workflow (in execution order):')\n"
        "for i, node in enumerate(_WORKFLOW_NODES):\n"
        "    print(f'  step {i}: {node}')\n"
        "print()\n"
        "print('tool -> node map:', _TOOL_NODE_MAP)"
    ),
    new_markdown_cell(
        "## The plan encoded in the agent body\n"
        "\n"
        "The fixed order is realized in `ComplaintAgent.propose_action`: the method "
        "branches on `state.step` and emits a specific `ToolCall` for each step, rather "
        "than asking the model which tool to call next. Step~0 classifies, step~1 "
        "extracts, step~2 searches policy, step~3 flags regulatory risk and step~4 drafts "
        "the reply. The source below shows the branch structure; the plan is literally the "
        "sequence of `if state.step == k` clauses."
    ),
    new_code_cell(
        "import inspect\n"
        "from agentlab.capstone.complaint_agent import ComplaintAgent\n"
        "\n"
        "src = inspect.getsource(ComplaintAgent.propose_action)\n"
        "# Show the step-dispatch skeleton: the lines that select a tool per step.\n"
        "for line in src.splitlines():\n"
        "    s = line.strip()\n"
        "    if s.startswith('if state.step ==') or s.startswith('tool_name='):\n"
        "        print(line)"
    ),
    new_markdown_cell(
        "## The workflow DAG: the `has_enables` relation\n"
        "\n"
        "The plan's ordering constraints are stored as triples over the relation "
        "`has_enables` in the GMS banking store. Each triple `(prev, has_enables, next)` "
        "asserts that `next` is an admissible successor of `prev`. The triples form a "
        "directed acyclic graph rooted at a synthetic `start` node. The edges below are "
        "read directly from the store's `triples.json`; they are the ground truth the "
        "plausibility gate scores against."
    ),
    new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "root = Path('.') if Path('data').exists() else Path('..')\n"
        "store_path = root / 'data' / 'gms_banking_store'\n"
        "triples = json.loads((store_path / 'triples.json').read_text())\n"
        "enables = [t for t in triples if t[1] == 'has_enables']\n"
        "print(f'{len(enables)} has_enables edges define the workflow DAG:')\n"
        "for head, rel, tail in enables:\n"
        "    print(f'  {head:16s} --{rel}--> {tail}')"
    ),
    new_markdown_cell(
        "The canonical spine `start -> classify -> extract -> search_policy -> "
        "flag_regulatory -> draft_response` is one path through this graph. The two "
        "additional edges into `escalate` --- from `flag_regulatory` and from "
        "`draft_response` --- encode the replanning branches of Chapter~8: a regulatory "
        "flag or an unsafe draft diverts the plan to escalation rather than continuing "
        "along the spine. The DAG therefore captures both the nominal work list and the "
        "sanctioned deviations from it."
    ),
    new_markdown_cell(
        "## The previous node of a transition\n"
        "\n"
        "The gate scores a transition `(prev_node, has_enables, tool_node)`. The context "
        "function `_prev_workflow_node` computes `prev_node` from the number of tool "
        "results recorded so far: before the first tool it returns `start`, and otherwise "
        "the node of the tool that ran last. The function reads only "
        "`len(state.tool_results)`, so its behavior can be exercised offline with a small "
        "stand-in object bearing that one attribute."
    ),
    new_code_cell(
        "from types import SimpleNamespace\n"
        "from agentlab.capstone.complaint_agent import _prev_workflow_node\n"
        "\n"
        "# _prev_workflow_node reads only len(state.tool_results); a stand-in suffices\n"
        "# for a structural demonstration (no store, no GPU).\n"
        "for n in range(6):\n"
        "    state = SimpleNamespace(tool_results=[{}] * n)\n"
        "    prev = _prev_workflow_node(action=None, state=state)\n"
        "    print(f'after {n} tool result(s): prev_node = {prev!r}')"
    ),
    new_markdown_cell(
        "## The plausibility gate enforces the plan\n"
        "\n"
        "`build_complaint_harness` assembles three gates around the executor: a "
        "`SyntaxGate`, the policy engine as a gate and the `GMSPlausibilityGate`. The "
        "plausibility gate is the one that enforces the plan. It is constructed by "
        "`_build_gms_plausibility_gate`, which loads the trained store, reads the "
        "calibrated threshold from `calibration.json`, and binds the gate to the "
        "`has_enables` relation with `_prev_workflow_node` as its context function. The "
        "gate scores each proposed `(prev_node, has_enables, tool_node)` transition and "
        "denies any whose geodesic distance exceeds the threshold. The construction below "
        "loads the store; it is reader-runnable rather than executed in this build."
    ),
    new_code_cell(
        "# Reader-runnable: this loads the trained GMS banking store (GPU if available).\n"
        "# It is shown, not executed, in the notebook build.\n"
        "from agentlab.capstone.complaint_agent import _build_gms_plausibility_gate\n"
        "\n"
        "gate = _build_gms_plausibility_gate()\n"
        "print('relation :', gate._relation)      # 'has_enables'\n"
        "print('theta    :', gate._theta)         # calibrated operating point\n"
        "print('on_missing:', gate._on_missing)   # behavior when the store is silent"
    ),
    new_markdown_cell(
        "The calibrated threshold is not a default. It is read from the store's "
        "`calibration.json`, where it was fitted on a cohort of admissible and "
        "inadmissible transitions to a chosen false-allow ceiling. The cell below reads "
        "that record offline; the threshold is the operating point the gate applies to "
        "every transition."
    ),
    new_code_cell(
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "root = next((c for c in (Path('.'), Path('..'), Path('../code'), Path('code')) if (c / 'data').exists()), Path('.'))\n"
        "cal_path = root / 'data' / 'gms_banking_store' / 'calibration.json'\n"
        "\n"
        "def _stale(path):\n"
        "    if not path.exists():\n"
        "        return True\n"
        "    payload = json.loads(path.read_text())\n"
        "    pg = payload.get('plausibility_gate', payload)\n"
        "    return 'relation_set' not in pg\n"
        "\n"
        "if _stale(cal_path):\n"
        "    print('calibration.json missing or stale -- recalibrating ...')\n"
        "    sys.path.insert(0, str(root / 'scripts'))\n"
        "    import calibrate_gms_thresholds\n"
        "    calibrate_gms_thresholds.main()\n"
        "\n"
        "cal = json.loads(cal_path.read_text())\n"
        "pg = cal['plausibility_gate'] if 'plausibility_gate' in cal else cal\n"
        "print('relation set     :', pg['relation_set'])\n"
        "print('threshold (theta):', pg['threshold'])\n"
        "print('cohort size      :', pg['cohort_n'])\n"
        "print('false-allow rate :', pg['false_allow_rate'])"
    ),
    new_markdown_cell(
        "## Reading an out-of-order transition\n"
        "\n"
        "The plan forbids skipping steps: `draft_response` may not follow `classify`, "
        "because no `has_enables` edge connects them. A free planner could propose such a "
        "jump; the fixed workflow does not, and the gate rejects it if it were proposed. "
        "The cell below shows the two transitions --- one on the spine, one a forbidden "
        "skip --- checked against the DAG edges read earlier, so the admissibility "
        "distinction is visible without loading the store. The trained gate scores these "
        "transitions geometrically rather than by exact edge lookup, but the DAG membership "
        "is what calibration teaches it to reproduce."
    ),
    new_code_cell(
        "edge_set = {(h, t) for h, _r, t in enables}\n"
        "\n"
        "candidates = [\n"
        "    ('extract', 'search_policy'),      # on the spine: enabled\n"
        "    ('flag_regulatory', 'escalate'),   # sanctioned replanning branch: enabled\n"
        "    ('classify', 'draft_response'),    # forbidden skip: not enabled\n"
        "    ('start', 'search_policy'),        # forbidden skip: not enabled\n"
        "]\n"
        "for prev, nxt in candidates:\n"
        "    admissible = (prev, nxt) in edge_set\n"
        "    verdict = 'ALLOW (in DAG)' if admissible else 'DENY (not in DAG)'\n"
        "    print(f'{prev:16s} -> {nxt:16s} : {verdict}')"
    ),
    new_markdown_cell(
        "## Summary and connections\n"
        "\n"
        "The capstone realizes Chapter~8 by fixing the plan rather than planning at run "
        "time. The work list is the ordered `_WORKFLOW_NODES`; the ordering constraints "
        "and the sanctioned replanning branches are the `has_enables` DAG in the GMS "
        "banking store; and enforcement is the `GMSPlausibilityGate`, which scores each "
        "transition against that DAG at a calibrated threshold and denies what the plan "
        "does not admit. Decomposition is therefore a design-time decision made explicit "
        "in a graph, and replanning is confined to the escalation edges the graph "
        "sanctions. Chapter~8 develops the general contrast between free planners and "
        "fixed workflows and the conditions under which each is appropriate; the capstone "
        "chapter (Chapter~15) assembles this workflow together with the typed tools of "
        "Chapter~5 and the execution guard of Chapter~6 into the governed agent that runs "
        "end to end."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
