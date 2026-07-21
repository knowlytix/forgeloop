#!/usr/bin/env python
"""Builder for notebook 12_5_component_testing.ipynb (Section 12.5, sec:capstone-component).

Dependency-light: the executable cells load the pinned artifacts, print each
component's numbers and run assert-based self-checks. One section per testable
component -- the gate stack, then the four model-backed tools (complaint
classification, policy search, regulatory escalation and the response draft) --
each scored by its OWN metric, not a single "accuracy". The heavy paths (the gate
probes, the campaign, the draft judge) are shown as reference only.

The gate stack and the draft groundedness are DETERMINISTIC and asserted exactly.
The per-tool model accuracies come from the pinned campaign (capstone_run.json,
restored to the published Chapter 12 values); they are asserted exactly so the
notebook, the chapter table and the artifact agree.
"""
import contextlib
import io
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output
from nbformat import NO_CONVERT
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "12_5_component_testing.ipynb"

# Embed computed outputs WITHOUT a Jupyter kernel: each code cell is rendered by
# running its own source in one shared in-process namespace against the PINNED
# artifacts and capturing stdout (see memory: never exec the notebook via a kernel
# on drifting data). The data is pinned, so this is deterministic; the asserts in
# the self-check run here too, so a stale number fails the build.
_NS = {}


def code_cell(src):
    cell = new_code_cell(src)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(src, _NS)
    text = buf.getvalue()
    if text:
        cell.outputs = [new_output("stream", name="stdout", text=text)]
    return cell


cells = []

# 1. Title + intro
cells.append(new_markdown_cell(
    "# 12.5 Component Testing: Each Part Against Ground Truth\n"
    "\n"
    "Testing runs from the parts to the whole. Before asking whether the assembled "
    "agent behaves, each component is tested in isolation against ground truth, on the "
    "runs in which it executed: a part that is wrong on its own cannot be trusted in "
    "composition, and a part that is right localizes any later system failure to the "
    "wiring rather than to itself. The agent has five testable components -- the **gate "
    "stack** that screens every call, and the four model-backed tools: **complaint "
    "classification**, **policy search**, **regulatory escalation** and the **response "
    "draft**. Each is scored by the metric that fits it, not one shared 'accuracy'. Fact "
    "extraction has no row of its own; its grounded fact/query is read through the two "
    "tools it drives (search and the flagger)."
))

# 2. Reference (run separately)
cells.append(new_markdown_cell(
    "**Reference (run separately):** the heavier calls that produced these numbers. "
    "The gate probes and the draft judge are deterministic -- they read the gates' "
    "decisions and the store's geometry directly, owing nothing to the model-generated "
    "campaign. The per-tool numbers are read off the end-to-end campaign rows.\n"
    "\n"
    "```python\n"
    "from apps.complaint_sut import probe_policy_gate, probe_plausibility_gate\n"
    "\n"
    "# Gate stack, graded on its admit-path and its refuse-path.\n"
    "pol = probe_policy_gate(cases)        # Policy gate on the canonical messages\n"
    "pla = probe_plausibility_gate()       # sweep every workflow transition vs has_enables\n"
    "\n"
    "# Per-tool scoring happens inside the end-to-end run, on the runs where each tool\n"
    "# executed; search_policy is scored separately by benchmark_retrieval against the\n"
    "# graph ground truth (parse->bind->retrieve, value match), not a top-k label.\n"
    "\n"
    "# Response draft: groundedness-as-distance (Chapter on GMS). score_triple aligns\n"
    "# the draft's fee figure to a store triple; the tier bands per relation are\n"
    "# calibrated from the store, not hard-coded.\n"
    "rows = evaluate.run(scns, sut, judge_fn=store_judge)   # adds draft_tier, draft_score\n"
    "```"
))

# 3. Load artifacts
cells.append(new_markdown_cell(
    "**What the next cell does** — loads the pinned artifacts the section reads:\n"
    "\n"
    "1. **Find the repo root.** Walk up from `Path.cwd()` until `data/capstone_run.json` is found.\n"
    "2. **Load the campaign.** `D = json.loads(...capstone_run.json)` holds the end-to-end run, the `gate_stack` block and the `per_tool` numbers.\n"
    "3. **Load the graph-truth retrieval.** `R = json.loads(...capstone_retrieval.json)` holds the `benchmark_retrieval` result for policy search: recall/precision against the value in the graph, with parse/bind/retrieve localization.\n"
    "4. **Load the draft companion.** `C = json.loads(...capstone_companions.json)` holds `draft_groundedness` (the store-calibrated tier bands and the judged example drafts).\n"
    "5. **Bind the per-tool view.** `pt = D['per_tool']` is the handle the label-scored tools read from below."
))
cells.append(code_cell(
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "root = Path.cwd()\n"
    "while not (root/'data'/'capstone_run.json').exists() and root != root.parent:\n"
    "    root = root.parent\n"
    "D = json.loads((root/'data'/'capstone_run.json').read_text())   # campaign + gate stack\n"
    "R = json.loads((root/'data'/'capstone_retrieval.json').read_text())  # graph-truth retrieval\n"
    "C = json.loads((root/'data'/'capstone_companions.json').read_text())  # draft groundedness\n"
    "pt = D['per_tool']\n"
    "print('loaded artifacts from', root/'data')"
))

# 4. The gate -----------------------------------------------------------------
cells.append(new_markdown_cell(
    "## Component 1 -- The gate stack\n"
    "\n"
    "The gate stack is the first component and the one a normal run least exercises. Two "
    "gates carry a decision and are graded directly on the deployed agent, each on its "
    "**refuse-path** (does it stop what it must) and its **admit-path** (does it let "
    "benign work through). Both checks are deterministic, owing nothing to the campaign."
))

cells.append(new_markdown_cell(
    "**What the next cell does** — reports both gates, each on its refuse-path and its admit-path:\n"
    "\n"
    "1. **Read the gate block.** `gs = D['gate_stack']`, then `pg, pl = gs['policy_gate'], gs['plausibility_gate']`.\n"
    "2. **Policy gate.** Print the refuse-path (`pg['refuse_path']` -- `correct`/`n` and `detection_rate`) and the admit-path (`pg['allow_path']` -- `no_false_refusal`/`n` and `rate`).\n"
    "3. **Plausibility gate.** Print the threshold `pl['theta']`, the legal transitions admitted (`pl['legal']` -- `allowed`/`n`, `max_score`) and the illegal ones denied (`pl['illegal']` -- `denied`/`n`) with `len(pl['errors'])` near-neighbor misses."
))
cells.append(code_cell(
    "gs = D['gate_stack']\n"
    "pg, pl = gs['policy_gate'], gs['plausibility_gate']\n"
    "print('Policy gate (run on the canonical messages)')\n"
    "print(f\"  refuse-path : stopped {pg['refuse_path']['correct']}/{pg['refuse_path']['n']}\"\n"
    "      f\"   detection_rate={pg['refuse_path']['detection_rate']:.3f}\")\n"
    "print(f\"  admit-path  : passed  {pg['allow_path']['no_false_refusal']}/{pg['allow_path']['n']}\"\n"
    "      f\"   no-false-refusal={pg['allow_path']['rate']:.3f}\")\n"
    "print()\n"
    "print(f\"Plausibility gate (swept over every workflow transition, theta={pl['theta']})\")\n"
    "print(f\"  legal   : allowed {pl['legal']['allowed']}/{pl['legal']['n']}\"\n"
    "      f\"   max_score={pl['legal']['max_score']:.3f} (< theta)\")\n"
    "print(f\"  illegal : denied  {pl['illegal']['denied']}/{pl['illegal']['n']}\"\n"
    "      f\"   {len(pl['errors'])} near-neighbor misses\")"
))

cells.append(new_markdown_cell(
    "The **Policy gate** is perfect on its refuse-path: all five adversarial inputs are "
    "stopped at the first tool call -- three PII messages escalated, two prompt-injection "
    "attempts denied -- before the message reaches any tool. On the admit-path it passes "
    "all fifteen benign messages with no false refusal, so the injection guard is not paying "
    "for its precision with benign work turned away. The **plausibility "
    "gate** admits all five legal transitions (largest score 0.326, under $\\theta=0.4$) "
    "and denies fourteen of twenty illegal ones; the six it misses are near-neighbor "
    "transitions (self-repeats and single-step skips) the fixed workflow never proposes -- "
    "a latent gap an LLM planner could reach, a recalibration question, not a bug."
))

# 5. Classification -----------------------------------------------------------
cells.append(new_markdown_cell(
    "## Component 2 -- Complaint classification\n"
    "\n"
    "The classifier is scored against the labeled category on every run the gate let it "
    "execute -- a plain correctness rate, the metric that fits a single-label tool."
))

cells.append(new_markdown_cell(
    "**What the next cell does** — prints the classifier's correctness rate: read `pt['classify_complaint']` and report `accuracy` with the underlying `correct` of `scored` runs."
))
cells.append(code_cell(
    "t = pt['classify_complaint']\n"
    "print(f\"classification : {t['accuracy']:.2f}  ({t['correct']} of {t['scored']} runs)\")"
))

cells.append(new_markdown_cell(
    "It is the strongest component: the LoRA-fine-tuned encoder holds its accuracy across "
    "the presentation factors the earlier frozen-head version collapsed on."
))

# 6. Policy search (RAG) ------------------------------------------------------
cells.append(new_markdown_cell(
    "## Component 3 -- Policy search (RAG), against the graph ground truth\n"
    "\n"
    "Policy search binds a question to the graph and returns the fact that answers it, so "
    "the question its test asks is whether it retrieved the **right fact**, measured against "
    "the value that lives in the graph -- not whether a policy label appeared in a top-$k$. "
    "The faithful measure is **graph-truth** retrieval (`benchmark_retrieval`): each question "
    "carries the value the graph holds, the retriever parses it into a "
    "`(head, relation, tail)` triple, binds and retrieves, and a hit requires the retrieved "
    "**value** to match. Because grading runs the agent's own three steps, every miss is "
    "localized to **parse**, **bind** or **retrieve**."
))

cells.append(new_markdown_cell(
    "**What the next cell does** — prints policy search as graph-truth recall/precision: read `R['gms']` (recall, precision, parse and bind rates), contrast the dense baseline `R['dense']`, and localize the misses via `R['miss_stage']`."
))
cells.append(code_cell(
    "g, dn = R['gms'], R['dense']\n"
    "print(f\"graph-truth recall    : {g['recall']:.2f}   precision : {g['precision']:.2f}   (n={R['n']})\")\n"
    "print(f\"parse rate            : {g['parse_rate']:.2f}   bind rate : {g['bind_rate']:.2f}\")\n"
    "print(f\"dense baseline recall : {dn['recall']:.2f}   precision@k : {dn['precision_at_k']:.2f}\")\n"
    "print(f\"miss stages           : {R['miss_stage']}\")"
))

cells.append(new_markdown_cell(
    "The operator-native retriever reaches recall 0.85 and precision 0.85, with parse and "
    "bind rates 1.00: every question parsed into store vocabulary and bound to the correct "
    "policy entity. A dense top-5 baseline on the same tuned encoder reaches recall 0.65 and "
    "precision@$k$ 0.31, so the gap is the retrieval mechanism -- triple-mediation -- not the "
    "embedding. All three misses fall at the **retrieve** stage: the right policy, an adjacent "
    "field (a fee amount where the per-occurrence basis was asked, an escalation window where "
    "the dollar threshold was). None is a wrong-policy bind, so the work left is relation "
    "disambiguation on the bound head, not retrieval."
))

# 7. Regulatory escalation ----------------------------------------------------
cells.append(new_markdown_cell(
    "## Component 4 -- Regulatory escalation, with fact retrieval\n"
    "\n"
    "The regulatory flagger grounds on the same bound facts and is scored against the "
    "case's expected regulation -- the escalate / don't-escalate decision, judged on the "
    "runs where it executed."
))

cells.append(new_markdown_cell(
    "**What the next cell does** — prints the regulatory flagger's decision rate: read `pt['flag_regulatory']` and report `accuracy` (the escalate / don't-escalate decision) over `correct` of `scored` graded runs."
))
cells.append(code_cell(
    "t = pt['flag_regulatory']\n"
    "print(f\"escalation decision : {t['accuracy']:.2f}  ({t['correct']} of {t['scored']} graded runs)\")"
))

cells.append(new_markdown_cell(
    "It is correct on all but one of the runs it is graded on. Like policy search it consumes "
    "the fact extraction rather than the raw message, so a poor query would surface here too -- and "
    "on both rows the grounded fact/query holds up, which is how the extraction is read "
    "without a row of its own."
))

# 8. Response draft -----------------------------------------------------------
cells.append(new_markdown_cell(
    "## Component 5 -- The response draft\n"
    "\n"
    "The drafted reply is the one field the agent generates rather than transcribes, so "
    "the component test is whether it is **grounded in policy or merely sounds like it**. "
    "This is groundedness-as-distance: the draft's verifiable claim -- a dollar figure -- "
    "is aligned to a store triple and scored with `score_triple`, and the bands that turn "
    "the distance into a tier are **calibrated per relation from the store**, not "
    "hard-coded (the `score_triple` scale differs by relation). The check is deterministic "
    "and replayable: the same draft yields the same number a year later, which an LLM judge "
    "cannot promise."
))

cells.append(new_markdown_cell(
    "**What the next cell does** — shows the draft groundedness as distance, bands then examples:\n"
    "\n"
    "1. **Read the companion block.** `dg = C['draft_groundedness']`.\n"
    "2. **Print the tier bands.** Loop `dg['bands']` and print each relation's `grounded_max` and `fabricated_min` -- the store-calibrated cutoffs, distinct per relation.\n"
    "3. **Print the judged drafts.** Loop `dg['examples']` and print each draft's `label`, the aligned `triple`, its `geodesic` distance and the resulting `tier` (grounded, fabrication or n/a)."
))
cells.append(code_cell(
    "dg = C['draft_groundedness']\n"
    "print('Store-calibrated tier bands (per relation)')\n"
    "print(f\"{'relation':18s}{'grounded<=':>12s}{'>=fabricated':>14s}\")\n"
    "for rel, b in dg['bands'].items():\n"
    "    print(f\"{rel:18s}{b['grounded_max']:>12.3f}{b['fabricated_min']:>14.3f}\")\n"
    "print()\n"
    "print('judge_draft on representative drafts')\n"
    "print(f\"{'draft':26s}{'triple':42s}{'distance':>9s}  tier\")\n"
    "for e in dg['examples']:\n"
    "    print(f\"{e['label']:26s}{e['triple']:42s}{e['geodesic']:>9.3f}  {e['tier']}\")"
))

cells.append(new_markdown_cell(
    "The committed overdraft fee ($35) scores a geodesic distance of 0.056 and lands in "
    "the grounded band; a fabricated $50 fee scores near 1.55 and lands in the fabrication "
    "band -- the same contrast holds for the reversal cap on its own relation. A draft that "
    "states no such claim (an inquiry, or a reply naming only a policy and a deadline) has "
    "nothing to verify and is scored `n/a` rather than forced into a verdict. Run over every "
    "claim a draft makes, this pairs with a **provenance** check that traces each claim back "
    "to the policy text: across the drafted replies every claim that binds cites a span that "
    "contains the value it states (consistency 1.00, 33 of 33 claims; 26 of 33 resolve to a "
    "sentence-level span). A claim that **inverts** a policy's stance is caught separately by "
    "the value-polarity check of the resilience section, which flags three of four stance "
    "reversals as contradicted."
))

# 9. Self-check
cells.append(new_markdown_cell(
    "**What the next cell does** — asserts every reported number against the pinned artifacts so the notebook, the chapter table and the JSON stay in agreement:\n"
    "\n"
    "1. **Gate stack.** Assert the policy gate's `refuse_path` (detection 1.0, 5 of 5) and `allow_path` (14 of 15), and the plausibility gate's `theta`, `legal`, `illegal` and six `errors`.\n"
    "2. **Label-scored tools.** Assert `classify_complaint` and `flag_regulatory` exactly match the chapter values.\n"
    "3. **Graph-truth retrieval.** Assert `R['gms']` recall/precision and perfect parse/bind, and that all misses fall at the retrieve stage.\n"
    "4. **Draft groundedness.** Index `C['draft_groundedness']['examples']` by `label` and assert each `tier` and the grounded fee's `geodesic`.\n"
    "5. **Overall decision.** Assert the overall decision accuracy `D['overall']['accuracy']` is reported and scored per query, separate from the per-tool numbers."
))
cells.append(code_cell(
    "# --- Gate stack (deterministic) ---\n"
    "assert pg['refuse_path']['detection_rate'] == 1.0          # all adversarial inputs stopped\n"
    "assert pg['refuse_path']['correct'] == 5 and pg['refuse_path']['n'] == 5\n"
    "assert pg['allow_path']['no_false_refusal'] == 15 and pg['allow_path']['n'] == 15\n"
    "assert pl['theta'] == 0.4\n"
    "assert pl['legal']['allowed'] == 5 and pl['legal']['max_score'] < pl['theta']\n"
    "assert pl['illegal']['denied'] == 14 and pl['illegal']['n'] == 20\n"
    "assert len(pl['errors']) == 6                              # six near-neighbor misses\n"
    "\n"
    "# --- Label-scored tools match the chapter table (pinned reproducible campaign) ---\n"
    "assert pt['classify_complaint'] == {'scored': 92, 'correct': 83, 'accuracy': 0.902}\n"
    "assert pt['flag_regulatory'] == {'scored': 18, 'correct': 17, 'accuracy': 0.944}\n"
    "\n"
    "# --- Policy search: graph-truth retrieval (parse->bind->retrieve, value match) ---\n"
    "assert R['gms']['recall'] == 0.85 and R['gms']['precision'] == 0.85\n"
    "assert R['gms']['parse_rate'] == 1.0 and R['gms']['bind_rate'] == 1.0\n"
    "assert set(R['miss_stage']) == {'retrieve'}                # every miss on the correct head\n"
    "\n"
    "# --- Response draft groundedness (deterministic, store-calibrated) ---\n"
    "ex = {e['label']: e for e in C['draft_groundedness']['examples']}\n"
    "assert ex['grounded overdraft fee']['tier'] == 'grounded'\n"
    "assert abs(ex['grounded overdraft fee']['geodesic'] - 0.056) < 1e-2\n"
    "assert ex['distorted overdraft fee']['tier'] == 'fabrication'\n"
    "assert ex['grounded reversal cap']['tier'] == 'grounded'\n"
    "assert ex['fabricated reversal cap']['tier'] == 'fabrication'\n"
    "\n"
    "# --- Overall decision accuracy: reported per query, separate from the per-tool numbers ---\n"
    "assert 'accuracy' not in D                                 # not a bare top-level rate\n"
    "assert D['overall']['accuracy'] == 0.675                   # right classification AND escalation\n"
    "print('self-check passed')"
))

nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}

# Outputs were computed and embedded above (no kernel); just write the notebook.
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    nbformat.write(nb, f, version=NO_CONVERT)
print('wrote', OUT, 'cells', len(nb.cells))
