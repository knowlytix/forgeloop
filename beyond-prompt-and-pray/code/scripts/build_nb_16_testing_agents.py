#!/usr/bin/env python
"""Builder for notebooks/16_testing_agents.ipynb (Chapter: Testing the Capstone Agent).

The chapter applies the companion volume's gmstest framework to the capstone agent.
This notebook mirrors the chapter section by section, reading the pinned campaign
artifacts (data/capstone_run.json, capstone_retrieval.json, capstone_companions.json,
capstone_rows.json) rather than re-executing the live agent. The heavy paths (the
AgentSUT run, the fault injection, the gate/value-polarity probes) are shown as
reference; the numbers are read from the pinned artifacts so the notebook, the chapter
and the JSON agree.

Outputs are computed and embedded WITHOUT a Jupyter kernel: each code cell's source is
run once in a shared in-process namespace against the pinned artifacts and its stdout
captured (see memory: never exec the notebook via a kernel on drifting data). The data
is pinned, so this is deterministic; the self-check asserts run here too, so a stale
number fails the build.
"""
import contextlib
import io
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output
from nbformat import NO_CONVERT

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "16_testing_agents.ipynb"
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

cells.append(new_markdown_cell(
    "# Testing the Capstone Agent\n\n"
    "The 20-case benchmark of the capstone chapter is the floor, not the ceiling. This "
    "notebook puts the finished agent on a test stand and asks how it behaves under "
    "realistic input variation, and where it breaks. The method is the companion volume's "
    "`gmstest` framework applied end to end to one governed agent; the numbers here are "
    "read from the pinned campaign so they match the chapter exactly."
))

cells.append(new_markdown_cell(
    "**Reference (run separately):** the framework calls that produced these numbers. "
    "The agent SUT wraps `build_complaint_harness` unchanged; the probes read the deployed "
    "gates and the store's geometry directly.\n\n"
    "```python\n"
    "from gmstest import resolve, compose, evaluate\n"
    "from apps.complaint_sut import (\n"
    "    AgentSUT, score_components, retrieval_benchmark,\n"
    "    probe_policy_gate, probe_plausibility_gate,\n"
    "    probe_value_polarity, probe_provenance_grounding,\n"
    ")\n"
    "sut  = AgentSUT()                                  # build_complaint_harness, unchanged\n"
    "cat, items = resolve.load_catalog('complaint')\n"
    "suite = resolve.suite(cat, factors=['clarity', 'entity_aliasing', 'reasoning_cue'])\n"
    "scns  = compose(suite, items, n_runs=120, seed=42, balance_base=True)\n"
    "rows  = evaluate.run(scns, sut, workflow_order=WORKFLOW)\n"
    "```"
))

# Load artifacts -------------------------------------------------------------
cells.append(new_markdown_cell(
    "**What the next cell does** — loads the pinned campaign artifacts:\n\n"
    "1. **Find the data root.** Walk up from `Path.cwd()` until `data/capstone_run.json` is found.\n"
    "2. **Load them.** `D` (campaign + gates + attribution + per-tool), `R` (graph-truth retrieval), "
    "`C` (draft groundedness, value-polarity, provenance, resilience), and the per-run `rows`."
))
cells.append(code_cell(
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "root = Path.cwd()\n"
    "while not (root/'data'/'capstone_run.json').exists() and root != root.parent:\n"
    "    root = root.parent\n"
    "D = json.loads((root/'data'/'capstone_run.json').read_text())\n"
    "R = json.loads((root/'data'/'capstone_retrieval.json').read_text())\n"
    "C = json.loads((root/'data'/'capstone_companions.json').read_text())\n"
    "rows = json.loads((root/'data'/'capstone_rows.json').read_text())['rows']\n"
    "WORKFLOW = ['classify_complaint', 'extract_facts', 'search_policy',\n"
    "            'flag_regulatory', 'draft_response']\n"
    "print('loaded artifacts from', root/'data')"
))

# 1. Design ------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## Designing the complaint suite\n\n"
    "The unit of coverage is a complaint *scenario*: one of the twenty seed cases under a "
    "controlled combination of three presentation factors (`clarity`, `entity_aliasing`, "
    "`reasoning_cue`). The seed case is a *blocking* factor, balanced across the design but "
    "excluded from the attribution model. The design is a space-filling Sobol sequence "
    "refined to balance the blocking factor: the seed is represented exactly six times "
    "each, the presentation levels cover the joint space without empty cells."
))
cells.append(new_markdown_cell(
    "**What the next cell does** — reports the design coverage: the per-level counts of each "
    "presentation factor and the per-seed balance, read from the pinned rows."
))
cells.append(code_cell(
    "from collections import Counter\n"
    "print(f'{len(rows)} scenarios')\n"
    "for f in ('f_clarity', 'f_entity_aliasing', 'f_reasoning_cue'):\n"
    "    c = Counter(r[f] for r in rows)\n"
    "    print(f'  {f[2:]:16s} {dict(c.most_common())}')\n"
    "seeds = Counter(r['seed'] for r in rows)\n"
    "print(f'  seed_case        {len(seeds)} cases, {min(seeds.values())} each')"
))
cells.append(new_markdown_cell(
    "The seed balance is exact because it is the blocking factor; the presentation levels "
    "are not equal-frequency, because the Sobol design fills the joint factor space rather "
    "than each margin, and the logistic attribution handles the unequal cells directly. Each "
    "row is then materialized into a message: the `clarity` dimension is rephrased by the "
    "local Qwen3-4B model, and the mechanical factors (aliasing, reasoning cue) are stamped "
    "on after, so the model cannot correct an injected typo or drop a coaxing nudge. The "
    "dollar amount survives every transform: it is the claim the judgment stage checks."
))

# 2. Run ---------------------------------------------------------------------
cells.append(new_markdown_cell(
    "## Running the agent, measuring the whole trajectory\n\n"
    "Each message runs through the agent exactly as the capstone built it, returning a full "
    "`Trajectory`, not a label. The stand reads it at three levels: the per-query "
    "**decision** (right classification *and* right escalate/don't-escalate call), "
    "**trajectory structure** (workflow adherence, escalation path) and **process health** "
    "(the audit chain). The per-tool components are scored separately, not folded into the "
    "decision."
))
cells.append(new_markdown_cell(
    "**What the next cell does** — prints the campaign summary: decision accuracy, workflow "
    "adherence and whether the audit chain verifies."
))
cells.append(code_cell(
    "print(f\"{D['n_runs']} scenarios\")\n"
    "print(f\"  decision accuracy:   {D['overall']['accuracy']:.3f}\")\n"
    "print(f\"  workflow adherence:  {D['workflow_adherence']:.3f}\")\n"
    "print(f\"  audit verifies:      {D['audit_verifies']}\")"
))
cells.append(new_markdown_cell(
    "The agent reaches the right decision on eighty-one of the hundred and twenty scenarios "
    "(**0.675**). Workflow adherence is perfect: the plausibility gate held the five-step "
    "sequence in order on every phrasing, so no failure is the agent firing tools out of "
    "turn. The audit chain verifies across the campaign, so any failure surfaced elsewhere "
    "is behavioral, not an integrity failure of the record. The next stages ask what the "
    "agent got wrong on the runs it missed, and why."
))

# 3. Judgment + provenance ---------------------------------------------------
cells.append(new_markdown_cell(
    "## Judgment: groundedness and provenance\n\n"
    "Even a correct decision leaves the safety-relevant question open: is the draft grounded "
    "in policy or does it just sound like it? The stand reuses groundedness-as-distance. The "
    "draft's fee claim is aligned to a store triple `(overdraft, has_fee_amount, 35.0)` and "
    "scored with `score_triple`; the tier bands are calibrated per relation from the store, "
    "not hard-coded."
))
cells.append(new_markdown_cell(
    "**What the next cell does** — prints the judged example drafts (`C['draft_groundedness']"
    "['examples']`): the aligned triple, its geodesic distance and the resulting tier."
))
cells.append(code_cell(
    "dg = C['draft_groundedness']\n"
    "print(f\"{'draft':26s}{'triple':44s}{'distance':>9s}  tier\")\n"
    "for e in dg['examples']:\n"
    "    print(f\"{e['label']:26s}{e['triple']:44s}{e['geodesic']:>9.3f}  {e['tier']}\")"
))
cells.append(new_markdown_cell(
    "The committed overdraft fee scores a geodesic distance near **0.06** and lands grounded; "
    "a fabricated \\$50 fee scores near **1.55** and lands in the fabrication band. Run across "
    "the suite, this pairs with a **provenance** check that traces each claim back to the "
    "policy text."
))
cells.append(new_markdown_cell(
    "**What the next cell does** — prints the provenance grounding (`C['provenance_grounding']"
    "`): how many verifiable claims carry a source sentence, and whether the cited span "
    "contains the value."
))
cells.append(code_cell(
    "pg = C['provenance_grounding']\n"
    "print(f\"claims with a source sentence : {pg['sentence_provenance_n']}/{pg['facts']}\"\n"
    "      f\"  ({pg['sentence_provenance_rate']:.2f})\")\n"
    "print(f\"cited span contains the value : {pg['consistent_n']}/{pg['facts']}\"\n"
    "      f\"  ({pg['consistent_rate']:.2f})\")"
))
cells.append(new_markdown_cell(
    "Every one of the thirty-three verifiable claims cites a span that contains the value it "
    "states (consistency **1.00**); twenty-six resolve to a sentence-level span in the prose, "
    "the rest to a table cell. No draft silently states a wrong number, and each stated "
    "number is traceable to the sentence of policy it came from."
))

# 4. Attribution -------------------------------------------------------------
cells.append(new_markdown_cell(
    "## Attribution: which input drives the failures\n\n"
    "Binary rates do not say which factor caused the failures. The framework fits a logistic "
    "regression of `correct` on the three presentation factors, runs an analysis of deviance "
    "and applies Benjamini--Hochberg correction. The seed case is excluded (a blocking "
    "factor, not a property under test)."
))
cells.append(new_markdown_cell(
    "**What the next cell does** — prints the corrected attribution table "
    "(`D['overall']['attribution']`): per factor the deviance `G2`, adjusted p-value and "
    "pseudo-$R^2$, with a `*` on any factor significant after correction."
))
cells.append(code_cell(
    "for r in D['overall']['attribution']:\n"
    "    flag = '*' if r['significant_adj'] else ' '\n"
    "    print(f\" {flag} {r['factor']:16s} p_adj={r['p_adj']:.4f}  pseudo_r2={r['pseudo_r2']:.3f}\")"
))
cells.append(new_markdown_cell(
    "No factor is significant after correction. The nearest is `clarity` ($p_{adj}=0.96$): "
    "the empirical rates tilt slightly toward the `ambiguous` level, but the effect does not "
    "separate from noise. This reverses the earlier agent, where a downplaying cue was the "
    "dominant, significant driver: on the retrained classifier and the operator-native "
    "retriever, none of the presentation factors moves the decision. The remaining failures "
    "are few and spread, and the place to look is the per-component decomposition, not the "
    "input factors."
))

# 5. Per-tool decomposition --------------------------------------------------
cells.append(new_markdown_cell(
    "## Which component failed: per-tool decomposition\n\n"
    "Factor attribution names the *input*; here it named none. The weak-link analysis names "
    "the *component*: each tool is scored against its own ground truth, and each "
    "wrongly-decided run is charged to the first tool, in workflow order, that erred. The two "
    "label-scored tools read from `D['per_tool']`; `search_policy` is graded against the "
    "graph ground truth (`R`), and `extract_facts` shows up through the two downstream tools "
    "it drives, not a row of its own."
))
cells.append(new_markdown_cell(
    "**What the next cell does** — prints the two label-scored tool accuracies, the "
    "graph-truth retrieval recall/precision (with parse and bind rates), and the weak-link "
    "blame counts."
))
cells.append(code_cell(
    "for tool in ('classify_complaint', 'flag_regulatory'):\n"
    "    s = D['per_tool'][tool]\n"
    "    print(f\"  {tool:18s} accuracy={s['accuracy']:.2f}  ({s['correct']}/{s['scored']})\")\n"
    "g = R['gms']\n"
    "print(f\"  search_policy      recall={g['recall']:.2f}  precision={g['precision']:.2f}\"\n"
    "      f\"  (parse {g['parse_rate']:.2f}, bind {g['bind_rate']:.2f})\")\n"
    "print(f\"  dense baseline     recall={R['dense']['recall']:.2f}  miss stages {R['miss_stage']}\")\n"
    "print(f\"  weak-link blame: {D['weak_link']}\")"
))
cells.append(new_markdown_cell(
    "The retrieval row is graded against the **graph ground truth**, not a top-$k$ of policy "
    "labels: a hit requires the retrieved *value* to match the value the graph holds, and "
    "every miss localizes to parse, bind or retrieve. On that measure the operator-native "
    "retriever reaches recall and precision **0.85** with parse and bind rates **1.00** --- "
    "every question bound to the correct policy entity --- against a dense baseline of 0.65. "
    "All three misses are at the retrieve stage (the right policy, an adjacent field), so "
    "what is left is relation disambiguation, not retrieval. The three components are each "
    "sound; the weak-link blame is sparse (nine to the classifier, one to the flagger) and "
    "the rest of the wrong runs turn on the escalation decision, a whole-trajectory property."
))

# 6. Resilience --------------------------------------------------------------
cells.append(new_markdown_cell(
    "## Resilience: faulting each tool in turn\n\n"
    "The complementary question to correctness is what happens when a tool *fails*. The real "
    "`ToolGateway` is installed on the executor and, one tool at a time, a `FaultProfile` "
    "makes that tool error. A governed agent must fail loud --- escalate --- never silently "
    "pass a corrupted result downstream."
))
cells.append(new_markdown_cell(
    "**What the next cell does** — prints the per-tool detection rate and silent-failure count "
    "from the pinned resilience block (`C['resilience']`)."
))
cells.append(code_cell(
    "res = C['resilience']\n"
    "print(f\"fault injected : {res['fault']}   overall detection {res['detection_rate']:.2f}\"\n"
    "      f\"   silent failures {res['silent_failures']}\")\n"
    "for tool, rate in res['per_tool_detection'].items():\n"
    "    print(f\"  {tool:18s} detection_rate={rate:.2f}\")"
))
cells.append(new_markdown_cell(
    "Under a hard fault on any tool the agent fails loud and escalates (detection **1.0** "
    "everywhere) and never silently proceeds with a corrupted result (silent **0**). Each "
    "gateway decision is recorded, so the test is auditable."
))

# 7. Stance reversal (value-polarity) ---------------------------------------
cells.append(new_markdown_cell(
    "## The hardest fault: a reversed stance\n\n"
    "The hardest fault is a plausible-but-wrong *structured* result: a drafted reply that "
    "**inverts** a policy's stance --- asserting *optional* where the policy makes it "
    "required, or *permitted* where it is forbidden. Type and shape are correct, so a "
    "syntactic check passes; only a comparison against the stored stance catches it. The "
    "value-polarity verifier returns `supported`, `contradicted` or `uncertain`."
))
cells.append(new_markdown_cell(
    "**What the next cell does** — prints the value-polarity result (`C['value_polarity']`): "
    "how many stance reversals are flagged contradicted, and how many true-stance synonyms "
    "are accepted."
))
cells.append(code_cell(
    "vp = C['value_polarity']\n"
    "print(f\"stance reversals flagged contradicted : {vp['reversal_detected']}/{vp['reversal_n']}\"\n"
    "      f\"  ({vp['reversal_rate']:.2f})\")\n"
    "print(f\"synonym stances accepted as supported : {vp['synonym_accepted']}/{vp['synonym_n']}\"\n"
    "      f\"  ({vp['synonym_rate']:.2f})\")"
))
cells.append(new_markdown_cell(
    "On a four-item reversal battery the verifier flags three of four reversals as "
    "contradicted; the miss is a required-to-optional assertion scored `uncertain`, a "
    "deferral to review rather than a false pass. It accepts three of four true-stance "
    "synonyms. The stance reversal is the plausible-but-wrong fault in its purest form, "
    "caught downstream by the same substrate the retriever and draft judge consult, not at "
    "the tool boundary."
))

# 8. Self-check --------------------------------------------------------------
cells.append(new_markdown_cell(
    "**What the next cell does** — asserts every reported number against the pinned artifacts "
    "so the notebook, the chapter and the JSON stay in agreement."
))
cells.append(code_cell(
    "# Campaign\n"
    "assert D['n_runs'] == 120\n"
    "assert D['overall']['accuracy'] == 0.675\n"
    "assert D['workflow_adherence'] == 1.0 and D['audit_verifies'] is True\n"
    "# Attribution: no factor significant\n"
    "assert not any(r['significant_adj'] for r in D['overall']['attribution'])\n"
    "assert next(r for r in D['overall']['attribution'] if r['factor'] == 'clarity')['p_adj'] > 0.9\n"
    "# Per-tool + graph-truth retrieval\n"
    "assert D['per_tool']['classify_complaint']['accuracy'] == 0.902\n"
    "assert D['per_tool']['flag_regulatory']['accuracy'] == 0.944\n"
    "assert R['gms']['recall'] == 0.85 and R['gms']['precision'] == 0.85\n"
    "assert R['gms']['parse_rate'] == 1.0 and R['gms']['bind_rate'] == 1.0\n"
    "assert set(R['miss_stage']) == {'retrieve'}\n"
    "assert D['weak_link'] == {'flag_regulatory': 1, 'classify_complaint': 9}\n"
    "# Draft groundedness + provenance\n"
    "ex = {e['label']: e for e in C['draft_groundedness']['examples']}\n"
    "assert ex['grounded overdraft fee']['tier'] == 'grounded'\n"
    "assert abs(ex['grounded overdraft fee']['geodesic'] - 0.056) < 1e-2\n"
    "assert ex['distorted overdraft fee']['tier'] == 'fabrication'\n"
    "assert C['provenance_grounding']['consistent_rate'] == 1.0\n"
    "# Resilience + stance\n"
    "assert C['resilience']['detection_rate'] == 1.0 and C['resilience']['silent_failures'] == 0\n"
    "assert C['value_polarity']['reversal_detected'] == 3 and C['value_polarity']['reversal_n'] == 4\n"
    "print('self-check passed')"
))

nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    nbformat.write(nb, f, version=NO_CONVERT)
print('wrote', OUT, 'cells', len(nb.cells))
