"""Builder for notebook 10 (Capstone): logistic attribution on the real campaign.

Companion to Chapter 10 (the attribution methodology): it runs the chapter's analysis
on the real capstone campaign -- PER-COMPONENT (empirical per-level failure rates, the
actionable "which component to fix") and the OVERALL per-query decision (120 runs) --
and hands off to notebook 12.6 for the diagnosis in context. On this retrained agent no
factor is significant after correction (clarity p_adj=0.96), reversing the old finding.

Outputs are computed and embedded WITHOUT a Jupyter kernel: each code cell's source is
run once in a shared in-process namespace against the PINNED artifact and its stdout
captured (see memory: never exec the notebook via a kernel on drifting data).
"""
import contextlib
import io
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output
from nbformat import NO_CONVERT

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "10_analysis_capstone.ipynb"
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
    "# 10 (Capstone) - Logistic Attribution on the Real Campaign\n"
    "\n"
    "Chapter 10 turns a table of binary outcomes and factor levels into a ranked, "
    "defensible statement about what causes failure. On a multi-component agent the "
    "decisive question is **which component to fix**, so the analysis is run "
    "**per component** -- the same deviance/Benjamini-Hochberg machinery, but on each "
    "tool's own correctness rather than on the end-to-end decision. Because the "
    "failures are few, a logistic fit is unstable and its odds ratios can diverge at "
    "zero-failure levels, so the robust report is the **empirical failure rate** at each "
    "factor level. The per-query decision is kept as a separate **overall** summary. This "
    "companion reads the pinned campaign; the diagnosis in context is notebook 12.6."
))

cells.append(new_markdown_cell(
    "**Reference (run separately):** the calls that produced these numbers, over the rows "
    "of one balanced 120-run campaign. The seed case is excluded -- it was a blocking "
    "factor, not a property under test.\n"
    "\n"
    "```python\n"
    "from gmstest.evaluate import attribute, weak_link\n"
    "\n"
    "# PER-COMPONENT: which input breaks which tool (empirical per-level failure rates).\n"
    "# search_policy is graded by graph-truth (Sec. 12.5), so it is not label-attributed here.\n"
    "for col in ('c_classify_complaint', 'c_flag_regulatory'):\n"
    "    sub = [r for r in rows if r[col] is not None]\n"
    "    attribute(sub, factors, metric=col)\n"
    "\n"
    "# OVERALL per-query decision (right classification AND escalation).\n"
    "overall = attribute(rows, factors)                  # metric defaults to 'correct'\n"
    "blame   = weak_link(rows, workflow)\n"
    "```"
))

cells.append(new_markdown_cell(
    "**What the next cell does** -- load the pinned campaign and define a table printer for the "
    "overall logistic attribution (one row per factor, with the worst level for a significant one)."
))
cells.append(code_cell(
    "import json\n"
    "from pathlib import Path\n"
    "\n"
    "root = Path.cwd()\n"
    "while not (root/'data'/'capstone_run.json').exists() and root != root.parent:\n"
    "    root = root.parent\n"
    "D = json.loads((root/'data'/'capstone_run.json').read_text())\n"
    "by_component = D['attribution_by_component']\n"
    "overall = D['overall']\n"
    "weak_link = D['weak_link']\n"
    "WORKFLOW = ['classify_complaint', 'extract_facts', 'search_policy',\n"
    "            'flag_regulatory', 'draft_response']\n"
    "FACTOR_ORDER = ['reasoning_cue', 'clarity', 'entity_aliasing']\n"
    "\n"
    "def show_factor_table(tbl):\n"
    "    print(f\"  {'factor':16s}{'G2':>8s}{'p_adj':>9s}{'pseudoR2':>10s}{'sig':>6s}   worst level (odds ratio)\")\n"
    "    for r in tbl:\n"
    "        ors = r['odds_ratios']; worst = min(ors, key=ors.get)\n"
    "        worst_str = f\"{worst} ({ors[worst]})\" if r['significant_adj'] else '---'\n"
    "        print(f\"  {r['factor']:16s}{r['G2']:>8.2f}{r['p_adj']:>9.4f}{r['pseudo_r2']:>10.3f}\"\n"
    "              f\"{str(r['significant_adj']):>6s}   {worst_str}\")\n"
    "print('loaded per-component + overall attribution from', root/'data')"
))

# Per-component (the focus) ---------------------------------------------------
cells.append(new_markdown_cell(
    "## Which input breaks which component\n"
    "\n"
    "Each label-scored tool is analyzed on its own correctness, on the runs where it was "
    "graded. If a level concentrates a tool's failures it is the input condition to fix; if "
    "the rates are flat, no presentation factor drives the tool. Reported as empirical failure "
    "rates (failed / graded) -- robust to the separation that would blow up a logistic odds "
    "ratio on so few failures."
))
cells.append(new_markdown_cell(
    "**What the next cell does** -- for each tool, print its graded/failure counts and, per factor "
    "(reasoning cue first), the `level=failed/n` failure rates."
))
cells.append(code_cell(
    "for tool in ('classify_complaint', 'flag_regulatory'):\n"
    "    a = by_component[tool]\n"
    "    print(f\"{tool}  (graded {a['scored']}, failures {a['failures']})\")\n"
    "    for f in FACTOR_ORDER:\n"
    "        levels = a['by_factor'][f]\n"
    "        cells_str = '  '.join(f\"{lvl}={v['failed']}/{v['n']}\" for lvl, v in levels.items())\n"
    "        print(f\"    {f:16s}{cells_str}\")\n"
    "    print()"
))
cells.append(new_markdown_cell(
    "The classifier's nine failures spread across the levels -- the nearest gradient is "
    "`clarity` (ambiguous against clear), the reasoning cue is flat -- and no level "
    "concentrates them; the corrected fit certifies nothing (`clarity` nearest at "
    "$p_{adj}=0.96$). That is the actionable diagnosis a single end-to-end rate cannot give, "
    "here a null one: on this retrained agent no presentation factor drives a component. This "
    "reverses the earlier agent, where a downplaying cue dominated. The flagger fails once and "
    "has nothing to attribute. (`search_policy` is graded by graph-truth in Section 12.5, so "
    "it is not label-attributed here.)"
))

# Overall (separate summary) --------------------------------------------------
cells.append(new_markdown_cell(
    "## The overall decision, as a separate summary\n"
    "\n"
    "The end-to-end view read as a whole is the per-query decision: the right classification "
    "and the right escalate/don't-escalate call. We report its rate, its 120-run factor "
    "attribution and the weak-link decomposition; the per-tool numbers are scored separately."
))
cells.append(new_markdown_cell(
    "**What the next cell does** -- print the overall rate and definition, the overall logistic "
    "attribution table, and the weak-link blame per workflow tool."
))
cells.append(code_cell(
    "print(f\"Overall decision accuracy: {overall['accuracy']:.3f}\")\n"
    "print(f\"  {overall['definition']}\\n\")\n"
    "print('Overall factor attribution (logistic, BH-corrected)')\n"
    "show_factor_table(overall['attribution'])\n"
    "print()\n"
    "print('Weak-link blame (wrongly-decisioned runs -> first erring tool)')\n"
    "for tool in WORKFLOW:\n"
    "    if tool in weak_link:\n"
    "        print(f\"  {tool:20s} {weak_link[tool]}\")\n"
    "print(f\"  {'total tool-attributed':20s} {sum(weak_link.values())}\")"
))
cells.append(new_markdown_cell(
    "Run on the decision bit, the factor attribution finds **nothing** significant after "
    "correction, which here agrees with the per-component view rather than contradicting it: "
    "the retrained agent does not break on the presentation factors the design varies. The "
    "lesson stands on a multi-component agent: attribute the components separately, because a "
    "single bit can wash out a signal the per-component view would localize. Notebook 12.6 "
    "reads both as the capstone's diagnosis."
))

# Self-check ------------------------------------------------------------------
cells.append(new_markdown_cell(
    "**What the next cell does** -- pin the SHAPE of the analysis, not volatile point estimates: each "
    "tool's per-level failure rates sum to its failure and graded counts; the overall decision is a "
    "valid rate with a well-formed logistic table; weak-link blame falls only on workflow tools."
))
cells.append(code_cell(
    "FACTORS = {'clarity', 'entity_aliasing', 'reasoning_cue'}\n"
    "assert set(by_component) == {'classify_complaint', 'search_policy', 'flag_regulatory'}\n"
    "for tool, a in by_component.items():\n"
    "    assert set(a['by_factor']) == FACTORS\n"
    "    for f, levels in a['by_factor'].items():\n"
    "        assert sum(v['failed'] for v in levels.values()) == a['failures']\n"
    "        assert sum(v['n'] for v in levels.values()) == a['scored']\n"
    "FACT_KEYS = {'factor', 'G2', 'p_adj', 'pseudo_r2', 'significant_adj', 'odds_ratios'}\n"
    "assert 0.0 <= overall['accuracy'] <= 1.0\n"
    "assert {r['factor'] for r in overall['attribution']} == FACTORS\n"
    "for r in overall['attribution']:\n"
    "    assert FACT_KEYS <= set(r) and 0.0 <= r['p_adj'] <= 1.0\n"
    "assert set(weak_link) <= set(WORKFLOW) and all(v > 0 for v in weak_link.values())\n"
    "print('self-check passed')"
))

nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}

# Outputs computed and embedded above (no kernel); just write the notebook.
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    nbformat.write(nb, f, version=NO_CONVERT)
print('wrote', OUT, 'cells', len(nb.cells))
