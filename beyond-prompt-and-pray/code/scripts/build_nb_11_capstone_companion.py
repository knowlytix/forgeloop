#!/usr/bin/env python
"""Builder for notebooks/11_capstone_companion.ipynb.

Capstone companion to Chapter 11 (Failure Modes, Adversarial Testing and Design
of Experiments): the concept read on the running banking complaint agent. Teaching
notebook in the style of the main chapter notebooks -- reads the pinned campaign
artifact data/capstone_run.json, no pre-embedded outputs (the reader runs it).
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "11_capstone_companion.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone companion --- Chapter 11: Failure Modes, Adversarial Testing and Design of Experiments\n"
        "\n"
        "Chapter~11 develops a discipline for locating an agent's failures rather than "
        "merely counting them. A factor-balanced suite varies the inputs along named "
        "axes, the agent is run across the resulting cells, and each failure is "
        "attributed to a factor level by a logistic model whose deviance test asks "
        "whether that factor explains the failures beyond chance. This companion reads "
        "that construction on the capstone banking complaint agent, using the campaign "
        "artifact `data/capstone_run.json` produced by the testing harness in "
        "`agentlab/testing`."
    ),
    new_markdown_cell(
        "The suite crosses three input factors. `clarity` controls whether the customer "
        "message states its issue plainly, ambiguously, or misleadingly; "
        "`entity_aliasing` controls whether products and parties are named canonically, "
        "by alias, or with a typo; and `reasoning_cue` controls whether the prompt "
        "carries a chain-of-thought cue, a misleading cue, or none. Twenty seed cases "
        "are each realized across the factor levels, so the campaign is a balanced "
        "design over the input variations rather than an ad hoc collection of examples."
    ),
    new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "root = Path('.') if Path('data').exists() else Path('..')\n"
        "run = json.loads((root / 'data' / 'capstone_run.json').read_text())\n"
        "print('runs in campaign :', run['n_runs'])\n"
        "print('workflow adherence:', run['workflow_adherence'])\n"
        "print('audit verifies    :', run['audit_verifies'])\n"
        "print('overall accuracy  :', run['overall']['accuracy'])\n"
        "print('accuracy meaning  :', run['overall']['definition'])"
    ),
    new_markdown_cell(
        "## The suite is balanced\n"
        "\n"
        "Attribution is only interpretable when the design is balanced, so that a "
        "factor level is not confounded with the difficulty of the seed cases it happens "
        "to appear in. The artifact records the realized counts per seed and per clarity "
        "level; the twenty seeds each appear the same number of times, and the clarity "
        "levels are close to even."
    ),
    new_code_cell(
        "seeds = run['seed_balance']\n"
        "print('distinct seeds     :', len(seeds))\n"
        "print('runs per seed (set):', sorted(set(seeds.values())))\n"
        "print('clarity balance    :', run['clarity_balance'])"
    ),
    new_markdown_cell(
        "## Factor attribution over the whole campaign\n"
        "\n"
        "For each factor, a logistic model regresses the per-run failure indicator on the "
        "factor levels. The likelihood-ratio statistic $G^2$ measures how much deviance "
        "the factor removes, and its $p$-value is adjusted for multiple factors. The "
        "decision rule of Chapter~11 is that a factor is implicated only when its "
        "adjusted $p$-value falls below the chosen level. The table below reports the "
        "real values from the campaign."
    ),
    new_code_cell(
        "print(f\"{'factor':16s} {'G2':>7s} {'p_adj':>8s} {'pseudo_r2':>10s} {'significant':>12s}\")\n"
        "print('-' * 56)\n"
        "for row in run['overall']['attribution']:\n"
        "    print(f\"{row['factor']:16s} {row['G2']:7.2f} {row['p_adj']:8.4f} \"\n"
        "          f\"{row['pseudo_r2']:10.3f} {str(row['significant_adj']):>12s}\")"
    ),
    new_markdown_cell(
        "No factor is significant after correction. The smallest adjusted $p$-value "
        "belongs to `clarity` at roughly $0.96$, far above any conventional level, and "
        "its pseudo-$R^2$ of about $0.015$ shows that the factor explains almost none of "
        "the variation in failure. The reading follows Chapter~11 exactly: the campaign "
        "does not license a claim that any single input axis drives the agent's failures. "
        "Clarity is merely the nearest of the three, not an implicated cause."
    ),
    new_code_cell(
        "clarity = next(r for r in run['overall']['attribution'] if r['factor'] == 'clarity')\n"
        "print('nearest factor    :', clarity['factor'])\n"
        "print('adjusted p-value  :', clarity['p_adj'])\n"
        "print('significant_adj   :', clarity['significant_adj'])\n"
        "print('level odds ratios :', clarity['odds_ratios'])\n"
        "assert not any(r['significant_adj'] for r in run['overall']['attribution']), \\\n"
        "    'no whole-campaign factor should be significant'"
    ),
    new_markdown_cell(
        "## Decomposing the weak link by component\n"
        "\n"
        "A campaign-wide null result does not mean the agent is uniformly healthy. "
        "Chapter~11 recommends decomposing failures by the component that produced them, "
        "because a factor that is invisible in the aggregate may concentrate in one tool. "
        "The artifact records the weak-link count per component: the number of runs in "
        "which each tool was the failing step."
    ),
    new_code_cell(
        "weak = run['weak_link']\n"
        "for comp, count in sorted(weak.items(), key=lambda kv: -kv[1]):\n"
        "    print(f'{comp:20s} failing-step count = {count}')\n"
        "print()\n"
        "print('weak link concentrates in:', max(weak, key=weak.get))"
    ),
    new_markdown_cell(
        "The failures concentrate in `classify_complaint` (nine failing runs) with a "
        "single failing run in `flag_regulatory`. Attribution is therefore most "
        "informative when restricted to the component that carries the failures. The "
        "per-component deviance table below repeats the logistic attribution within each "
        "tool that had enough failures to fit."
    ),
    new_code_cell(
        "abc = run['attribution_by_component']\n"
        "for comp, block in abc.items():\n"
        "    print(f\"=== {comp}  (scored={block['scored']}, failures={block['failures']}) ===\")\n"
        "    dev = block.get('deviance')\n"
        "    if not dev:\n"
        "        print('  too few failures to fit a per-factor model')\n"
        "        continue\n"
        "    for factor, d in sorted(dev.items(), key=lambda kv: kv[1]['p_adj']):\n"
        "        print(f\"  {factor:16s} G2={d['G2']:6.2f}  p_adj={d['p_adj']:.4f}  \"\n"
        "              f\"pseudo_r2={d['pseudo_r2']:.3f}  sig={d['significant_adj']}\")"
    ),
    new_markdown_cell(
        "Within `classify_complaint`, clarity is again the nearest factor, with an "
        "adjusted $p$-value around $0.17$ and the highest pseudo-$R^2$ of the block, yet "
        "it still does not clear the significance threshold. The per-level failure rates "
        "make the tendency concrete without overstating it: the ambiguous level fails "
        "more often than the clear level, but the sample is too small for the deviance "
        "test to rule out chance."
    ),
    new_code_cell(
        "clf = abc['classify_complaint']['by_factor']['clarity']\n"
        "print(f\"{'clarity level':16s} {'failed':>7s} {'n':>4s} {'rate':>7s}\")\n"
        "for level, cell in clf.items():\n"
        "    print(f\"{level:16s} {cell['failed']:7d} {cell['n']:4d} {cell['rate']:7.3f}\")"
    ),
    new_markdown_cell(
        "## The harness that produced the campaign\n"
        "\n"
        "The artifact is the output of the capstone testing harness, which runs the "
        "balanced suite against the agent and fits the attribution models. The companion "
        "reads the pinned result rather than re-running the campaign, but the harness "
        "class can be imported to confirm the provenance of the numbers above."
    ),
    new_code_cell(
        "from agentlab.testing.capstone_harness import CapstoneTestHarness\n"
        "print('harness class     :', CapstoneTestHarness.__name__)\n"
        "print('defined in module :', CapstoneTestHarness.__module__)"
    ),
    new_markdown_cell(
        "This is the capstone's realization of Chapter~11. The banking agent is exercised "
        "by a factor-balanced suite, each failure is attributed to a factor level by a "
        "logistic deviance test, and the campaign reports a null result: no input factor "
        "is implicated after correction, with clarity the nearest at an adjusted "
        "$p$-value near $0.96$. Decomposing by component then localizes the failures to "
        "`classify_complaint`, which is where subsequent work would be directed. "
        "Chapter~16 assembles this attribution logic into the governed testing workflow "
        "that decides whether the agent is fit to ship."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
