#!/usr/bin/env python
"""Build the gms-testing tutorial notebooks with nbformat.

One runnable notebook per chapter, each ending in a self-check assert. Notebooks
use the real knowlytix API where it exists (the GMS store and its primitives, the
calibrated gate, embedding SFT for the v- and u-channels, the GEODE build, the
suite catalogs) and small stand-ins only where no API applies. Figures are written
to book/figures/ for the chapters to include.

    python scripts/build_notebooks.py        # regenerate notebooks/
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB = Path(__file__).resolve().parents[2] / "notebooks"

_SETUP = """\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "code"))   # import gmstest from code/
import gmstest as gt
cat = gt.Catalog.load()
print(cat.summary())"""

# Headless matplotlib + a figures dir the book chapters \includegraphics from.
_FIG = """\
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, pathlib
FIGDIR = pathlib.Path("..") / "book" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)"""


def _nb(*cells):
    nb = new_notebook()
    nb.cells = list(cells)
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                  "language": "python"},
                   "language_info": {"name": "python"}}
    return nb


# Shared: load the real GEODE-built policy store (CPU), found by walking up to data/.
_STORE_SETUP = """\
import torch
from pathlib import Path
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore

def _find_store(name="gms_policy_store_geode"):
    for p in [Path.cwd(), *Path.cwd().parents]:
        c = p / "data" / name
        if c.exists():
            return c
    raise FileNotFoundError(name + " not found; run scripts/build_geode_rag_store.py")

STORE = _find_store()
store = GMSExpertStore(DocGMSConfig(store_path=str(STORE), ingest_mode="regex"),
                       device=torch.device("cpu"))
assert store.load(), "store failed to load"
print("loaded store:", len(store.adapter.relation_to_idx), "relations,",
      len(store.adapter.entity_to_idx), "entities")"""


# === 02 knowledge graphs and exact memory ==================================
def kg_nb():
    return _nb(
        new_markdown_cell(
            "# 2 - Knowledge Graphs and Exact Memory\n\n"
            "A knowledge graph stores knowledge as **triples** "
            "`(head, relation, tail)`. We build a tiny baseline graph in pure "
            "Python to show the standard operations and their one limit, then "
            "contrast it with the **real GMS store** (the rebuilt policy store), "
            "which makes membership a graded distance."),
        new_markdown_cell("## The baseline knowledge graph (pure Python)"),
        new_code_cell(
            "triples = [\n"
            "    ('overdraft', 'has_fee_amount',  '35'),\n"
            "    ('overdraft', 'governed_by',     'reg_e'),\n"
            "    ('dispute',   'has_window_days', '60'),\n"
            "]\n"
            "def query(h=None, r=None, t=None):\n"
            "    return [tr for tr in triples\n"
            "            if (h is None or tr[0]==h) and (r is None or tr[1]==r)\n"
            "            and (t is None or tr[2]==t)]\n"
            "print('about overdraft:', query(h='overdraft'))\n"
            "# membership is binary: the graph says yes/no, not 'how wrong'\n"
            "print(('overdraft','has_fee_amount','35') in triples,\n"
            "      ('overdraft','has_fee_amount','50') in triples)"),
        new_markdown_cell(
            "## The real GMS store: membership becomes a graded distance\n"
            "We load the trained policy store and call its real primitives. "
            "`query_triples` returns asserted edges exactly (as the baseline did); "
            "`score_triple` returns a geodesic distance --- low for a committed "
            "fact, high for a fabricated one --- the graded signal the baseline "
            "lacked."),
        new_code_cell(_STORE_SETUP),
        new_code_cell(
            "print('asserted edges:', store.query_triples(head='overdraft')[:3])\n"
            "d_true  = store.score_triple('overdraft', 'has_fee_amount', '35.0')\n"
            "d_false = store.score_triple('overdraft', 'has_fee_amount', '50.0')\n"
            "print(f'score_triple 35.0 = {d_true:.3f}  (committed)')\n"
            "print(f'score_triple 50.0 = {d_false:.3f}  (not committed)')"),
        new_markdown_cell("## Self-check"),
        new_code_cell(
            "assert query(h='overdraft')                              # baseline pattern query\n"
            "assert ('overdraft','has_fee_amount','35') in triples\n"
            "assert d_true < d_false                                  # GMS grades membership\n"
            "print('OK - knowledge graphs & exact memory')"),
    )


# === 03 GMS primitives, calibration, and embedding SFT =====================
def gms_nb():
    return _nb(
        new_markdown_cell(
            "# 3 - GMS Primitives, Calibration, and Embedding SFT\n\n"
            "This notebook runs the **real** GMS store: the geometric primitives, "
            "the **calibrated** decision gate, the two-channel embedding "
            "fine-tuning (v-space rotation, u-space contradiction) and its effect "
            "shown by PCA."),
        new_code_cell(_STORE_SETUP),
        new_markdown_cell(
            "## The primitives (real store)\n"
            "`score_triple` (plausibility as distance), `query_triples` (asserted "
            "edges) and `tension_energy` (contradiction)."),
        new_code_cell(
            "d35 = store.score_triple('overdraft', 'has_fee_amount', '35.0')\n"
            "d50 = store.score_triple('overdraft', 'has_fee_amount', '50.0')\n"
            "print(f'score_triple: 35.0={d35:.3f}  50.0={d50:.3f}')\n"
            "print('asserted:', store.query_triples(head='overdraft')[:2])\n"
            "print('tension(overdraft, udaap) =', round(store.tension_energy('overdraft','udaap'), 3))"),
        new_markdown_cell(
            "## The decision is made by a CALIBRATED gate\n"
            "A distance is not a decision. To call a fee value *grounded* or "
            "*fabricated* we need a boundary, and that boundary is the per-measure "
            "**geodesic** operating point that `GMSJudge.calibrate()` fits from the "
            "store's own facts and persists in `calibration.json` --- **not** a "
            "hand-picked value. A claim is grounded when its `score_triple` distance "
            "falls inside that calibrated band. (The retrieval gate in "
            "`rag_gate_calibration.json` is a different, more permissive gate; we "
            "meet it in the capstone.)"),
        new_code_cell(
            "import json\n"
            "from knowlytix.harness.testing.judge import GMSJudge\n"
            "# Per-measure operating points are fit by GMSJudge.calibrate() and persisted\n"
            "# in calibration.json. We read the persisted geodesic plausibility band ---\n"
            "# a calibrated distance with a known false-accept rate, never hand-picked.\n"
            "tau_d = json.loads((STORE / GMSJudge.CALIBRATION_JSON).read_text())['thresholds']['geodesic']\n"
            "def decide(h, r, t):\n"
            "    d = store.score_triple(h, r, t)\n"
            "    return round(d, 3), ('grounded' if d <= tau_d else 'fabricated')\n"
            "print('calibrated geodesic threshold (distance) =', round(tau_d, 3))\n"
            "print('overdraft 35.0 ->', decide('overdraft','has_fee_amount','35.0'))\n"
            "print('overdraft 50.0 ->', decide('overdraft','has_fee_amount','50.0'))"),
        new_code_cell(
            _FIG + "\n"
            "vals = [25, 30, 35, 45, 50, 60]        # tail values that are store entities\n"
            "ds = [store.score_triple('overdraft', 'has_fee_amount', f'{v}.0') for v in vals]\n"
            "fig, ax = plt.subplots(figsize=(4.6, 3))\n"
            "ax.plot(vals, ds, 'o-', color='#36c')\n"
            "ax.axhline(tau_d, ls='--', c='#c33', label=f'calibrated grounded band (d<={tau_d:.2f})')\n"
            "ax.axvline(35, ls=':', c='#2a8', label='committed (35)')\n"
            "ax.set_xlabel('tail value'); ax.set_ylabel('distance (score_triple)')\n"
            "ax.set_title('Plausibility as distance; the band is calibrated')\n"
            "ax.legend()\n"
            "fig.tight_layout(); fig.savefig(FIGDIR / 'fig_score_distance.png', dpi=150)\n"
            "from IPython.display import Image, display\n"
            "display(Image(filename=str(FIGDIR / 'fig_score_distance.png')))"),
        new_markdown_cell(
            "## Fine-tuning the two channels (real API)\n"
            "The v-space (semantic) is tuned by `finetune_embedding` in **rotation** "
            "mode; the u-space (logical) by `finetune_contradiction` in **full** "
            "mode (required --- a rotation preserves angles and cannot move tension). "
            "Both insert into the store's dual embedding via `EmbeddingConfig`."),
        new_code_cell(
            "import json, tempfile\n"
            "from knowlytix.embedding import (EmbeddingSFTConfig, finetune_embedding,\n"
            "                                 finetune_contradiction)\n"
            "from knowlytix.core.config import EmbeddingConfig\n"
            "from knowlytix.core.graph.embeddings import DualEmbedding\n"
            "from knowlytix.core.graph.encoders import init_dual_embeddings\n"
            "rows = [{'text': t, 'label': l} for t, l in [\n"
            "    ('overdraft fee', 'overdraft'), ('nsf charge', 'overdraft'), ('od charge', 'overdraft'),\n"
            "    ('dispute a charge', 'disputes'), ('chargeback', 'disputes'), ('unauthorized charge', 'disputes')]]\n"
            "lab = Path(tempfile.mktemp(suffix='.jsonl')); lab.write_text('\\n'.join(json.dumps(r) for r in rows))\n"
            "names = ['overdraft', 'disputes']\n"
            "v_ft = finetune_embedding(str(lab), EmbeddingSFTConfig(mode='rotation', rank=4, epochs=20, out_dim=64, device='cpu'),\n"
            "                          text_col='text', label_col='label')\n"
            "groups = {'has_fee_amount': ['overdraft fee', 'nsf charge'],\n"
            "          'has_interest_rate': ['apr', 'interest rate'],\n"
            "          'has_window_days': ['filing window', 'days to file']}\n"
            "u_ft = finetune_contradiction(groups,\n"
            "        EmbeddingSFTConfig(mode='full', objective='contradiction', rank=4, epochs=20, out_dim=64, device='cpu'))\n"
            "pv = Path(tempfile.mktemp(suffix='.pt')); pu = Path(tempfile.mktemp(suffix='.pt'))\n"
            "torch.save(v_ft.export_vectors(names), pv); torch.save(u_ft.export_vectors(names), pu)\n"
            "de = DualEmbedding(num_entities=len(names), d_v=64, d_u=64, m=64)\n"
            "init_dual_embeddings(de, names, EmbeddingConfig(v_vectors_path=str(pv), u_vectors_path=str(pu),\n"
            "                                                warm_start=False, freeze_base=True))\n"
            "row0_ok = torch.allclose(de.v_embed.weight.data[0], v_ft.export_vectors(names)['overdraft'], atol=1e-4)\n"
            "print('v mode:', v_ft.adapter.mode, '| u mode:', u_ft.adapter.mode,\n"
            "      '| inserted into GMS:', bool(row0_ok), '| v frozen:', not de.v_embed.weight.requires_grad)"),
        new_markdown_cell(
            "## Before vs after fine-tuning (PCA)\n"
            "The u-space contradiction objective separates logically-distinct "
            "attributes the raw encoder confuses. We embed phrasings of three "
            "attributes before (raw MiniLM) and after (contradiction-tuned), reduce "
            "each to 2-D by PCA and plot. (A v-space rotation preserves all "
            "distances, so it produces no PCA-visible separation change; its purpose "
            "is coordinate alignment --- turning the embedding into the orthonormal "
            "frame the GMS primitives are computed in --- not separation. The "
            "contradiction channel is where the geometry visibly moves.)"),
        new_code_cell(
            _FIG + "\n"
            "import numpy as np\n"
            "from knowlytix.core.graph.encoders import encode_texts\n"
            "ph = {'fee': ['overdraft fee', 'nsf charge', 'service fee'],\n"
            "      'interest': ['apr', 'interest rate', 'finance charge rate'],\n"
            "      'window': ['filing window', 'days to file', 'deadline in days']}\n"
            "texts = [w for g in ph for w in ph[g]]; labs = [g for g in ph for _ in ph[g]]\n"
            "before = encode_texts(texts, 'sentence-transformers/all-MiniLM-L6-v2', 'cpu').numpy()\n"
            "after = u_ft.encode(texts).numpy()\n"
            "def pca2(X):\n"
            "    Xc = X - X.mean(0); _, _, Vt = np.linalg.svd(Xc, full_matrices=False); return Xc @ Vt[:2].T\n"
            "cmap = {'fee': '#36c', 'interest': '#c33', 'window': '#2a8'}\n"
            "fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))\n"
            "for ax, (P, title) in zip(axes, [(pca2(before), 'before (raw encoder)'),\n"
            "                                 (pca2(after), 'after (u-space contradiction SFT)')]):\n"
            "    for g in ph:\n"
            "        idx = [i for i, l in enumerate(labs) if l == g]\n"
            "        ax.scatter(P[idx, 0], P[idx, 1], c=cmap[g], label=g, s=30)\n"
            "    ax.set_title(title); ax.legend(fontsize=7)\n"
            "fig.suptitle('Contradiction fine-tuning separates confusable attributes')\n"
            "fig.tight_layout(); fig.savefig(FIGDIR / 'fig_embed_pca.png', dpi=150)\n"
            "from IPython.display import Image, display\n"
            "display(Image(filename=str(FIGDIR / 'fig_embed_pca.png')))"),
        new_markdown_cell("## Self-check"),
        new_code_cell(
            "assert d35 < d50                                  # plausibility is graded\n"
            "assert decide('overdraft','has_fee_amount','35.0')[1] == 'grounded'    # calibrated\n"
            "assert decide('overdraft','has_fee_amount','50.0')[1] == 'fabricated'  # calibrated\n"
            "assert v_ft.adapter.mode == 'rotation' and u_ft.adapter.mode == 'full'\n"
            "assert row0_ok                                   # encoders inserted into GMS\n"
            "print('OK - primitives, calibration, embedding SFT')"),
    )


# === 04 GEODE: building the oracle =========================================
def geode_nb():
    return _nb(
        new_markdown_cell(
            "# 4 - GEODE: Building the Oracle\n\n"
            "GEODE turns a document into a trained, self-corrected store: ingest, a "
            "propose -> diagnose -> repair loop that drops geometrically-implausible "
            "triples, then training and calibration. In this notebook we **run** a "
            "real (small) build, inspect what it produced, then look at the full "
            "production store."),
        new_markdown_cell(
            "## Run GEODE on a small document\n"
            "A well-formed markdown table is enough. `build_rag_store` runs the "
            "`GeodeLoop` self-correction orchestrator and a geometry-supervised "
            "embedding loop, then trains the production store (small config here so "
            "it runs in seconds on CPU)."),
        new_code_cell(
            "import tempfile, torch\n"
            "from pathlib import Path\n"
            "from knowlytix.knowledge.geode import build_rag_store, make_default_trainer\n"
            "from knowlytix.knowledge.config import DocGMSConfig\n\n"
            "doc = Path(tempfile.mktemp(suffix='.md'))\n"
            "doc.write_text('''# Bank Fee Schedule\\n\\n## Fee Schedule\\n\\n"
            "| product | fee_amount | type |\\n"
            "| --- | --- | --- |\\n"
            "| overdraft | 35.00 | per_occurrence |\\n"
            "| wire_international | 45.00 | per_transaction |\\n"
            "| stop_payment | 30.00 | per_request |\\n''')\n\n"
            "cfg = DocGMSConfig(store_path=tempfile.mkdtemp(), ingest_mode='regex')\n"
            "cfg.train.epochs = 60                       # small, for a fast notebook build\n"
            "res = build_rag_store(doc, cfg, device=torch.device('cpu'),\n"
            "                      geode_trainer=make_default_trainer(torch.device('cpu'), epochs=60),\n"
            "                      max_iters=3)\n"
            "print(f'converged={res.converged} iters={res.iterations} '\n"
            "      f'entities={res.n_entities} triples={res.n_triples} enm={res.n_enm} '\n"
            "      f'corrections={len(res.corrections)}')"),
        new_markdown_cell(
            "The build returns a `GMSExpertStore` trained on the surviving triples. "
            "We query it directly --- the same primitives as Chapter 3, now on a "
            "store we just built."),
        new_code_cell(
            "print('asserted edges:', res.store.query_triples(head='overdraft'))\n"
            "print('score_triple(overdraft, has_fee_amount, 35.0) =',\n"
            "      round(res.store.score_triple('overdraft', 'has_fee_amount', '35.0'), 3))"),
        new_markdown_cell(
            "## What the loop does\n"
            "The `GeodeLoop` proposes the regex-extracted triples, diagnoses the ones "
            "that sit implausibly far on the manifold (or contradict an established "
            "fact) and repairs or drops them, recording every correction. On a clean "
            "table there is little to correct; on noisy prose the loop removes the "
            "extraction errors an oracle must not inherit. A geometry-supervised "
            "embedding loop also tunes the encoder to the document's vocabulary, so "
            "later queries bind colloquial wording to the right entity --- the "
            "ingestion improvement. (Mechanics live in the GMS substrate.)"),
        new_markdown_cell(
            "## The full production store\n"
            "The book's policy store is the same build at scale. It covers every "
            "policy, including the prose ones (filing windows, closure notice, "
            "escalation windows) that a fee-table-only ingest would miss."),
        new_code_cell(_STORE_SETUP),
        new_code_cell(
            "for r in ['has_fee_amount', 'has_filing_window_days',\n"
            "          'has_bank_initiated_notice_days', 'has_escalation_window_days']:\n"
            "    print(f'{r}:', store.query_triples(relation=r)[:1])"),
        new_markdown_cell("## Self-check"),
        new_code_cell(
            "assert res.converged and res.n_triples > 0          # we built a real store\n"
            "assert res.store.query_triples(head='overdraft')    # and can query it\n"
            "assert store.query_triples(relation='has_filing_window_days')  # prose policy captured\n"
            "print('OK - GEODE: building the oracle')"),
    )


# === 01 why average accuracy is not enough =================================


# === 01 why average accuracy is not enough =================================


# === 01 why average accuracy is not enough =================================
def foundations_nb():
    return _nb(
        new_markdown_cell(
            "# 1 - Why Average Accuracy Is Not Enough\n\n"
            "A benchmark reports one number; a deployment decision needs to know "
            "*which conditions* cause failure. A system can be accurate on average "
            "and concentrate its errors on one consequential condition. This "
            "notebook makes that concrete."),
        new_code_cell(
            "# 100 items under two conditions; failures concentrate on 'misleading'.\n"
            "items = [('clear', 1)] * 78 + [('clear', 0)] * 2 \\\n"
            "      + [('misleading', 1)] * 8 + [('misleading', 0)] * 12\n"
            "overall = sum(o for _, o in items) / len(items)\n"
            "by_cond = {c: sum(o for cc, o in items if cc == c) / sum(cc == c for cc, _ in items)\n"
            "           for c in ('clear', 'misleading')}\n"
            "print(f'overall accuracy : {overall:.2f}')\n"
            "print('by condition     :', {k: round(v, 2) for k, v in by_cond.items()})"),
        new_markdown_cell(
            "The aggregate (0.86) hides that the system fails 60% of the time on "
            "misleading phrasing. The rest of the book builds the designed "
            "experiment that surfaces and attributes exactly this."),
        new_markdown_cell("## Visualization"),
        new_code_cell(
            _FIG + "\n"
            "fig, ax = plt.subplots(figsize=(4, 3))\n"
            "labels = ['overall', 'clear', 'misleading']\n"
            "vals = [overall, by_cond['clear'], by_cond['misleading']]\n"
            "ax.bar(labels, vals, color=['#444', '#2a8', '#c33'])\n"
            "ax.axhline(overall, ls='--', c='#444', lw=0.8)\n"
            "ax.set_ylim(0, 1); ax.set_ylabel('accuracy')\n"
            "ax.set_title('Aggregate hides the weak condition')\n"
            "fig.tight_layout(); fig.savefig(FIGDIR / 'fig_avg_accuracy.png', dpi=150)\n"
            "print('wrote', FIGDIR / 'fig_avg_accuracy.png')"),
        new_markdown_cell("## Self-check"),
        new_code_cell(
            "assert overall > 0.8 and by_cond['misleading'] < 0.5   # average hides the weak condition\n"
            "print('OK - why average accuracy is not enough')"),
    )


# === 08 generating training data ===========================================
def training_nb():
    return _nb(
        new_markdown_cell(
            "# 8 - Generating Training Data\n\n"
            "The composed scenarios feed two SFT emitters: label-preserving "
            "classifier records, and grounded generative pairs validated against a "
            "golden contract. No store is needed when bases are user-supplied."),
        new_code_cell(_SETUP),
        new_code_cell(
            "items = gt.UserBaseSource([\n"
            "    {'query': 'I was overcharged on my overdraft', 'answer': 'complaint'},\n"
            "    {'query': 'How do I close my account?',        'answer': 'inquiry'},\n"
            "]).items()\n"
            "suite = gt.resolve(cat, ['exact_recall'], ['clarity', 'noise'], mode='cross')\n"
            "scns = gt.compose(suite, items, n_runs=4, seed=1)\n"
            "recs = gt.emit_classifier_sft(scns)\n"
            "print('classifier records:', len(recs), '| labels:', {r['label'] for r in recs})"),
        new_markdown_cell(
            "Draft pairs are validated against a golden contract (cite the "
            "keyword, hold the byte-exact number, avoid forbidden phrases); "
            "contract-failing pairs are dropped."),
        new_code_cell(
            "goldens = {'overdraft': {'citation_keywords': ['overdraft'],\n"
            "                         'required_numbers': ['35'],\n"
            "                         'forbidden_phrases': ['we will waive']}}\n"
            "pol = [gt.QAItem(qid='p0', query='charged $35 overdraft', answer=None,\n"
            "                 metadata={'policy_id': 'overdraft'})]\n"
            "psc = gt.compose(gt.resolve(cat, ['exact_recall'], ['clarity'], mode='cross'), pol, n_runs=2)\n"
            "good = lambda s: 'Your overdraft fee of $35 may be reversed once per year.'\n"
            "bad  = lambda s: 'We will waive your fee.'\n"
            "kept_g, drop_g = gt.emit_draft_sft(psc, good, goldens=goldens)\n"
            "kept_b, drop_b = gt.emit_draft_sft(psc, bad,  goldens=goldens)\n"
            "print('good kept/dropped:', len(kept_g), len(drop_g))\n"
            "print('bad  kept/dropped:', len(kept_b), len(drop_b))"),
        new_markdown_cell("## Self-check"),
        new_code_cell(
            "assert {r['label'] for r in recs} == {'complaint', 'inquiry'}   # labels preserved\n"
            "assert len(kept_g) == 2 and len(kept_b) == 0                    # contract enforced\n"
            "print('OK - generating training data')"),
    )


# === 09 agentic: what to test ==============================================
def agentic_nb():
    return _nb(
        new_markdown_cell(
            "# 9 - Agentic Systems and What to Test\n\n"
            "An agent is not a single answer but a **trajectory**: a sequence of "
            "tool calls, gate decisions and an outcome. The framework scores it at "
            "three levels, independently, so a run can pass one and fail another:\n\n"
            "- **outcome** --- did the final answer / labeled decisions match ground truth;\n"
            "- **trajectory structure** --- did the workflow run in order and escalate "
            "by the right path;\n"
            "- **process health** --- step count, tool failures, clean termination."),
        new_markdown_cell(
            "## The system under test is an interface\n"
            "A SUT is any `(query, context) -> answer` callable. Returning a "
            "`SUTResult` (answer, per-component predictions, trajectory, status) "
            "unlocks the trajectory and component levels. In production the SUT wraps "
            "the real agent --- `apps/complaint_sut.AgentSUT` maps a live "
            "`build_complaint_harness` trajectory to a `SUTResult`. Here we use a "
            "small deterministic stand-in so the notebook runs without a GPU; the "
            "scoring API is identical."),
        new_code_cell(_SETUP),
        new_code_cell(
            "from gmstest.evaluate import SUTResult, run, summary, weak_link\n"
            "workflow = ['classify', 'extract', 'search', 'flag', 'draft']\n\n"
            "def sut(query, context=''):                      # stand-in for AgentSUT\n"
            "    bad = '[clarity=Misleading]' in query        # this SUT slips on misleading framing\n"
            "    return SUTResult(answer='OK' if not bad else 'WRONG',\n"
            "                     components={'classification': 'complaint' if not bad else 'other'},\n"
            "                     trajectory=[{'tool': t, 'ok': True} for t in workflow],\n"
            "                     status='ok' if not bad else 'failed')"),
        new_markdown_cell(
            "`evaluate.run` executes each scenario through the SUT and returns one "
            "row per scenario; `summary` aggregates outcome accuracy and workflow "
            "adherence."),
        new_code_cell(
            "items = [gt.QAItem(qid=f's{i}', query='[clarity=Clear] msg', answer='OK',\n"
            "                   components={'expected_classification': 'complaint',\n"
            "                               'expected_escalation': False}) for i in range(3)]\n"
            "suite = gt.resolve(cat, ['exact_recall'], ['clarity'], mode='cross')\n"
            "scns = gt.compose(suite, items, n_runs=2, seed=1)\n"
            "rows = run(scns, sut, workflow_order=workflow)\n"
            "print(summary(rows))"),
        new_markdown_cell("## Self-check"),
        new_code_cell(
            "assert summary(rows)['workflow_adherence'] == 1.0      # ran in order every time\n"
            "assert 'accuracy' in summary(rows)\n"
            "print('OK - agentic: what to test')"),
    )


# === 10 logistic attribution ===============================================
def analysis_nb():
    return _nb(
        new_markdown_cell(
            "# 10 - Identifying Weakness: Logistic Attribution\n\n"
            "Given a table of binary outcomes and factor levels, `evaluate.attribute` "
            "fits a logistic analysis of deviance per factor and corrects across "
            "the family of tests. Here one factor (clarity) drives failure."),
        new_code_cell(_SETUP),
        new_code_cell(
            "import random\n"
            "from gmstest.evaluate import attribute\n"
            "rng = random.Random(7)\n"
            "suite = gt.resolve(cat, ['multi_hop'], ['clarity', 'entity_aliasing', 'reasoning_cue'], mode='embedded')\n"
            "scns = gt.compose(suite, [gt.QAItem(qid=f's{i}', query='m', answer='ok') for i in range(6)],\n"
            "                  n_runs=120, seed=7)\n"
            "rows = [{'correct': 0 if rng.random() < (0.6 if s.factor_levels['clarity'] == 'Misleading' else 0.15) else 1,\n"
            "         **{f'f_{k}': v for k, v in s.factor_levels.items()}} for s in scns]\n"
            "tab = attribute(rows, suite.factor_names)\n"
            "for t in tab:\n"
            "    print(f\"{t['factor']:16s} p_adj={t['p_value_adj']:.4f} R2={t['pseudo_r2']:.3f} sig={t['significant_adj']}\")"),
        new_markdown_cell("## Visualization"),
        new_code_cell(
            _FIG + "\n"
            "names = [t['factor'] for t in tab]\n"
            "r2 = [t['pseudo_r2'] for t in tab]\n"
            "colors = ['#c33' if t['significant_adj'] else '#999' for t in tab]\n"
            "fig, ax = plt.subplots(figsize=(4.6, 3))\n"
            "ax.bar(names, r2, color=colors)\n"
            "ax.set_ylabel('pseudo-$R^2$')\n"
            "ax.set_title('Factor effect (red = significant after FDR)')\n"
            "plt.setp(ax.get_xticklabels(), rotation=20, ha='right')\n"
            "fig.tight_layout(); fig.savefig(FIGDIR / 'fig_attribution.png', dpi=150)\n"
            "print('wrote', FIGDIR / 'fig_attribution.png')"),
        new_markdown_cell("## Self-check"),
        new_code_cell(
            "clar = [t for t in tab if t['factor'] == 'clarity'][0]\n"
            "assert clar['significant_adj']                          # clarity drives failure\n"
            "print('OK - logistic attribution')"),
    )


# === 11 resilience under tool faults =======================================
def resilience_nb():
    return _nb(
        new_markdown_cell(
            "# 11 - Resilience Under Tool Faults\n\n"
            "One check beyond correctness: does the agent **fail loud** when a tool "
            "fails? This notebook illustrates fault detection with a toy model; the "
            "real `ToolGateway` call is shown for reference."),
        new_markdown_cell(
            "## Fault injection: detect, do not silently propagate\n"
            "A governed agent must escalate when a tool fails. Detection rate is "
            "the fraction of faulted runs that escalate rather than proceed."),
        new_code_cell(
            "def agent(tool_ok):\n"
            "    # a governed agent escalates the moment a tool result is unusable\n"
            "    return 'escalated' if not tool_ok else 'answered'\n"
            "runs = [agent(tool_ok=False) for _ in range(8)]    # tool faulted on every run\n"
            "detected = sum(r == 'escalated' for r in runs)\n"
            "print(f'detected {detected}/{len(runs)}  silent {len(runs)-detected}')"),
        new_markdown_cell(
            "## The fault taxonomy\n"
            "Faults differ in difficulty. An error or timeout is easy: the tool "
            "returns nothing usable and the agent escalates. Stale data is harder. "
            "A plausible-but-wrong structured result is hardest, and is caught only "
            "downstream by a substrate verifier (the exact-register check on a "
            "number, the geometric check on a claim)."),
        new_markdown_cell(
            "## The real call (reference)\n"
            "```python\n"
            "from knowlytix.harness.testing import ToolGateway, FaultProfile\n"
            "# install a gateway that errors one tool, run the suite, record\n"
            "# detected (escalated) vs silent per tool.\n"
            "```"),
        new_markdown_cell("## Self-check"),
        new_code_cell(
            "assert detected == len(runs)                       # fail loud on every faulted run\n"
            "print('OK - resilience')"),
    )


# === 12 capstone: testing a governed agent end to end =======================
# Ch12 is built per-section (12_3..12_8) by scripts/build_nb_12_*.py, each reading
# the same pinned campaign. There is no single 12_capstone notebook.


# === 01 base catalog =======================================================
def base_nb():
    return _nb(
        new_markdown_cell(
            "# 5 - The Base Taxonomy: *what* is asked, *what* is true\n\n"
            "A **base** entry defines a question type mined from a GMS store, whose "
            "ground truth is computed by a GMS primitive (and is therefore provably "
            "correct). Base entries are domain-agnostic templates; pick the families "
            "and categories your domain supports.\n\n"
            "This complements the **enrichment** catalog (notebook 2), which only "
            "changes *how* a question is presented, never the answer."),
        new_code_cell(_SETUP),
        new_markdown_cell(
            "## The five families\n"
            "The 18 base categories are the deduplicated union of the JASA-10, the "
            "13-category testing taxonomy, and the 6 knowlytix generators."),
        new_code_cell(
            "for fam, names in cat.families.items():\n"
            "    print(f'{fam:24s} {len(names):2d}  {\", \".join(names)}')"),
        new_markdown_cell(
            "## One entry in detail\n"
            "Each base carries the graph capability it `requires`, the `generator` "
            "that mines it (or `None` if not yet built), its `answer_type`, the GMS "
            "primitive that defines `ground_truth`, and a `status`."),
        new_code_cell(
            "b = cat.bases['exact_recall']\n"
            "for f in ('family','requires','generator','answer_type','ground_truth','status','source'):\n"
            "    print(f'{f:14s} {getattr(b, f)}')"),
        new_markdown_cell(
            "## Three ways to supply base questions\n"
            "**`UserBaseSource`** - bring your own `(query, answer)` pairs. No GMS is "
            "used: your answer *is* the ground truth, and queries are not expanded by "
            "a store. This is the lightweight path for augmenting an existing labeled "
            "set."),
        new_code_cell(
            "user = gt.UserBaseSource([\n"
            "    {'query': 'What is the overdraft fee?', 'answer': '35'},\n"
            "    {'query': 'How long to dispute a charge?', 'answer': '60 days'},\n"
            "])\n"
            "items = user.items()\n"
            "for it in items:\n"
            "    print(it.qid, '|', it.query, '->', it.answer, '| base =', it.base)"),
        new_markdown_cell(
            "**`SeedCaseSource`** - hand-labeled cases carrying *per-component* ground "
            "truth (classification, product, issue, policy, escalation). This is what "
            "trajectory + weak-link testing of an agent consumes."),
        new_code_cell(
            "cases = [{'id': 'case-001', 'message': 'I was charged a $35 overdraft fee and want it removed.',\n"
            "          'expected_classification': 'complaint', 'expected_escalation': False,\n"
            "          'expected_product': 'checking_account', 'expected_issue': 'overdraft_fee'}]\n"
            "it = gt.SeedCaseSource(cases).items()[0]\n"
            "print('qid       :', it.qid)\n"
            "print('query     :', it.query)\n"
            "print('components:', it.components)"),
        new_markdown_cell(
            "**`CatalogBaseSource`** - mine a trained GMS store with the selected "
            "categories' generators (the provably-correct path). We run it on the "
            "real policy store; each mined question carries a graph-derived ground "
            "truth."),
        new_code_cell(_STORE_SETUP),
        new_code_cell(
            "suite_cat = gt.resolve(cat, ['exact_recall', 'counting', 'contradiction'], [], mode='cross')\n"
            "cat_items = gt.CatalogBaseSource(store, max_per_category=3).items(suite_cat)\n"
            "print(len(cat_items), 'questions mined from the store; examples:')\n"
            "for it in cat_items[:3]:\n"
            "    print(' ', it.base, '|', it.query[:56], '-> ground truth:', it.answer)"),
        new_markdown_cell("## Self-check"),
        new_code_cell(
            "assert len(cat.bases) == 18 and len(cat.families) == 5\n"
            "assert cat.bases['exact_recall'].ground_truth == 'lookup_enm'\n"
            "assert items[0].base is None              # user-supplied carries no catalog base\n"
            "assert cat_items and cat_items[0].base in cat.bases   # mined from the real store\n"
            "print('OK - base catalog')"),
    )


# === 02 enrichment catalog =================================================
def factor_nb():
    return _nb(
        new_markdown_cell(
            "# 6 - The Enrichment Design Space: *how* a question is presented\n\n"
            "A **factor** changes the surface and context of a base question while "
            "leaving the ground truth invariant. Every factor is `gt_invariant: true` "
            "by construction - anything that could change the answer belongs in the "
            "base catalog, not here."),
        new_code_cell(_SETUP),
        new_markdown_cell(
            "## 40 factors in 10 groups, all ground-truth invariant"),
        new_code_cell(
            "from collections import Counter\n"
            "print('factors:', len(cat.factors))\n"
            "print('all gt_invariant:', all(f.gt_invariant for f in cat.factors.values()))\n"
            "print(Counter(f.section for f in cat.factors.values()))"),
        new_markdown_cell(
            "## `applies_to`: enrichment enriches the *base*\n"
            "Each factor declares which base families / answer types it is meaningful "
            "for. When you resolve a suite, factors that match no selected base are "
            "dropped automatically - so a numeric-format factor never decorates a "
            "set-valued cross-reference question."),
        new_code_cell(
            "# cross_reference answers set_str -> numeric_format does NOT apply\n"
            "s1 = gt.resolve(cat, ['cross_reference'], ['numeric_format','clarity'], mode='embedded')\n"
            "print('cross_reference  kept :', s1.factor_names)\n"
            "print('cross_reference  drop :', s1.dropped_factors)\n"
            "# exact_recall answers float -> numeric_format applies\n"
            "s2 = gt.resolve(cat, ['exact_recall'], ['numeric_format','clarity'], mode='embedded')\n"
            "print('exact_recall     kept :', s2.factor_names)"),
        new_markdown_cell(
            "## Factor groups are reusable bundles\n"
            "The catalog ships the same groups as the knowlytix factor catalog "
            "(`quick_screen`, `comprehensive`, `full`, ...)."),
        new_code_cell(
            "for g in ('quick_screen','adversarial','comprehensive'):\n"
            "    print(f'{g:14s} {cat.factor_groups[g]}')"),
        new_markdown_cell(
            "## Application level vocabulary\n"
            "The catalog is domain-neutral. An application supplies its own level "
            "labels via `level_overrides` (kept out of the general catalog). For "
            "example the complaint agent renames `entity_aliasing` levels to "
            "`canonical / alias / abbreviated` - applied at `compose` time, shown in "
            "notebook 3."),
        new_markdown_cell("## Self-check"),
        new_code_cell(
            "assert len(cat.factors) == 40\n"
            "assert all(f.gt_invariant for f in cat.factors.values())\n"
            "assert 'numeric_format' not in s1.factor_names   # dropped for set_str base\n"
            "assert 'numeric_format' in s2.factor_names        # kept for float base\n"
            "print('OK - enrichment catalog')"),
    )


# === 03 profiles & scenarios ===============================================
def profile_nb():
    return _nb(
        new_markdown_cell(
            "# 7 - Profiles, Scenarios, and the Two Consumers\n\n"
            "A **profile** pairs a base selection with a factor selection and a "
            "scenario mode. Resolving a profile expands families/groups and filters "
            "factors by `applies_to`. Composing turns the resolved suite + base items "
            "into concrete **scenarios**, which feed either consumer: SFT data "
            "generation (Ch15) or agent testing (Ch16)."),
        new_code_cell(_SETUP),
        new_markdown_cell("## Profiles ship as pick-and-choose bundles"),
        new_code_cell(
            "for name, p in cat.profiles.items():\n"
            "    print(f'{name:20s} mode={p.mode:9s} bases={p.bases} factors={p.factors[:3]}...')"),
        new_code_cell(
            "suite = gt.resolve_profile(cat, 'numeric_screen')\n"
            "print(suite.summary())"),
        new_markdown_cell(
            "## Two scenario modes\n"
            "**cross** (Option 1): base x design matrix - every base seen under every "
            "design row, size = bases * n_runs.\n\n"
            "**embedded** (Option 2): the base is a *balanced blocking* factor - every "
            "base exercised equally, size = n_runs. (At small n this avoids the "
            "joint phi_p design starving a high-cardinality base; see the JASA paper.)"),
        new_code_cell(
            "from gmstest.compose import graphdoe_design   # real knowlytix Sobol+refine designs\n"
            "from collections import Counter\n"
            "items = [gt.QAItem(qid=f'q{i}', query=f'metric {i}?', answer=str(i)) for i in range(5)]\n"
            "cross = gt.compose(gt.resolve(cat, ['exact_recall'], ['clarity'], mode='cross'),\n"
            "                   items, n_runs=8, seed=1, design_fn=graphdoe_design)\n"
            "emb   = gt.compose(gt.resolve(cat, ['exact_recall'], ['clarity'], mode='embedded'),\n"
            "                   items, n_runs=8, seed=1, design_fn=graphdoe_design)\n"
            "print('cross   ', len(cross), 'scenarios')\n"
            "print('embedded', len(emb), 'scenarios; base balance', dict(Counter(s.base.qid for s in emb)))"),
        new_markdown_cell(
            "## The user-supplied variant (no GMS)\n"
            "Bring your own `(query, answer)` set, enrich it with presentation "
            "factors, and you never touch a store. Ground truth is your supplied "
            "answer; the queries are varied for coverage, not expanded by a graph."),
        new_code_cell(
            "user_items = gt.UserBaseSource([\n"
            "    {'query': 'Was I overcharged on my overdraft?', 'answer': 'complaint'},\n"
            "    {'query': 'How do I close my account?',         'answer': 'inquiry'},\n"
            "]).items()\n"
            "u_suite = gt.resolve(cat, ['exact_recall'], ['clarity','noise'], mode='cross')\n"
            "u_scn = gt.compose(u_suite, user_items, n_runs=4, seed=2)\n"
            "print(len(u_scn), 'scenarios from user pairs; ground truth preserved:',\n"
            "      all(s.base.answer in ('complaint','inquiry') for s in u_scn))"),
        new_markdown_cell(
            "## Consumer A - SFT data (Chapter 15)\n"
            "`emit_classifier_sft` produces label-preserving `(message, label)` "
            "records; `emit_draft_sft` produces grounded `(user, assistant)` pairs "
            "validated against a golden contract."),
        new_code_cell(
            "recs = gt.emit_classifier_sft(u_scn)\n"
            "print('records:', len(recs), '| keys:', sorted(recs[0]))\n"
            "print('labels preserved:', {r['label'] for r in recs})"),
        new_markdown_cell(
            "## Consumer B - agent testing (Chapter 16)\n"
            "`evaluate.run` sends each scenario through a System Under Test - any "
            "`(query, context) -> answer` callable - and scores outcome, trajectory "
            "and process. Here a trivial SUT stands in for the real agent (which "
            "`apps/complaint_sut.AgentSUT` wraps)."),
        new_code_cell(
            "from gmstest.evaluate import run, summary\n"
            "rows = run(u_scn, lambda q, context='': 'complaint' if 'overcharged' in q else 'inquiry')\n"
            "print(summary(rows))"),
        new_markdown_cell(
            "## Visualization: coverage of a space-filling design\n"
            "The design generator favors Sobol+refine because it spreads points "
            "evenly. Random points clump and leave gaps; Sobol fills the space."),
        new_code_cell(
            _FIG + "\n"
            "import numpy as np\n"
            "from scipy.stats import qmc\n"
            "rand = np.random.default_rng(0).random((64, 2))\n"
            "sob = qmc.Sobol(d=2, scramble=True, seed=0).random_base2(m=6)   # 64 points\n"
            "fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))\n"
            "for ax, (pts, title) in zip(axes, [(rand, 'random'), (sob, 'Sobol (space-filling)')]):\n"
            "    ax.scatter(pts[:, 0], pts[:, 1], s=14, color='#36c')\n"
            "    ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')\n"
            "fig.suptitle('64 points over two factors')\n"
            "fig.tight_layout(); fig.savefig(FIGDIR / 'fig_design_coverage.png', dpi=150)\n"
            "print('wrote', FIGDIR / 'fig_design_coverage.png')"),
        new_markdown_cell("## Self-check"),
        new_code_cell(
            "assert len(cross) == 5 * 8 and len(emb) == 8        # cross = bases*runs, embedded = runs\n"
            "assert set(recs[0]) == {'message', 'label', '_seed', '_factors'}\n"
            "assert all(s.base.answer in ('complaint', 'inquiry') for s in u_scn)  # GT preserved, no GMS\n"
            "print('OK - profiles & scenarios')"),
    )


def main():
    NB.mkdir(exist_ok=True)
    for name, builder in (("01_why_average_accuracy", foundations_nb),
                          ("02_knowledge_graphs", kg_nb),
                          ("03_gms_primitives", gms_nb),
                          ("04_geode", geode_nb),
                          ("05_base_taxonomy", base_nb),
                          ("06_enrichment_design_space", factor_nb),
                          ("07_design_and_composition", profile_nb),
                          ("08_generating_training_data", training_nb),
                          ("09_agentic", agentic_nb),
                          ("10_logistic_attribution", analysis_nb),
                          ("11_resilience", resilience_nb)):
        # Ch12 is built per-section (12_3..12_8) by scripts/build_nb_12_*.py,
        # not as a single 12_capstone notebook.
        path = NB / f"{name}.ipynb"
        nbf.write(builder(), path)
        print("wrote", path)
    import sys
    if "--execute" in sys.argv:
        from nbclient import NotebookClient
        for path in sorted(NB.glob("*.ipynb")):
            nb = nbf.read(path, as_version=4)
            NotebookClient(nb, timeout=600, kernel_name="python3",
                           resources={"metadata": {"path": str(NB)}}).execute()
            nbf.write(nb, path)
            print("executed", path)


if __name__ == "__main__":
    main()
