#!/usr/bin/env python
"""Builder for notebooks/demos/demo_5_gms_from_md.ipynb.

Teaching demo 5: build a GMS store from a markdown document with the GEODE loop,
show the extracted triples, visualize the trained graph as geodesic arcs on the
embedding sphere, and query it.

The domain is the family running example from the KG-embedding book (Ann, Bob,
Carol, Dave, Eve, Frank; spouse, parentOf, siblingOf), chosen because it is small
enough to check by eye yet exhibits the two patterns the notebook demonstrates:
a many-to-many relation (parentOf) and a multi-hop chain (grandparent = parentOf
composed with parentOf).

Every call is real knowlytix: `build_rag_store` runs the GEODE loop and trains the
production GMS; `kg_sphere` renders the geodesics; `GMSExpertStore.query_triples`
and `score_triple` answer the queries. The notebook is executed so the figure and
outputs are present. CPU-only build (no LLM, no Qwen): `ingest_mode="regex"` and
`llm=None`, so the GEODE extractor is the deterministic table parser.
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUT = (Path(__file__).resolve().parents[2]
       / "notebooks" / "demos" / "demo_5_gms_from_md.ipynb")

# The source document: the KG-embedding book's family, written as three small
# markdown tables. The regex ingester reads each row as `entity has_<column> value`,
# so the column headers (spouse, parent_of, sibling_of) become the relation names.
FAMILY_MD = '''# Family Knowledge Base

A tiny knowledge base: six people across three generations. Ann and Bob are
married and are the parents of Carol and Dave; Carol and Eve are married and are
the parents of Frank.

## Marriages

| Person | spouse |
| --- | --- |
| Ann | Bob |
| Carol | Eve |

## Parenthood

| Parent | parent_of |
| --- | --- |
| Ann | Carol |
| Ann | Dave |
| Bob | Carol |
| Bob | Dave |
| Carol | Frank |
| Eve | Frank |

## Siblings

| Person | sibling_of |
| --- | --- |
| Carol | Dave |
| Dave | Carol |
'''

cells = [
    new_markdown_cell(
        "# Demo 5 --- A GMS store from a markdown document\n"
        "\n"
        "This notebook builds a knowledge store from a single `.md` file, end to end:\n"
        "\n"
        "1. `build_rag_store` runs the **GEODE loop**, which reads triples from the "
        "document, self-corrects them, and trains a geometric memory (a GMS).\n"
        "2. We read back the **extracted triples**.\n"
        "3. We draw the trained graph as **geodesic arcs on the embedding sphere**.\n"
        "4. We **query** it --- a many-to-many relation and a multi-hop chain.\n"
        "\n"
        "The domain is the family running example from the KG-embedding book: six "
        "people across three generations. It is small enough to check by eye, and it "
        "carries the two patterns worth showing --- `parentOf` is **many-to-many** "
        "(a parent has several children, a child has several parents), and "
        "`grandparent` is a **multi-hop** composition of `parentOf` with itself."
    ),
    new_markdown_cell(
        "## The source document\n"
        "\n"
        "The input is ordinary markdown. The regex ingester reads each table row as "
        "`entity has_<column> value`, so the column headers (`spouse`, `parent_of`, "
        "`sibling_of`) become the relation names. We write the file next to the "
        "other demo data and display it."
    ),
    new_code_cell(
        "import warnings\n"
        "from pathlib import Path\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "FAMILY_MD = " + repr(FAMILY_MD) + "\n"
        "\n"
        "# Resolve the repo data dir (the dir that already holds the demo data).\n"
        "root = next((c for c in (Path('.'), Path('..'), Path('../..'),\n"
        "                         Path('../../code'), Path('../code'))\n"
        "             if (c / 'data').is_dir()), Path('.'))\n"
        "DATA = root / 'data'\n"
        "MD = DATA / 'family_kb.md'\n"
        "STORE = DATA / 'gms_family_store'\n"
        "\n"
        "MD.write_text(FAMILY_MD)\n"
        "print(MD.read_text())"
    ),
    new_markdown_cell(
        "## Build the store with the GEODE loop\n"
        "\n"
        "`build_rag_store` does everything in one call: it ingests the markdown into "
        "triples, runs the GEODE self-correction loop (each pass trains a small GMS, "
        "flags triples the geometry finds inconsistent, and integrates the survivors), "
        "then trains the production GMS on the final clean graph and saves it to disk.\n"
        "\n"
        "The geometry is deliberately small (`d_v=d_u=32`) because the graph is small; "
        "`cap` admissibility with a low `n_boundary` suits six entities. `ingest_mode="
        "'regex'` and `llm=None` keep the run deterministic and offline --- no LLM is "
        "used to read the document."
    ),
    new_code_cell(
        "import torch\n"
        "from knowlytix.core.config import GeometryConfig, TrainConfig, CapLossConfig\n"
        "from knowlytix.knowledge.config import DocGMSConfig\n"
        "from knowlytix.knowledge.geode.rag import build_rag_store\n"
        "from knowlytix.knowledge.geode.loop import make_default_trainer\n"
        "\n"
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "config = DocGMSConfig(\n"
        "    store_path=str(STORE),\n"
        "    ingest_mode='regex',          # deterministic table parser, no LLM\n"
        "    loss_mode='cap',              # relation-conditioned spherical-cap admissibility\n"
        "    geometry=GeometryConfig(d_v=32, d_u=32, m=16, d=16),\n"
        "    cap=CapLossConfig(n_boundary=2),   # few entities -> few boundary negatives\n"
        "    train=TrainConfig(epochs=250, batch_size=32, neg_samples=4,\n"
        "                      lr=5e-3, lr_riemannian=2e-3),\n"
        ")\n"
        "\n"
        "res = build_rag_store(\n"
        "    str(MD), config, device=device,\n"
        "    geode_trainer=make_default_trainer(device, epochs=60),\n"
        "    llm=None,\n"
        ")\n"
        "store = res.store\n"
        "print(f'converged={res.converged}  iterations={res.iterations}')\n"
        "print(f'entities={res.n_entities}  relations={len(store.adapter.relation_to_idx)}'\n"
        "      f'  triples={res.n_triples}')\n"
        "print(f'GEODE corrections={len(res.corrections)}  '\n"
        "      f'anchor_violations={len(res.anchor_violations)}')"
    ),
    new_markdown_cell(
        "Six entities, three relations, and a handful of triples. The GEODE loop drops "
        "the organizational `in_section` edges the ingester emits (they carry no fact), "
        "so what remains is the family graph itself. The clean corpus has no "
        "contradictions, so GEODE reports zero corrections."
    ),
    new_markdown_cell(
        "## The extracted triples\n"
        "\n"
        "These are the facts the store was trained on --- read straight back from the "
        "store, grouped by relation. `has_parent_of` is the many-to-many relation: Ann "
        "appears with two children, and Frank appears with two parents."
    ),
    new_code_cell(
        "from collections import defaultdict\n"
        "\n"
        "by_rel = defaultdict(list)\n"
        "for h, r, t in sorted(store.query_triples()):\n"
        "    by_rel[r].append((h, t))\n"
        "\n"
        "for r in sorted(by_rel):\n"
        "    print(r)\n"
        "    for h, t in by_rel[r]:\n"
        "        print(f'    {h:6s} -> {t}')"
    ),
    new_markdown_cell(
        "## Visualize: geodesics on the embedding sphere\n"
        "\n"
        "Training places each entity on a sphere in the learned embedding space. We "
        "reduce the entity embeddings to three dimensions with PCA, project onto the "
        "unit sphere, and draw each triple as a **geodesic arc** (a great-circle "
        "segment) from head to tail, colored by relation. For a rotation-operator "
        "embedding the head-to-tail path of a relation is exactly such a geodesic. "
        "`kg_sphere.py` sits beside the chapter notebooks; we add that directory to "
        "the path and load the store straight from disk."
    ),
    new_code_cell(
        "import sys\n"
        "# kg_sphere.py lives in the notebooks/ dir; find it walking up from here.\n"
        "for up in (Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent):\n"
        "    if (up / 'kg_sphere.py').exists():\n"
        "        sys.path.insert(0, str(up))\n"
        "        break\n"
        "import kg_sphere as K\n"
        "\n"
        "kg = K.load_gms_store(str(STORE), source='v')   # entity embeddings + triples\n"
        "print(f'{len(kg.labels)} entities, {len(kg.triples)} triples, '\n"
        "      f'relations={kg.relations}')\n"
        "K.visualize(kg, show=False)"
    ),
    new_markdown_cell(
        "## Query 1 --- a many-to-many relation\n"
        "\n"
        "`query_triples` pattern-matches the graph: fix the head to read a person's "
        "children, fix the tail to read a person's parents. `parent_of` is "
        "many-to-many, so both directions return more than one answer."
    ),
    new_code_cell(
        "def children_of(name):\n"
        "    return [t for h, r, t in store.query_triples(head=name, relation='has_parent_of')]\n"
        "\n"
        "def parents_of(name):\n"
        "    return [h for h, r, t in store.query_triples(relation='has_parent_of', tail=name)]\n"
        "\n"
        "print('children of ann :', children_of('ann'))    # one parent, several children\n"
        "print('parents of carol:', parents_of('carol'))   # one child, several parents\n"
        "print('parents of frank:', parents_of('frank'))\n"
        "print('spouse of ann   :', [t for h, r, t in\n"
        "                            store.query_triples(head='ann', relation='has_spouse')])"
    ),
    new_markdown_cell(
        "## Query 2 --- a multi-hop chain\n"
        "\n"
        "`grandparent` is not a stored relation. It is `parent_of` composed with "
        "`parent_of`: the grandparents of Frank are the parents of Frank's parents. We "
        "answer it by walking two hops over the graph."
    ),
    new_code_cell(
        "def grandparents_of(name):\n"
        "    out = set()\n"
        "    for p in parents_of(name):\n"
        "        out.update(parents_of(p))\n"
        "    return sorted(out)\n"
        "\n"
        "print('parents of frank      :', parents_of('frank'))\n"
        "print('grandparents of frank :', grandparents_of('frank'))"
    ),
    new_markdown_cell(
        "## Calibrate the decision gates\n"
        "\n"
        "The geometry scores a fact by its geodesic distance to the relation's cap "
        "center (lower is more plausible), but a decision needs an operating point, and "
        "no gate should read a hardcoded default. `GMSJudge.calibrate` fits one from the "
        "store's own graph: for each channel it takes the real triples as positives and "
        "corrupted ones as negatives and fits the cut. It calibrates every channel --- "
        "geodesic plausibility, two-hop path transport, u-space tension (contradiction), "
        "and holonomy (path consistency) --- and reports the accuracy of each.\n"
        "\n"
        "Calibration also decides *whether a channel is usable at all*. On this graph the "
        "geodesic channel separates cleanly, but the tension channel comes back "
        "`degenerate`: a family declares no oppositions (`opposite_of`) and no functional "
        "relations, so there is nothing for a contradiction cut to fit, and the gate "
        "abstains rather than guess. We persist the operating point next to the store."
    ),
    new_code_cell(
        "import json\n"
        "from knowlytix.harness.testing.judge import GMSJudge\n"
        "from knowlytix.harness.governance.reasoner import (\n"
        "    CalibratedThresholds, GeometricReasoner)\n"
        "\n"
        "judge = GMSJudge(store)\n"
        "judge.calibrate()                       # prints the per-channel calibration table\n"
        "thresholds = CalibratedThresholds.from_judge(judge)\n"
        "reasoner = GeometricReasoner(store, thresholds, device=device)\n"
        "\n"
        "# Persist the calibrated operating point (no gate reads a default).\n"
        "(STORE / 'threshold_calibration.json').write_text(\n"
        "    json.dumps(judge._thresholds, indent=2, default=str))\n"
        "print('\\naccept threshold (geodesic) =', round(thresholds.tau_plausibility, 3))\n"
        "print('tension channel            =', thresholds.tension_status)"
    ),
    new_markdown_cell(
        "## A query that is not plausible\n"
        "\n"
        "`is_plausible` scores a triple and compares it to the calibrated accept "
        "threshold. A true edge passes; a fact the graph does not support is rejected. "
        "Asking whether Ann is the *parent* of Frank fails --- she is his grandparent, "
        "two `parent_of` hops away, so the direct edge lands outside the cap."
    ),
    new_code_cell(
        "def check(h, rel, t):\n"
        "    ok, d = reasoner.is_plausible(h, rel, t)\n"
        "    verdict = 'PLAUSIBLE' if ok else 'rejected'\n"
        "    print(f'  {h:5s} -{rel[4:]:9s}-> {t:5s}  distance={d:.3f}  {verdict}')\n"
        "\n"
        "print(f'accept threshold = {thresholds.tau_plausibility:.3f}\\n')\n"
        "check('ann', 'has_parent_of', 'carol')   # true edge\n"
        "check('ann', 'has_parent_of', 'frank')   # grandparent asked as parent -> rejected"
    ),
    new_markdown_cell(
        "## A contradiction: flip parent and child\n"
        "\n"
        "`parent_of` is antisymmetric: if Ann is Carol's parent, Carol is not Ann's "
        "parent. So flipping the head and tail of a true edge produces a contradiction, "
        "and the calibrated gate must reject the flipped triple while accepting the "
        "original. We take Ann's children (read above) and check both directions."
    ),
    new_code_cell(
        "for child in children_of('ann'):\n"
        "    ok_true, d_true = reasoner.is_plausible('ann', 'has_parent_of', child)\n"
        "    ok_flip, d_flip = reasoner.is_plausible(child, 'has_parent_of', 'ann')\n"
        "    print(f'  ann parent_of {child:5s}: {\"ok\" if ok_true else \"no\":3s} (d={d_true:.3f})'\n"
        "          f'   |   flip {child} parent_of ann: '\n"
        "          f'{\"ok\" if ok_flip else \"CONTRADICTION\"} (d={d_flip:.3f})')\n"
        "    assert ok_true and not ok_flip\n"
        "\n"
        "# The dedicated tension channel is degenerate here (no declared opposition),\n"
        "# so this antisymmetry contradiction is caught by the calibrated plausibility\n"
        "# gate: the flipped edge falls outside the relation's cap.\n"
        "print('\\ntension_status:', thresholds.tension_status,\n"
        "      '-> contradiction caught by the plausibility gate, not u-space tension')"
    ),
    new_markdown_cell(
        "## Reload check\n"
        "\n"
        "The store is just files under `store_path`. A fresh `GMSExpertStore` loads it "
        "with no rebuild and answers the same queries."
    ),
    new_code_cell(
        "from knowlytix.knowledge.store import GMSExpertStore\n"
        "\n"
        "reloaded = GMSExpertStore(config, device)\n"
        "assert reloaded.load(), f'no store at {STORE}'\n"
        "assert sorted(reloaded.query_triples()) == sorted(store.query_triples())\n"
        "assert sorted(h for h, r, t in\n"
        "        reloaded.query_triples(relation='has_parent_of', tail='frank')) == ['carol', 'eve']\n"
        "print('reload OK:', len(reloaded.query_triples()), 'triples')"
    ),
    new_markdown_cell(
        "## Summary\n"
        "\n"
        "One markdown file became a trained, queryable, calibrated GMS store. The GEODE "
        "loop extracted the triples and trained the geometry; the store answers pattern "
        "queries (including many-to-many relations) directly and multi-hop questions by "
        "traversal. `GMSJudge.calibrate` fit an operating point for every decision "
        "channel from the graph itself, so plausibility is a calibrated verdict rather "
        "than a hardcoded cut: an unsupported fact (Ann as Frank's *parent*) is "
        "rejected, and flipping a true edge (`carol parent_of ann`) is caught as a "
        "contradiction. Calibration also reported the tension channel as `degenerate` "
        "on this corpus --- no declared opposition to fit --- so that gate abstains "
        "rather than guess. The same calls scale to real documents; swap the family "
        "tables for a financial report and the pipeline is unchanged."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
