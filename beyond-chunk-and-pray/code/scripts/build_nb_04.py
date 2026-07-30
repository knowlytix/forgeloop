# SPDX-License-Identifier: Apache-2.0
"""Builder for notebooks/04_document_to_graph_a_inline.ipynb (Ch4 — From document to graph).

CPU-only: emits a valid nbformat-4 notebook with nbformat. Does NOT execute the
notebook (no store build, no Qwen). Run:  python scripts/build_nb_04.py
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "notebooks", "04_document_to_graph_a_inline.ipynb")


def build() -> nbf.NotebookNode:
    cells: list = []

    # --- Cell 0: the canonical bootstrap (global brief A.5, verbatim). ---
    cells.append(new_code_cell(
        "import os, sys\n"
        'KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", "")\n'
        "sys.path.insert(0, KNOWLYTIX_SRC)"
    ))

    cells.append(new_markdown_cell(
        "# Ch4 — From document to graph\n"
        "\n"
        "We turn `data/annual_report.md` into a trained, queryable "
        "`GMSExpertStore`. Ingestion is **deterministic** (regex mode, no LLM "
        "augmentation) so the graph is byte-stable, and every authoritative "
        "number is parsed once into Exact Numerical Memory (ENM) rather than "
        "left as a string a model might re-read. `build_rag_store` runs GEODE "
        "self-correction (Ch5) before training; here we read its diagnostics "
        "and confirm the store round-trips."
    ))

    # --- Cell: paths + config. ---
    cells.append(new_code_cell(
        "import torch\n"
        "\n"
        "from knowlytix.core.config import GeometryConfig, TrainConfig\n"
        "from knowlytix.knowledge.config import DocGMSConfig\n"
        "from knowlytix.knowledge.geode.rag import build_rag_store\n"
        "from knowlytix.knowledge.geode.loop import make_default_trainer\n"
        "\n"
        'REPO_ROOT = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()\n'
        'MD = os.path.join(REPO_ROOT, "data", "annual_report.md")\n'
        'STORE = os.path.join(REPO_ROOT, "data", "gms_annual_report_store")\n'
        "\n"
        'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
        "config = DocGMSConfig(\n"
        "    store_path=STORE,\n"
        '    ingest_mode="regex",   # deterministic: no model-extracted numbers\n'
        '    loss_mode="cap",\n'
        "    geometry=GeometryConfig(d_v=64, d_u=64, m=32, d=32),\n"
        "    train=TrainConfig(epochs=150, batch_size=64, neg_samples=16,\n"
        "                      lr=5e-3, lr_riemannian=2e-3),\n"
        ")"
    ))

    cells.append(new_markdown_cell(
        "## Listing 1 — `build_rag_store` over the annual report\n"
        "\n"
        "One call ingests the document, runs the GEODE correction loop, trains "
        "the production GMS, populates ENM, and saves the store to disk. We "
        "print the entity / relation / triple / ENM counts and the GEODE audit "
        "(corrections, anchor violations). **GPU/Qwen cell — the lead executes "
        "this in CI; do not run it during authoring.**"
    ))

    cells.append(new_code_cell(
        "res = build_rag_store(\n"
        "    MD, config, device=device,\n"
        "    geode_trainer=make_default_trainer(device, epochs=80),\n"
        ")\n"
        "store = res.store\n"
        "\n"
        'print(f"converged={res.converged}  iterations={res.iterations}")\n'
        'print(f"entities={res.n_entities}  triples={res.n_triples}  enm={res.n_enm}")\n'
        'print(f"relations={len(store.adapter.relation_to_idx)}")\n'
        'print(f"GEODE corrections:      {len(res.corrections)}")\n'
        'print(f"GEODE anchor violations:{len(res.anchor_violations)}")'
    ))

    cells.append(new_markdown_cell(
        "Expected output on the shipped corpus:\n"
        "\n"
        "```\n"
        "converged=True  iterations=1\n"
        "entities=... triples=... enm=21\n"
        "relations=10\n"
        "GEODE corrections:      0\n"
        "GEODE anchor violations:0\n"
        "```\n"
        "\n"
        "Ten relations — `has_amount`, `has_division`, `has_fy2024`, "
        "`has_fy2025`, `has_head`, `has_headcount`, `has_region`, "
        "`has_revenue`, `has_value`, `in_section` — and **21** ENM entries "
        "(the segment, income-statement, and balance-sheet figures). Because "
        "the clean corpus has no contradictions or broken sum anchors, GEODE "
        "reports zero corrections and zero anchor violations; Ch5 corrupts a "
        "figure to make the loop work."
    ))

    cells.append(new_markdown_cell(
        "## Listing 2 — reload from disk and query\n"
        "\n"
        "The store is just files under `store_path`. We construct a fresh "
        "`GMSExpertStore` from the same config and `load()` it — no rebuild, "
        "no GEODE, no training — then read a figure from ENM and pattern-match "
        "a triple. ENM is the authoritative numeric channel: `lookup_enm` "
        "returns the value **byte-exact**, never a string parsed at query time."
    ))

    cells.append(new_code_cell(
        "from knowlytix.knowledge.store import GMSExpertStore\n"
        "\n"
        "reloaded = GMSExpertStore(config, device)\n"
        "assert reloaded.load(), f'no store at {STORE}'\n"
        "\n"
        "# Exact numeric recall (segment_performance category, byte-exact).\n"
        'cloud_rev = reloaded.lookup_enm("segment_performance",\n'
        '                                "Cloud Platform/Technology/Revenue")\n'
        'total_rev = reloaded.lookup_enm("income_statement", "Revenue/FY2025")\n'
        'print(f"Cloud Platform revenue = {cloud_rev}")\n'
        'print(f"Total FY2025 revenue   = {total_rev}")\n'
        "\n"
        "# Pattern-match the relational triple that links the segment to its division.\n"
        'div = reloaded.query_triples(head="cloud platform", relation="has_division")\n'
        'print("cloud platform division:", div)'
    ))

    cells.append(new_markdown_cell(
        "Expected output:\n"
        "\n"
        "```\n"
        "Cloud Platform revenue = 120.0\n"
        "Total FY2025 revenue   = 355.0\n"
        "cloud platform division: [('cloud platform', 'has_division', 'technology')]\n"
        "```\n"
        "\n"
        "The reloaded store answers the same as the freshly built one: the "
        "figures match the report's Segment Performance and Income Statement "
        "tables, and the `cloud platform -> technology` edge is present for "
        "the multi-hop chains in Ch8."
    ))

    cells.append(new_markdown_cell(
        "## Exercise — add a row, rebuild, retrieve\n"
        "\n"
        "Add a fifth segment to the report, rebuild, and confirm the new fact "
        "is retrievable. We append `Media | Technology | 40.0 | 90` to the "
        "Segment Performance table (writing to a copy so the shipped corpus "
        "stays untouched), rebuild into a scratch store, and query the new "
        "segment's revenue. **GPU/Qwen cell — lead executes in CI.**"
    ))

    cells.append(new_code_cell(
        "import shutil, tempfile\n"
        "\n"
        "with open(MD, encoding='utf-8') as f:\n"
        "    text = f.read()\n"
        "# Insert the new row directly above the Total row of the segment table.\n"
        'new_row = "| Media | Technology | 40.0 | 90 |\\n"\n'
        'total_line = "| Total | All | 355.0 | 1500 |"\n'
        "augmented = text.replace(total_line, new_row + total_line)\n"
        "assert new_row in augmented\n"
        "\n"
        "scratch_dir = tempfile.mkdtemp(prefix='ch4_aug_')\n"
        "aug_md = os.path.join(scratch_dir, 'annual_report_aug.md')\n"
        "with open(aug_md, 'w', encoding='utf-8') as f:\n"
        "    f.write(augmented)\n"
        "\n"
        "import dataclasses\n"
        "aug_config = dataclasses.replace(\n"
        "    config, store_path=os.path.join(scratch_dir, 'store'))\n"
        "aug = build_rag_store(aug_md, aug_config, device=device,\n"
        "                      geode_trainer=make_default_trainer(device, epochs=80))\n"
        "media_rev = aug.store.lookup_enm('segment_performance',\n"
        "                                 'Media/Technology/Revenue')\n"
        "print('new segment Media revenue =', media_rev)\n"
        "assert media_rev == 40.0\n"
        "shutil.rmtree(scratch_dir)"
    ))

    cells.append(new_markdown_cell(
        "## Self-check — the built store round-trips and has the segments\n"
        "\n"
        "Proves the chapter's claim: a store built from the document can be "
        "saved and reloaded byte-for-byte (same ENM value, same segment "
        "triples). The asserted figures come straight from `corpus_facts.md`. "
        "**GPU cell — lead executes in CI.**"
    ))

    cells.append(new_code_cell(
        "# Round-trip: the reloaded store agrees with the freshly built one\n"
        "# on the authoritative figures, and carries the four segments.\n"
        "for cat, key, expected in [\n"
        "    ('segment_performance', 'Cloud Platform/Technology/Revenue', 120.0),\n"
        "    ('segment_performance', 'Devices/Technology/Revenue', 80.0),\n"
        "    ('segment_performance', 'Logistics/Operations/Revenue', 95.0),\n"
        "    ('segment_performance', 'Retail/Operations/Revenue', 60.0),\n"
        "    ('segment_performance', 'Total/All/Revenue', 355.0),\n"
        "]:\n"
        "    assert store.lookup_enm(cat, key) == expected\n"
        "    assert reloaded.lookup_enm(cat, key) == expected\n"
        "\n"
        "segments = {h for (h, r, t) in reloaded.triples if r == 'has_revenue'}\n"
        "assert {'cloud platform', 'devices', 'logistics', 'retail'} <= segments\n"
        "assert reloaded.stats()['enm_entries'] == store.stats()['enm_entries'] == 21\n"
        "print('OK: store round-trips; 4 segments present; 21 ENM entries')"
    ))

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def main() -> None:
    nb = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"wrote {OUT} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
