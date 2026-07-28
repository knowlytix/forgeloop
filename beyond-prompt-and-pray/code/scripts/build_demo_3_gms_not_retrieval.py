#!/usr/bin/env python
"""Builder for notebooks/demos/demo_3_gms_not_retrieval.ipynb.

Teaching demo 3: why GMS is not just retrieval. The multi-hop gap is real on Beyond
Chunk and Pray's annual-report corpus (large enough that the bridge fact is not near
the query), not on BPP's six-document banking policy. This reuses BCaP's tested wiring:
- GMS/GEODE side  : scripts/capstone_pipeline.py (load_store, make_qwen, build_pipeline)
- dense baseline  : scripts/baseline_rag.py (BaselineRAG -- chunk-and-pray, always answers)
- questions       : data/eval_cohort.json (two typed multi_hop questions + a single-hop control)
Executed so the deck shows the real answers each system returns.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "demos" / "demo_3_gms_not_retrieval.ipynb"

cells = [
    new_markdown_cell(
        "# Demo 3 --- Why GMS is not just retrieval\n"
        "\n"
        "Dense retrieval finds the passage nearest a query. That suffices for a single-hop "
        "fact, where the answer sits in one chunk. It fails on a multi-hop question --- one "
        "whose answer requires following a fact to the entity it names and then to a "
        "second fact about that entity --- because no single chunk holds the chain. The "
        "GMS/GEODE retriever traverses those hops through the graph.\n"
        "\n"
        "This runs on Beyond Chunk and Pray's tested pipeline over its annual-report "
        "corpus, which is large enough that the multi-hop gap appears (BPP's six-document "
        "banking policy is too small --- every bridge fact is already near the query). The "
        "GMS side is `capstone_pipeline.py`; the dense side is the `BaselineRAG` "
        "chunk-and-pray baseline; the questions come from the book's `eval_cohort.json`."
    ),
    new_code_cell(
        "import os, sys, json, warnings\n"
        "from pathlib import Path\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# Reuse Beyond Chunk and Pray's tested code + store; run from its code dir so the\n"
        "# relative data paths resolve exactly as the book uses them.\n"
        "bcap = next((c for c in (Path('../../../beyond-chunk-and-pray/code'),\n"
        "                         Path('../../beyond-chunk-and-pray/code'),\n"
        "                         Path('beyond-chunk-and-pray/code'))\n"
        "             if (c / 'data' / 'gms_annual_report_store').exists()), None)\n"
        "assert bcap is not None, 'beyond-chunk-and-pray/code not found'\n"
        "bcap = bcap.resolve()\n"
        "sys.path.insert(0, str(bcap / 'scripts'))\n"
        "os.chdir(bcap)\n"
        "print('cwd:', Path.cwd())\n"
        "\n"
        "from capstone_pipeline import device, load_store, make_qwen, build_pipeline\n"
        "from baseline_rag import BaselineRAG\n"
        "\n"
        "cohort = {q['id']: q for q in json.loads(Path('data/eval_cohort.json').read_text())}"
    ),
    new_markdown_cell(
        "## Build both retrievers over the same corpus\n"
        "\n"
        "Both answer from `data/annual_report.md` with the same Qwen model. The only "
        "difference is the retrieval substrate: the GMS pipeline parses the query to graph "
        "entities and traverses triples; the baseline embeds fixed-size chunks and returns "
        "the nearest by cosine similarity, then stuffs them into the prompt."
    ),
    new_code_cell(
        "dev = device()\n"
        "store = load_store('data/gms_annual_report_store', dev)\n"
        "qwen = make_qwen(dev)\n"
        "\n"
        "# GMS/GEODE retriever (triple-mediated, verified, can abstain)\n"
        "gms = build_pipeline(store, qwen)\n"
        "\n"
        "# Dense 'chunk and pray' baseline over the same document, same LLM\n"
        "def qwen_generate(prompt: str) -> str:\n"
        "    return qwen.call(system='You are a financial-report assistant. Answer concisely '\n"
        "                            'from the passages.', user=prompt, max_tokens=64)\n"
        "corpus = Path('data/annual_report.md').read_text()\n"
        "dense = BaselineRAG(corpus, qwen_generate, device=str(dev))\n"
        "print('both retrievers built over data/annual_report.md')"
    ),
    new_markdown_cell(
        "## Single-hop control: both answer\n"
        "\n"
        "\"What is Cloud Platform revenue?\" is answered by one passage. Nearest-chunk "
        "retrieval finds it, so dense and GMS both get it right. Retrieval alone is not "
        "the differentiator here."
    ),
    new_code_cell(
        "q = cohort['q-seg-rev']\n"
        "print('Q:', q['question'], '| expected:', q['expected_answer'])\n"
        "d = dense.query(q['question']).answer\n"
        "g = gms.query(q['question'])\n"
        "print('dense :', d)\n"
        "print('GMS   :', g.answer, f'(decision={g.decision})')"
    ),
    new_markdown_cell(
        "## Multi-hop: dense misses the chain, GMS traverses it\n"
        "\n"
        "\"Which region runs the division that contains Cloud Platform?\" needs two hops: "
        "Cloud Platform to its division, then the division to its region. No single chunk "
        "contains both, so the dense baseline retrieves passages about Cloud Platform or "
        "about regions and cannot connect them. The GMS retriever walks the graph and "
        "returns the grounded answer."
    ),
    new_code_cell(
        "for qid in ['q-multihop-reg', 'q-multihop-head']:\n"
        "    q = cohort[qid]\n"
        "    exp = str(q['expected_answer']).lower()\n"
        "    d = dense.query(q['question']).answer\n"
        "    g = gms.query(q['question'])\n"
        "    print('Q:', q['question'])\n"
        "    print(f\"  expected : {q['expected_answer']}\")\n"
        "    print(f\"  dense    : {d}   -> {'correct' if exp in d.lower() else 'WRONG'}\")\n"
        "    print(f\"  GMS      : {g.answer} (decision={g.decision})   -> {'correct' if exp in g.answer.lower() else 'WRONG'}\")\n"
        "    print()"
    ),
    new_markdown_cell(
        "The single-hop question is a tie; the multi-hop questions are where they part. "
        "The corpus, the LLM and the questions are held fixed --- the only variable is the "
        "retrieval substrate. Dense retrieval has no graph to walk, so it answers from "
        "whatever chunks are nearest and cannot chain the hops. GMS assembles the answer "
        "by traversing the graph and verifies each claim against it. That is the property "
        "a nearest-neighbor lookup cannot provide, and it is why GMS is a retrieval "
        "substrate, not a better vector store."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
