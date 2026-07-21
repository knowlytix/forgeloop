# SPDX-License-Identifier: Apache-2.0
"""Build notebooks/10_answering_through_the_gms_a_inline.ipynb (Ch8) with nbformat.

CPU-only: this only assembles the .ipynb JSON. It does NOT execute any cell —
no store load, no Qwen. The lead executes the notebook in CI.

Run:  python scripts/build_nb_08.py
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "notebooks", "10_answering_through_the_gms_a_inline.ipynb")

BOOTSTRAP = (
    "import os, sys\n"
    'KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", "/home/user/jupyterlab/GMS-knowlytix")\n'
    "sys.path.insert(0, KNOWLYTIX_SRC)"
)

LOAD_STORE = '''\
# Reload the trained store built by scripts/build_store.py (F2). This is the only
# GPU/Qwen-touching part of the chapter at build time -- the lead runs it in CI;
# retrieval itself is pure geometry and needs no LLM.
import os
import torch

from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.store import GMSExpertStore
from knowlytix.knowledge.geode.provenance import ProvenanceLedger

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), "..")))
STORE = os.environ.get("GMS_STORE",
                       os.path.join(os.getcwd(), "..", "data", "gms_annual_report_store"))

store = GMSExpertStore(DocGMSConfig(store_path=STORE),
                       device=torch.device("cpu"))
assert store.load(), f"no trained store at {STORE}; run scripts/build_store.py first"
ledger = ProvenanceLedger.from_text(store.markdown)
print("entities:", store.adapter.num_entities, " relations:", store.adapter.num_relations)'''

SINGLE_HOP = '''\
# Single-hop: bind a question's terms, then answer it *through the graph*.
# The asked value is the bare "?" slot; the retriever returns the asserted edge,
# never a parsed number, and attaches the source span.
from knowlytix.knowledge.rag import Retriever, TripleBinder
from knowlytix.knowledge.rag.query_triples import QueryTriple

binder = TripleBinder(store)
retriever = Retriever(store, ledger)

bt = binder.bind(QueryTriple("cloud platform", "has_revenue", "?"))
assert bt.bound  # every non-variable slot resolved to real graph vocabulary

result = retriever.retrieve([bt])
for f in result.facts:
    print(f"{f.head} {f.relation} {f.tail}  "
          f"[{f.source} conf={f.confidence:.2f}]  @ {f.location}")
    print("   raw:", f.raw)
print("answers:", result.answers)'''

SINGLE_HOP_OUT = '''\
# Expected (from data/corpus_facts.md; exact values from the trained store):
#   cloud platform has_revenue 120.0  [triple conf=1.00]  @ <report>:15:553-558
#      raw: <the table cell text the triple resolved to>
#   answers: [('120.0', 1.0)]
#
# The fact's source is "triple" (an *asserted* edge), score 0.0, confidence 1.0:
# the GMS preferred the in-graph fact over any link_predict guess. The 120.0 is
# the byte-exact figure carried by ENM, with a file:line:char span -- not a
# number scraped from prose.'''

MULTI_HOP = '''\
# Multi-hop: "which region hosts the highest-revenue segment?"
# Cloud Platform leads revenue (120.0). We resolve its division (?x), then the
# region of that division (?). The intermediate variable ?x links the two hops:
# the retriever resolves it from hop 1 and feeds it into hop 2 (variable env).
chain = [
    binder.bind(QueryTriple("cloud platform", "has_division", "?x")),
    binder.bind(QueryTriple("?x", "has_region", "?")),
]
assert all(b.bound for b in chain)

res = retriever.retrieve(chain)
print("hops:")
for f in res.facts:
    print(f"  {f.head} {f.relation} {f.tail}  [{f.source}]  @ {f.location}")
print("answer:", res.answers)'''

MULTI_HOP_OUT = '''\
# Expected (corpus_facts.md sample retrieval):
#   hops:
#     cloud platform has_division technology  [triple]  @ <report>:..
#     technology has_region north america     [triple]  @ <report>:..
#   answer: [('north america', 1.0)]
#
# Both hops are recorded as separate, provenance-bearing facts. The chain is the
# audit trail: a reader can see *why* the answer is "north america" -- segment ->
# division -> region -- and check each edge against its source span.'''

ASSERTED_VS_PREDICT = '''\
# Asserted edges are preferred over link_predict (Anti-pattern: guessing over a
# fact you hold). For an edge the graph asserts, source == "triple", score 0.0;
# link_predict only fills genuinely missing edges and is scored lower.
asserted = store.query_triples(head="cloud platform", relation="has_revenue")
print("asserted in graph:", asserted)

# A missing edge falls back to link_predict (a ranked guess, lower confidence):
guess = store.link_predict("cloud platform", "has_region", top_k=3)
print("link_predict (no asserted edge):", guess)'''

ASSERTED_VS_PREDICT_OUT = '''\
# Expected:
#   asserted in graph: [('cloud platform', 'has_revenue', '120.0')]
#   link_predict (no asserted edge): [(<entity>, <distance>), ...]
#
# The retriever's _tail_query checks query_triples first; only an empty result
# triggers link_predict. So an answer the graph *asserts* is always returned as
# a hard fact, never a prediction. The segment->region link is not asserted
# directly (region lives on the division), which is exactly why the region
# question must be answered multi-hop, not by a single-edge guess.'''

SELF_CHECK = '''\
# Self-check: the multi-hop answer matches the cohort label, and every supporting
# fact carries provenance (the chapter's claim: bound triples -> facts, with
# multi-hop resolved through a variable env and provenance attached per fact).
res = retriever.retrieve([
    binder.bind(QueryTriple("cloud platform", "has_division", "?x")),
    binder.bind(QueryTriple("?x", "has_region", "?")),
])
assert res.answers, "multi-hop produced no answer"
assert res.answers[0][0] == "north america", res.answers
assert {(f.head, f.relation, f.tail) for f in res.facts} == {
    ("cloud platform", "has_division", "technology"),
    ("technology", "has_region", "north america"),
}
assert all(f.source == "triple" for f in res.facts), \\
    "expected asserted edges, not predictions"
assert all(f.location for f in res.facts), "a fact lost its provenance span"
print("OK: multi-hop answer = north america, both hops asserted with provenance")'''

EXERCISE = '''\
# Exercise (solution): a 3-hop chain over segment -> division -> region, then ask
# for the head of that region's division. We reuse ?x (division) for the head hop.
ex = [
    binder.bind(QueryTriple("logistics", "has_division", "?x")),  # -> operations
    binder.bind(QueryTriple("?x", "has_region", "?y")),           # -> europe
    binder.bind(QueryTriple("?x", "has_head", "?")),              # -> sam reyes
]
assert all(b.bound for b in ex)
ex_res = retriever.retrieve(ex)
print("3-hop facts:")
for f in ex_res.facts:
    print(f"  {f.head} {f.relation} {f.tail}")
print("answer (division head):", ex_res.answers)
# Expected facts: logistics has_division operations; operations has_region europe;
# operations has_head sam reyes.  answer: [('sam reyes', 1.0)]
assert ex_res.answers[0][0] == "sam reyes", ex_res.answers'''


def main() -> None:
    nb = new_notebook()
    cells = [
        new_markdown_cell(
            "# Ch8 — Answering through the GMS\n\n"
            "Binding turned the question's words into graph vocabulary (Ch7). This "
            "chapter turns **bound query triples into facts**. The retriever "
            "resolves each triple *through the GMS*: it prefers an asserted edge "
            "over a `link_predict` guess, chains multi-hop questions through a "
            "variable environment (`?x` feeds the next hop, the bare `?` is the "
            "asked value), attaches a `file:line:char` provenance span to every "
            "fact, and returns ENM-exact numerics.\n\n"
            "No LLM is involved in *retrieval* — only the geometry. Synthesis "
            "(Ch9) and verification (Ch10) come after."),
        new_code_cell(BOOTSTRAP),
        new_code_cell(LOAD_STORE),
        new_markdown_cell("## Listing 8.1 — Single-hop retrieval with provenance"),
        new_code_cell(SINGLE_HOP),
        new_code_cell(SINGLE_HOP_OUT),
        new_markdown_cell("## Listing 8.2 — Multi-hop via a variable environment"),
        new_code_cell(MULTI_HOP),
        new_code_cell(MULTI_HOP_OUT),
        new_markdown_cell("## Listing 8.3 — Asserted edges preferred over link_predict"),
        new_code_cell(ASSERTED_VS_PREDICT),
        new_code_cell(ASSERTED_VS_PREDICT_OUT),
        new_markdown_cell("## Exercise solution — a 3-hop chain"),
        new_code_cell(EXERCISE),
        new_markdown_cell("## Self-check"),
        new_code_cell(SELF_CHECK),
    ]
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
