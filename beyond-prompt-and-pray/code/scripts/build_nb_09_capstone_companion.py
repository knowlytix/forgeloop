#!/usr/bin/env python
"""Builder for notebooks/09_capstone_companion.ipynb.

Capstone companion to Chapter 9 (Memory: Types, Retrieval and Hybrid Stores):
the concept read on the running banking complaint agent. Teaching notebook in the
style of the main chapter notebooks -- real capstone imports, no pre-embedded
outputs. The store/encoder loads are GPU-heavy, so the live retrieval cell is
reader-runnable rather than executed here; a pinned graph-truth artifact
(data/capstone_retrieval.json) illustrates the numbers cheaply.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "09_capstone_companion.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone companion --- Chapter 9: Memory as a Trained Triple Store\n"
        "\n"
        "Chapter~9 distinguishes three access patterns an agent's memory must serve --- "
        "recency, similarity and relationships --- and adds a fourth tier for correctness: "
        "a trained geometric triple store that recalls exact values and rejects "
        "contradictions on write. This companion reads that fourth tier on the capstone "
        "banking complaint agent, whose policy knowledge is precisely such a store. A "
        "customer question is not matched to a passage by vector similarity; it is parsed "
        "into a (head, relation, tail) query triple, bound to the graph's canonical "
        "vocabulary and answered by the governing fact the graph actually holds."
    ),
    new_markdown_cell(
        "## The problem with similarity retrieval for a governing fact\n"
        "\n"
        "Consider the question *\"How much is the overdraft fee?\"*. A vector store returns "
        "the passage whose embedding lies nearest the question, and synthesis reads a "
        "number out of that prose. Two failure modes follow. First, an adjacent-but-wrong "
        "attribute --- the overdraft *interest rate* rather than the *fee* --- embeds close "
        "enough to be retrieved and paraphrased as the answer. Second, the number itself is "
        "decoded by a language model from prose, so it can drift. The capstone's retriever "
        "instead treats the relation as a geometric operator: the question binds to the "
        "policy *entity* (`overdraft`), and the fee is read as the exact tail of the "
        "`overdraft --has_fee_amount--> ?` edge the graph asserts. Memory here is a "
        "structured store queried by structure, not a passage index queried by proximity."
    ),
    new_markdown_cell(
        "## The retriever and its store\n"
        "\n"
        "`PolicyRagRetriever` in `agentlab.capstone.policy_rag` drives knowlytix's GEODE "
        "graph-RAG pipeline over a self-corrected geometric store built from the banking "
        "policy corpus. Construction loads the trained store, the document-tuned encoder "
        "and the local query-time model, so it is deferred to the reader. The cell below "
        "confirms only that the symbols import."
    ),
    new_code_cell(
        "from agentlab.capstone import policy_rag\n"
        "from agentlab.capstone.policy_rag import (\n"
        "    PolicyRagRetriever,\n"
        "    get_default_retriever,\n"
        ")\n"
        "\n"
        "# The default store the deployed retriever loads (a trained triple store,\n"
        "# not a vector index). Construction is GPU/heavy; we only report the path.\n"
        "print('default store :', policy_rag._DEFAULT_STORE)\n"
        "print('query-time LLM :', policy_rag._RAG_LLM_MODEL)"
    ),
    new_markdown_cell(
        "## Parse, then bind: from a customer message to a query triple\n"
        "\n"
        "The parse step is isolated on the retriever as `extract`. It runs the GEODE "
        "parse-and-bind loop on a raw message and returns the grounded extraction: the "
        "query triples the question was parsed into, each bound to the graph's real "
        "vocabulary so colloquial wording (*\"I overdrew my account\"*) maps onto the "
        "canonical policy entity (`overdraft`). The return carries `query_facts` --- the "
        "bound `(head, relation, tail)` triples --- and `is_bound`, which is `False` when "
        "nothing grounds, the abstain that keeps a vague message from acquiring a "
        "fabricated label.\n"
        "\n"
        "This cell issues a live parse and requires the loaded store, so it is written for "
        "the reader to run; it is not executed in this notebook."
    ),
    new_code_cell(
        "# READER-RUNNABLE (loads the GPU store + encoder). Not executed here.\n"
        "#\n"
        "# retriever = get_default_retriever()\n"
        "# parsed = retriever.extract('I was charged a $35 overdraft fee I did not authorize.')\n"
        "# print('is_bound    :', parsed['is_bound'])\n"
        "# for h, r, t in parsed['query_facts']:\n"
        "#     print(f'  {h} --{r}--> {t}')\n"
        "#\n"
        "# Expected shape: is_bound=True and a bound triple headed by the policy\n"
        "# entity, e.g.  overdraft --has_fee_amount--> ?  (the tail is the slot the\n"
        "# retrieval fills from the graph)."
    ),
    new_markdown_cell(
        "## Retrieve the governing fact\n"
        "\n"
        "`search` routes the query through the pipeline and returns the grounded answer in "
        "the `search_policy` tool's shape. The bound head's admissible facts are retrieved "
        "through the relation operators, an answering fact is selected (or the pipeline "
        "abstains), and the result carries the resolved policy `id`, the retrieved "
        "`policies` ranked by plausibility, the provenance `text`, the synthesized "
        "`answer`, a confidence `score` and the `query_facts` that bound. An `extraction` "
        "from a prior `extract` call may be passed back so the message is parsed once. This "
        "cell also requires the loaded store and is left for the reader to run."
    ),
    new_code_cell(
        "# READER-RUNNABLE (loads the GPU store + query-time LLM). Not executed here.\n"
        "#\n"
        "# retriever = get_default_retriever()\n"
        "# hits = retriever.search('How much is the overdraft fee?', k=3)\n"
        "# top = hits[0]\n"
        "# print('policy id   :', top['id'])\n"
        "# print('policies    :', top['policies'])       # plausibility-ranked recall\n"
        "# print('answer      :', top['answer'])\n"
        "# print('score       :', top['score'], 'decision:', top['decision'])\n"
        "# print('query_facts :', top['query_facts'])    # the bound (h, r, t) triples\n"
        "# print('provenance  :')\n"
        "# print(top['text'])\n"
        "#\n"
        "# An empty list is the honest abstain: nothing bound, or the answer rested\n"
        "# on inadmissible evidence (the cap/tension gate dropped every retrieved fact)."
    ),
    new_markdown_cell(
        "## What the structured store recovers, at graph truth\n"
        "\n"
        "To read the effect of structured retrieval without loading the store, we inspect a "
        "pinned artifact of graph-truth numbers over a fixed question set. It reports the "
        "triple store (`gms`) against a similarity baseline (`dense`) on the same "
        "questions. The store binds every question (`parse_rate` and `bind_rate` at 1.0) "
        "and recovers the governing fact at markedly higher precision than top-k similarity "
        "--- the direct measurement of the failure mode the opening section described."
    ),
    new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "root = Path('.') if Path('data').exists() else Path('..')\n"
        "art = json.loads((root / 'data' / 'capstone_retrieval.json').read_text())\n"
        "gms, dense = art['gms'], art['dense']\n"
        "print(f\"questions        : {art['n']}\")\n"
        "print(f\"gms   recall     : {gms['recall']:.2f}   precision  : {gms['precision']:.2f}\")\n"
        "print(f\"gms   parse_rate : {gms['parse_rate']:.2f}   bind_rate  : {gms['bind_rate']:.2f}\")\n"
        "print(f\"dense recall     : {dense['recall']:.2f}   prec@1     : {dense['precision_at_1']:.2f}   prec@k : {dense['precision_at_k']:.2f}\")"
    ),
    new_markdown_cell(
        "## A miss is a bound triple, not a lost passage\n"
        "\n"
        "Where the store misses, the artifact records *which* query triple was constructed "
        "and what the graph returned for it. A miss is legible: the question parsed and "
        "bound to a `(head, relation, tail)`, and the retrieved tail was the wrong "
        "register (a per-occurrence question answered by a fee amount). This is the "
        "diagnostic value of memory-as-structure --- a failure names the triple it "
        "resolved, rather than a passage that happened to rank."
    ),
    new_code_cell(
        "for m in art.get('misses', []):\n"
        "    (h, r, t), = m['triples']\n"
        "    got = m['answers'][0][0] if m.get('answers') else '(none)'\n"
        "    print(f\"Q: {m['question']}\")\n"
        "    print(f\"   triple   : {h} --{r}--> {t}\")\n"
        "    print(f\"   expected : {m['expected']:<16} got: {got}   (missed at: {m['stage']})\")\n"
        "    print()"
    ),
    new_markdown_cell(
        "This is the capstone's realization of Chapter~9. The policy knowledge that "
        "`search_policy` draws on is the chapter's fourth memory tier: a trained geometric "
        "triple store, queried by parse-and-bind rather than by vector similarity, so a "
        "governing fact is recovered as the exact tail of the relation the question names "
        "and a miss is a legible bound triple. The three general access patterns of "
        "Chapter~9 --- recency, similarity, relationships --- remain the agent's working "
        "memory; correctness-critical policy lookup rests on the structured tier. "
        "Chapter~15 assembles this retriever into the governed complaint workflow, where "
        "the retrieved fact and its provenance span constrain the drafted response."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
