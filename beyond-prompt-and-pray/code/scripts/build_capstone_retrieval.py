#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Build data/capstone_retrieval.json — graph-truth retrieval benchmark.

Grades ``search_policy`` retrieval against the policy graph ground truth.
Each case in ``eval_cases/policy_retrieval_cohort.json`` is a customer
question whose expected answer lives in the graph. ``benchmark_retrieval``
parses the question, binds it to a (head, relation, tail) triple, retrieves,
and counts a hit only when the retrieved value matches the ground-truth value.
Misses are localized to parse / bind / retrieve. A dense top-k baseline on
the same encoder runs in parallel so the gap is the retrieval mechanism, not
the embedding.

Requires ``gms_policy_store_cap`` (Tier-2, GPU + Qwen).

Run::

    python scripts/build_capstone_retrieval.py
    python scripts/build_capstone_retrieval.py --store data/gms_policy_store_cap
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="data/gms_policy_store_cap",
                    help="path to the GEODE policy store (default: data/gms_policy_store_cap)")
    ap.add_argument("--cohort", default="data/eval_cases/policy_retrieval_cohort.json",
                    help="retrieval cohort JSON")
    ap.add_argument("--out", default="data/capstone_retrieval.json",
                    help="output path")
    args = ap.parse_args()

    store_path = _REPO / args.store
    cohort_path = _REPO / args.cohort
    out_path = _REPO / args.out

    if not (store_path / "model.pt").exists():
        print(f"FAIL: store not found at {store_path}; run build_geode_rag_store.py first",
              file=sys.stderr)
        return 1
    if not cohort_path.exists():
        print(f"FAIL: cohort not found at {cohort_path}", file=sys.stderr)
        return 1

    import torch
    from knowlytix.embedding import FineTunedEmbedding
    from knowlytix.knowledge.geode.provenance import ProvenanceLedger
    from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
    from knowlytix.knowledge.rag import EvalCase, benchmark_retrieval

    from agentlab.capstone.policy_rag import PolicyRagRetriever

    spec = json.loads(cohort_path.read_text())
    cases = [EvalCase(question=c["question"], expected_answer=c["expected_answer"])
             for c in spec["cases"]]
    print(f"cohort: {len(cases)} cases from {cohort_path.name}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GMSExpertStore(DocGMSConfig(store_path=str(store_path), ingest_mode="regex"), device=dev)
    store.load()
    PolicyRagRetriever._inject_aliases(store, store_path)
    v = FineTunedEmbedding.load(os.path.join(store_path, "tuned_encoder"))
    ledger = ProvenanceLedger.from_text(store.markdown, "<report>", prefer_prose=True)

    print("running benchmark_retrieval (GMS + dense baseline) ...", flush=True)
    rb = benchmark_retrieval(store, cases, encoder=v.encode, ledger=ledger,
                             top_k=5, include_dense=True)

    misses = [r for r in rb["results"] if r.gms_hit is False]
    result = {
        "n": len(cases),
        "gms": {
            "recall": rb["gms_recall"],
            "precision": rb["gms_precision"],
            "parse_rate": rb.get("parse_rate"),
            "bind_rate": rb.get("bind_rate"),
        },
        "dense": {
            "recall": rb["dense_recall"],
            "precision_at_k": rb["dense_precision_at_k"],
            "precision_at_1": rb["dense_precision_at_1"],
        },
        "miss_stage": dict(Counter(r.miss_stage for r in misses)),
        "misses": [
            {"question": r.question, "expected": r.expected,
             "stage": r.miss_stage, "triples": r.gms_triples,
             "answers": r.gms_answers}
            for r in misses
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"gms recall={result['gms']['recall']}  precision={result['gms']['precision']}")
    print(f"dense recall={result['dense']['recall']}  "
          f"prec@1={result['dense']['precision_at_1']}  "
          f"prec@k={result['dense']['precision_at_k']}")
    print(f"misses: {len(misses)} ({result['miss_stage']})")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
