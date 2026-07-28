# SPDX-License-Identifier: Apache-2.0
"""DoE test of the GEODE-RAG with the GMS as ground truth (Beyond Ship and Pray).

The DoE-enriched cohort (scripts/enrich_data.py) is the test set: every question
carries a graph-derived answer, so the GMS itself is the oracle. We run each
question through the capstone RAG and measure, with knowlytix functionality:

  * precision@k / recall@k  -- did the retriever rank the grounding fact in top-k
                               (GMS value-equality, not string overlap)?
  * groundedness            -- geodesic confidence of the retrieved facts
                               (RetrievedFact.score -> exp(-d)).
  * completeness            -- TripleCompletenessScorer: did the generated answer
                               restate every fact the retriever grounded on?
  * weakness attribution    -- logistic regression (suite.evaluate.attribute ->
                               graphdoe DOEAnalyzer + Benjamini-Hochberg): which
                               presentation FACTORS (clarity, style, ...) drive the
                               failures, with corrected p-values and odds ratios.

Run (GPU):  python scripts/rag_doe_test.py --limit 150 --k 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import REPO_ROOT, use_branch_library  # noqa: E402

use_branch_library()

from knowlytix.harness.suite import evaluate as ev  # noqa: E402
from knowlytix.harness.testing.triple_completeness import (  # noqa: E402
    TripleCompletenessScorer, triples_from_rag_answer,
)

import capstone_pipeline as cp  # noqa: E402

STORE = os.path.join(REPO_ROOT, "data", "gms_annual_report_store")
COHORT = os.path.join(REPO_ROOT, "data", "enrichment", "rag_cohort.json")
FACTORS = ["clarity", "style", "length", "expertise", "paraphrase_depth"]


def _gt_value(expected) -> str:
    # multi_hop answers are [entry, value]; fact lookups are a scalar.
    if isinstance(expected, list) and len(expected) == 2:
        return str(expected[1])
    return str(expected)


def _num_eq(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return str(a).strip().lower() == str(b).strip().lower()


def _fact_matches(fact, gt: str) -> bool:
    return _num_eq(str(getattr(fact, "tail", "")), gt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    dev = cp.device()
    store = cp.load_store(STORE, dev)
    llm = cp.make_qwen(dev)
    # accept gate off: this test measures retrieval + answer quality, not the
    # accept/abstain cut (that gate is calibrated separately on this same cohort).
    pipe = cp.build_pipeline(store, llm, accept_threshold=0.0)
    scorer = TripleCompletenessScorer(store)

    cohort = json.load(open(COHORT))[: args.limit]
    K = args.k
    rows, prec, rec, ground, comp = [], [], [], [], []
    print(f"running {len(cohort)} DoE cohort questions through the RAG (k={K})...")
    for i, c in enumerate(cohort):
        ans = pipe.query(c["question"])
        gt = _gt_value(c["expected_answer"])
        topk = ans.sources[: K]
        hits = [f for f in topk if _fact_matches(f, gt)]
        p = len(hits) / max(1, len(topk))
        r = 1.0 if any(_fact_matches(f, gt) for f in ans.sources[: K]) else 0.0
        g = (sum(f.confidence for f in ans.sources) / len(ans.sources)
             if ans.sources else 0.0)
        rep = scorer.score(ans.answer or "", triples_from_rag_answer(ans))
        prec.append(p); rec.append(r); ground.append(g); comp.append(rep.score)
        rows.append({"sid": c["id"], "base": c["base"],
                     **{f"f_{k}": v for k, v in c["_factors"].items()},
                     "correct": int(r > 0)})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(cohort)}", flush=True)

    n = len(rows) or 1
    print("\n=== RAG DoE test (GMS as ground truth) ===")
    print(f"  n              : {len(rows)}")
    print(f"  precision@{K}    : {sum(prec)/n:.3f}")
    print(f"  recall@{K}       : {sum(rec)/n:.3f}")
    print(f"  groundedness   : {sum(ground)/n:.3f}  (mean geodesic confidence of sources)")
    print(f"  completeness   : {sum(comp)/n:.3f}  (answer restates retrieved facts)")

    print("\n=== weakness attribution (logistic on `correct`, BH-corrected) ===")
    table = ev.attribute(rows, FACTORS, metric="correct")
    for row in sorted(table, key=lambda r: r.get("p_value", 1.0)):
        sig = "  <-- significant" if row.get("significant") else ""
        print(f"  {str(row.get('factor','')):16} p={row.get('p_value', float('nan')):.3f} "
              f"pseudo_r2={row.get('pseudo_r2', float('nan')):.3f}{sig}")

    out = os.path.join(REPO_ROOT, "data", "enrichment", "rag_doe_results.json")
    json.dump({"n": len(rows), "k": K,
               "precision_at_k": sum(prec)/n, "recall_at_k": sum(rec)/n,
               "groundedness": sum(ground)/n, "completeness": sum(comp)/n,
               "attribution": table, "rows": rows}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
