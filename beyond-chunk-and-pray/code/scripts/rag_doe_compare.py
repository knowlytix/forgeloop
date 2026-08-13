# SPDX-License-Identifier: Apache-2.0
"""Compare the GEODE-RAG against the traditional chunk-and-pray baseline (App E)
on the SAME DoE cohort, with the SAME GMS-as-oracle metrics.

Both systems answer the DoE-enriched questions (scripts/enrich_data.py); both are
scored identically:
  * recall@k / precision@k -- is the ground-truth value among the top-k retrieved
    units (GEODE triples / baseline chunks), by GMS value-equality?
  * groundedness  -- mean GMS plausibility (exp(-geodesic) via score_triple) of the
    claims the ANSWER asserts (parsed with the same GeometricClaimExtractor).
  * completeness  -- TripleCompletenessScorer: does the answer restate the golden
    triple recovered from the store?
  * abstention    -- GEODE may abstain; the baseline never can (architectural).
  * weakness      -- logistic attribution (suite.evaluate.attribute / DOEAnalyzer)
    of failures to the presentation factors, per system.

Run (GPU):  python scripts/rag_doe_compare.py --limit 150 --k 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import REPO_ROOT, use_branch_library  # noqa: E402

use_branch_library()

import re  # noqa: E402

from knowlytix.harness.suite import evaluate as ev  # noqa: E402
from knowlytix.harness.testing.completeness import CompletenessEvaluator  # noqa: E402
from knowlytix.harness.testing.hallucination import HallucinationOracle  # noqa: E402

import torch  # noqa: E402

import baseline_rag  # noqa: E402
import capstone_pipeline as cp  # noqa: E402

STORE = os.path.join(REPO_ROOT, "data", "gms_annual_report_store")
COHORT = os.path.join(REPO_ROOT, "data", "enrichment", "rag_cohort.json")
REPORT = os.path.join(REPO_ROOT, "data", "annual_report.md")
FACTORS = ["clarity", "style", "length", "expertise", "paraphrase_depth"]


def _gt_value(expected) -> str:
    if isinstance(expected, list) and len(expected) == 2:
        return str(expected[1])
    return str(expected)


def _num_eq(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return str(a).strip().lower() == str(b).strip().lower()


def _value_in_text(gt: str, text: str) -> bool:
    if gt in text:
        return True
    if gt.endswith(".0") and gt[:-2] in text:   # 120.0 -> "120"
        return True
    return False


def _golden(store, case):
    gt = _gt_value(case["expected_answer"])
    attr = case["attribute"]
    out = []
    for h, r, t in store.triples:
        if not _num_eq(str(t), gt):
            continue
        if attr.startswith("has_") and r != attr:
            continue
        out.append((h, r, str(t)))
    return out


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _asserted_value(answer: str, gt: str):
    """The value the answer commits to for the asked attribute. Numeric target ->
    the answer's first number; entity target -> the gt string if the answer states
    it. This is the answer-driven claim the calibrated oracle then judges."""
    try:
        float(gt)
        m = _NUM.findall((answer or "").replace(",", ""))
        return m[0] if m else None
    except ValueError:
        return gt if gt and gt.lower() in (answer or "").lower() else None


def _eval_system(name, query_fn, source_hit, oracle, comp_eval, cohort, store, K):
    prec, rec, abst, rows = [], [], [], []
    correct_ans, comp_ans = [], []     # answered cases only
    print(f"\n[{name}] running {len(cohort)} questions (k={K})...")
    for i, c in enumerate(cohort):
        ans = query_fn(c["question"])
        gt = _gt_value(c["expected_answer"])
        srcs = list(getattr(ans, "sources", []))[:K]
        hits = [s for s in srcs if source_hit(s, gt)]
        prec.append(len(hits) / max(1, len(srcs)))
        rec.append(1.0 if hits else 0.0)
        answered = getattr(ans, "decision", "accept") == "accept"
        abst.append(0.0 if answered else 1.0)
        # Answer-driven, calibrated correctness + completeness (the Ch16 method),
        # answered cases only so coverage (abstention) does not conflate with it.
        if answered:
            gold = _golden(store, c)               # store triples (head, rel, value)
            head, rel, _ = gold[0] if gold else (None, None, None)
            # correctness: does the answer assert a value that GROUNDS under the
            # store's calibrated per-relation cut for the asked (head, rel)? For a
            # numeric target, scan EVERY number in the prose (a leading fiscal year
            # otherwise gets mistaken for the value); a hallucinated wrong value
            # grounds for none. For an entity target, check the gt string.
            ok = 0.0
            if head:
                try:
                    float(gt)
                    cands = [m.replace(",", "") for m in _NUM.findall(ans.answer or "")]
                except ValueError:
                    cands = [gt] if _asserted_value(ans.answer, gt) is not None else []
                for cand in cands:
                    try:
                        if oracle.assess_claim(head, rel, cand).passed:
                            ok = 1.0
                            break
                    except Exception:  # noqa: BLE001
                        pass
            correct_ans.append(ok)
            # completeness: recall of the expected store-grounded atoms.
            rep = comp_eval.evaluate(ans.answer or "", [gt], {"head": head})
            comp_ans.append(rep.score)
        rows.append({"sid": c["id"], "base": c["base"],
                     **{f"f_{k}": v for k, v in c["_factors"].items()},
                     "correct": int(bool(hits))})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(cohort)}", flush=True)
    n = len(cohort) or 1
    na = len(correct_ans) or 1
    return {"precision_at_k": sum(prec) / n, "recall_at_k": sum(rec) / n,
            "correctness": sum(correct_ans) / na, "completeness": sum(comp_ans) / na,
            "answered": len(correct_ans), "abstention_rate": sum(abst) / n,
            "attribution": ev.attribute(rows, FACTORS, metric="correct")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    dev = cp.device()
    store = cp.load_store(STORE, torch.device("cpu"))  # tiny model; CPU saves VRAM for Qwen
    llm = cp.make_qwen(dev)
    oracle = HallucinationOracle(store=store)     # calibrated per-relation grounded cut
    comp_eval = CompletenessEvaluator(store)
    cohort = json.load(open(COHORT))[: args.limit]

    # GEODE-RAG (triple-mediated, GMS-tuned encoders, verify+abstain).
    geode = cp.build_pipeline(store, llm, accept_threshold=0.0)
    geode_res = _eval_system(
        "GEODE-RAG", lambda q: geode.query(q),
        lambda s, gt: _num_eq(str(getattr(s, "tail", "")), gt),
        oracle, comp_eval, cohort, store, args.k)

    # Chunk-and-pray baseline (cosine top-k chunks + Qwen, never abstains).
    def qwen_gen(prompt: str) -> str:
        return llm.call(system="", user=prompt, max_tokens=128)
    baseline = baseline_rag.BaselineRAG(open(REPORT).read(), qwen_gen, device=str(dev))
    base_res = _eval_system(
        "baseline", lambda q: baseline.query(q),
        lambda s, gt: _value_in_text(gt, getattr(s, "text", "")),
        oracle, comp_eval, cohort, store, args.k)

    # Comparison table.
    print("\n" + "=" * 64)
    print(f"{'metric':18} {'GEODE-RAG':>12} {'chunk-and-pray':>16}")
    print("-" * 64)
    for key, lab in [("precision_at_k", f"precision@{args.k}"),
                     ("recall_at_k", f"recall@{args.k}"),
                     ("correctness", "correctness"),
                     ("completeness", "completeness"),
                     ("abstention_rate", "abstention")]:
        print(f"{lab:18} {geode_res[key]:>12.3f} {base_res[key]:>16.3f}")

    for name, res in [("GEODE-RAG", geode_res), ("baseline", base_res)]:
        print(f"\n[{name}] weakness attribution (logistic, BH-corrected):")
        for r in sorted(res["attribution"], key=lambda r: r.get("p_value", 1.0)):
            sig = "  <-- significant" if r.get("significant") else ""
            print(f"  {str(r.get('factor','')):16} p={r.get('p_value', float('nan')):.3f}{sig}")

    out = os.path.join(REPO_ROOT, "data", "enrichment", "rag_doe_compare.json")
    json.dump({"k": args.k, "n": len(cohort),
               "geode": geode_res, "baseline": base_res}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
