#!/usr/bin/env python
"""Benchmark the claim-route extractor on the held-out rephrased DoE rows.

Same 89 rows / scorers as scripts/benchmark_extractors.py. For each message:
route claims through the v2 product-augmented GEODE store; product = grounded
has_product (else fall back to the Qwen product from the diagnostic); issue =
grounded coarse issue (else `general`). Reports product/issue/joint under strict
and accept-set scorers, the grounded fraction, and the issue confusion -- to
compare against hybrid_qwen (0.46 issue) and geo_task (0.62 issue, +cc confusion).
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_V2 = str(_REPO / "data" / "gms_policy_store_geode_v2")


def _load_cases():
    cases = json.loads((_REPO / "data" / "eval_cases" / "cases.json").read_text())
    cases = cases if isinstance(cases, list) else cases.get("cases", cases)
    return {c["id"]: c for c in cases}


def main() -> int:
    os.environ.setdefault("AGENTLAB_POLICY_STORE", _V2)
    from agentlab.models.claim_route_extractor import ClaimRouteExtractor

    diag = json.loads((_REPO / "data" / "diag_extract_facts.json").read_text())
    rows = diag["rows"]
    by_id = _load_cases()
    ext = ClaimRouteExtractor(store_path=_V2)

    acc = {k: [0, 0] for k in ("p_strict", "i_strict", "j_strict", "i_acc", "j_acc")}
    conf = Counter()
    grounded = 0
    for r in rows:
        msg = r["message"]
        ep, ei = r["expected_product"], r["expected_issue"]
        case = by_id[r["seed_case"]]
        acc_p = set(case.get("acceptable_products") or ([ep] if ep else []))
        acc_i = set(case.get("acceptable_issues") or ([ei] if ei else []))
        routed = ext.route(msg)
        if routed:
            grounded += 1
            issue = routed["issue"]
            product = routed["product"] or r["hybrid"].get("product")
        else:
            issue = "general"
            product = r["hybrid"].get("product")   # keep strong Qwen product
        sp = (product == ep) if ep else None
        si = (issue == ei) if ei else None
        ap = (product in acc_p) if ep else None
        ai = (issue in acc_i) if ei else None
        for key, val in (("p_strict", sp), ("i_strict", si), ("i_acc", ai)):
            if val is not None:
                acc[key][0] += int(val); acc[key][1] += 1
        sj = [v for v in (sp, si) if v is not None]
        aj = [v for v in (ap, ai) if v is not None]
        if sj:
            acc["j_strict"][0] += int(all(sj)); acc["j_strict"][1] += 1
        if aj:
            acc["j_acc"][0] += int(all(aj)); acc["j_acc"][1] += 1
        if ei is not None:
            conf[(ei, issue)] += 1

    rate = {k: (round(c / n, 3) if n else None, f"{c}/{n}") for k, (c, n) in acc.items()}
    print("claim_route on", len(rows), "rows; grounded", grounded,
          f"({grounded/len(rows):.0%})")
    for k in ("p_strict", "i_strict", "j_strict", "i_acc", "j_acc"):
        print(f"  {k:10} {rate[k][0]}  ({rate[k][1]})")
    print("\nissue confusion (expected -> predicted):")
    for (e, p), n in conf.most_common():
        print(f"   {e:18} -> {str(p):18} {n}{'' if e == p else '  <-- miss'}")
    (_REPO / "data" / "benchmark_claim_route.json").write_text(
        json.dumps({"n_rows": len(rows), "grounded": grounded, "rates": rate,
                    "issue_confusion": {f"{e} -> {p}": n for (e, p), n in conf.most_common()}},
                   indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
