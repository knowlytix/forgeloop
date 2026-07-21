#!/usr/bin/env python
"""Benchmark the entity-linking + GMS link-prediction extractor (89 held-out rows).

Calibrate the link-accept threshold on the CLEAN seed messages (abstain cohort =
cases whose expected_issue is general/None -> no policy entity should link), then
evaluate on the rephrased DoE rows. product = grounded link-predict (else Qwen
fallback); issue = grounded domain (else general). Reports product/issue/joint
(strict + accept-set), grounded fraction, and issue confusion -- vs hybrid_qwen
(0.46) and geo_task (0.62).
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
    return cases, {c["id"]: c for c in cases}


def main() -> int:
    os.environ.setdefault("AGENTLAB_POLICY_STORE", _V2)
    from agentlab.models.entity_link_extractor import EntityLinkExtractor

    diag = json.loads((_REPO / "data" / "diag_extract_facts.json").read_text())
    rows = diag["rows"]
    cases, by_id = _load_cases()

    ext = EntityLinkExtractor(store_path=_V2)
    abstain_msgs = [c["message"] for c in cases
                    if c.get("expected_issue") in (None, "general")]
    thr = ext.calibrate(abstain_msgs, ceiling=0.10)
    (_REPO / "data" / "entity_link_calibration.json").write_text(
        json.dumps({"threshold": thr, "store": _V2,
                    "n_abstain_cohort": len(abstain_msgs)}, indent=2) + "\n")
    print(f"calibrated link threshold = {thr:.4f} (abstain cohort {len(abstain_msgs)})")

    acc = {k: [0, 0] for k in ("p_strict", "i_strict", "j_strict", "i_acc", "j_acc")}
    conf = Counter()
    grounded = 0
    for r in rows:
        msg = r["message"]
        ep, ei = r["expected_product"], r["expected_issue"]
        case = by_id[r["seed_case"]]
        acc_p = set(case.get("acceptable_products") or ([ep] if ep else []))
        acc_i = set(case.get("acceptable_issues") or ([ei] if ei else []))
        routed = ext.extract(msg)
        if routed:
            grounded += 1
            issue = routed["issue"]
            product = routed["product"] or r["hybrid"].get("product")
        else:
            issue = "general"
            product = r["hybrid"].get("product")
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
    print(f"entity_link on {len(rows)} rows; grounded {grounded} ({grounded/len(rows):.0%})")
    for k in ("p_strict", "i_strict", "j_strict", "i_acc", "j_acc"):
        print(f"  {k:10} {rate[k][0]}  ({rate[k][1]})")
    print("\nissue confusion (expected -> predicted):")
    for (e, p), n in conf.most_common():
        print(f"   {e:18} -> {str(p):18} {n}{'' if e == p else '  <-- miss'}")
    (_REPO / "data" / "benchmark_entity_link.json").write_text(
        json.dumps({"threshold": thr, "n_rows": len(rows), "grounded": grounded,
                    "rates": rate,
                    "issue_confusion": {f"{e} -> {p}": n for (e, p), n in conf.most_common()}},
                   indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
