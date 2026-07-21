#!/usr/bin/env python
"""Diagnostic: decompose WHY extract_facts is the capstone's weakest tool.

The harness scores extract as a strict two-field exact-match conjunction
(product AND issue) but persists only the pass/fail flag -- not the predicted
fields. This reruns the EXACT materialized DoE messages through the extractor
and records, per scored scenario:

  - rule-only     : _rule_extract (deterministic keyword floor)
  - raw Qwen      : the LLM proposal BEFORE normalization/guards
  - hybrid        : _extract_impl == what the tool actually returns (Qwen + guards)

so we can tell whether the loss is Qwen itself, the taxonomy normalization,
the regulatory guards, or the labels/scorer. Also tabulates the two suspected
structural traps: (1) `general` is near-unreachable (the canonical map only
emits it for product=unknown), (2) the UDAAP fee-guard downgrades overdraft_fee
when a rephrasing strips the fee token -- broken out by clarity.

Writes data/diag_extract_facts.json. Qwen-heavy (~10-15 min).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from agentlab.testing import CapstoneTestHarness
from agentlab.capstone.banking_tools import (
    _rule_extract, _normalize_llm, _canonical_issue, _FEE_SIGNAL, _PRODUCTS,
)

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "data" / "diag_extract_facts.json"


def _field_ok(pred: dict, case: dict) -> tuple[bool | None, bool | None]:
    """(product_ok, issue_ok) vs the case ground truth, None when not defined."""
    p_ok = (pred.get("product") == case["expected_product"]
            if case.get("expected_product") is not None else None)
    i_ok = (pred.get("issue") == case["expected_issue"]
            if case.get("expected_issue") is not None else None)
    return p_ok, i_ok


def _joint(p_ok, i_ok) -> bool | None:
    checks = [c for c in (p_ok, i_ok) if c is not None]
    return all(checks) if checks else None


def main() -> int:
    h = CapstoneTestHarness(n_runs=120, seed=42, rephrase_method="qwen")
    from agentlab.models.qwen_extractor import get_default_extractor
    extractor = get_default_extractor()

    design = h.design()
    records = design.to_dict("records")

    rows = []
    for drow in records:
        case = h._cases_by_id[drow["seed_case"]]
        if case.get("expected_product") is None and case.get("expected_issue") is None:
            continue  # never scored on extract
        msg = h.materialize(drow)
        rule = _rule_extract(msg)
        raw = extractor.extract(msg)            # before normalize/guards
        hybrid = _normalize_llm(raw, msg, rule) if raw else dict(rule)

        # would the raw-LLM canonical issue be overdraft_fee, and did the guard
        # then downgrade it because the (rephrased) message lacks a fee token?
        raw_canon_issue = (_canonical_issue(
            raw.get("product") if (raw and raw.get("product") in _PRODUCTS) else rule["product"],
            raw.get("issue", "")) if raw else None)
        guard_flipped = bool(
            raw_canon_issue == "overdraft_fee"
            and not _FEE_SIGNAL.search(msg)
            and hybrid.get("issue") != "overdraft_fee"
        )

        rp, ri = _field_ok(rule, case)
        wp, wi = _field_ok({"product": (raw or {}).get("product"),
                            "issue": raw_canon_issue}, case)  # raw mapped to canon
        hp, hi = _field_ok(hybrid, case)

        rows.append({
            "seed_case": case["id"],
            "clarity": drow["clarity"],
            "expected_product": case.get("expected_product"),
            "expected_issue": case.get("expected_issue"),
            "message": msg,
            "rule": rule, "raw_llm": raw, "hybrid": hybrid,
            "rule_p": rp, "rule_i": ri, "rule_joint": _joint(rp, ri),
            "raw_p": wp, "raw_i": wi, "raw_joint": _joint(wp, wi),
            "hyb_p": hp, "hyb_i": hi, "hyb_joint": _joint(hp, hi),
            "guard_flipped": guard_flipped,
            "general_unreachable_miss": bool(
                case.get("expected_issue") == "general" and hi is False),
        })

    def rate(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    summary = {
        "n_scored_rows": len(rows),
        "accuracy": {
            variant: {
                "product": rate(f"{p}_p"), "issue": rate(f"{p}_i"),
                "joint": rate(f"{p}_joint"),
            } for variant, p in (("rule_only", "rule"), ("raw_qwen", "raw"), ("hybrid", "hyb"))
        },
        "guard_flips": sum(r["guard_flipped"] for r in rows),
        "general_unreachable_misses": sum(r["general_unreachable_miss"] for r in rows),
        "general_rows": sum(1 for r in rows if r["expected_issue"] == "general"),
    }

    # hybrid joint accuracy by clarity (rephrase-washout test)
    by_clar = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["hyb_joint"] is not None:
            by_clar[r["clarity"]][0] += int(r["hyb_joint"])
            by_clar[r["clarity"]][1] += 1
    summary["hybrid_joint_by_clarity"] = {
        c: {"acc": round(ok / n, 3), "n": n} for c, (ok, n) in sorted(by_clar.items())
    }

    # issue confusion (hybrid) where issue is scored
    conf = Counter()
    for r in rows:
        if r["expected_issue"] is not None:
            conf[(r["expected_issue"], r["hybrid"].get("issue"))] += 1
    summary["issue_confusion_hybrid"] = {f"{e} -> {p}": n for (e, p), n in conf.most_common()}

    _OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
