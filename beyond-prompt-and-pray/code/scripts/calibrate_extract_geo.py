#!/usr/bin/env python
"""Calibrate the geometric extractor's abstain thresholds and persist them.

Cohort = the CLEAN seed-case messages (data/eval_cases/cases.json). For each
classifier the abstain class is the "no concrete label" truth (product: unknown
/ unspecified; issue: general / unspecified). :meth:`GeometricLabelClassifier.
calibrate` picks the smallest threshold whose false-accept rate over that abstain
class is <= the ceiling, so concrete-label recall is preserved as far as the
ceiling allows. Calibrated on CLEAN messages, the extractor is then evaluated on
the REPHRASED DoE rows (scripts/benchmark_extractors.py) -- a clean train/test
split that measures robustness to the phrasing variation that washed out Qwen.

Writes data/extract_geo_calibration.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentlab.models.geometric_extractor import (
    _ISSUE_EXEMPLARS, _PRODUCT_EXEMPLARS, _CALIB_PATH, _load_encoder,
)
from knowlytix.knowledge.rag import GeometricLabelClassifier

_REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ceiling", type=float, default=0.10,
                    help="false-accept ceiling for the abstain gate")
    ap.add_argument("--encoder", default="minilm",
                    help="'minilm' or 'tuned:<store_path>'")
    args = ap.parse_args()

    cases = json.loads((_REPO / "data" / "eval_cases" / "cases.json").read_text())
    cases = cases if isinstance(cases, list) else cases.get("cases", cases)

    # Clean-message cohort with the abstain class folded in (None -> abstain).
    prod_cohort = [(c["message"], c.get("expected_product") or "unknown") for c in cases]
    issue_cohort = [(c["message"], c.get("expected_issue") or "general") for c in cases]

    enc = _load_encoder(args.encoder)
    prod = GeometricLabelClassifier(_PRODUCT_EXEMPLARS, enc)
    issue = GeometricLabelClassifier(_ISSUE_EXEMPLARS, enc)

    pt = prod.calibrate(prod_cohort, abstain_label="unknown",
                        false_accept_ceiling=args.ceiling)
    it = issue.calibrate(issue_cohort, abstain_label="general",
                         false_accept_ceiling=args.ceiling)

    n_prod_abstain = sum(1 for _, l in prod_cohort if l == "unknown")
    n_issue_abstain = sum(1 for _, l in issue_cohort if l == "general")

    out = {
        "encoder": args.encoder,
        "false_accept_ceiling": args.ceiling,
        "product_threshold": pt,
        "issue_threshold": it,
        "cohort": {
            "n_cases": len(cases),
            "n_product_abstain_class": n_prod_abstain,
            "n_issue_abstain_class": n_issue_abstain,
        },
    }
    _CALIB_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {_CALIB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
