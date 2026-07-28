# SPDX-License-Identifier: Apache-2.0
"""Calibrate the accept/abstain gate for the capstone pipeline.

The accept gate needs a labeled cohort, so it cannot be fit at build time (unlike
the relevance gate, which is calibrated from the SFT supervision in
scripts/build_store.py). This runs data/eval_cohort.json through the SAME
pipeline the capstone uses (oracle == agent) with the accept gate disabled
(accept_threshold=0), collects (label, abstained, confidence) per case, and fits
the operating point with the library's calibrate_accept_threshold under a stated
false-accept ceiling. The result is persisted as rag_gate_calibration.json beside
the store, which capstone_pipeline.build_rag then loads. No hand-set threshold.

Run (GPU, loads Qwen):  python scripts/calibrate_accept_gate.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import REPO_ROOT, use_branch_library  # noqa: E402

use_branch_library()

from knowlytix.knowledge.rag import RagPipeline  # noqa: E402
from knowlytix.knowledge.rag.eval import calibrate_accept_threshold  # noqa: E402

import capstone_pipeline as cp  # noqa: E402

STORE = os.path.join(REPO_ROOT, "data", "gms_annual_report_store")
COHORT = os.path.join(REPO_ROOT, "data", "eval_cohort.json")
OUT = os.path.join(STORE, "rag_gate_calibration.json")
MAX_FALSE_ACCEPT = 0.05


def main() -> None:
    dev = cp.device()
    store = cp.load_store(STORE, dev)
    llm = cp.make_qwen(dev)
    # Accept gate OFF for the calibration pass so it does not pre-filter.
    rag = cp.build_rag(store, llm, accept_threshold=0.0)
    pipe = RagPipeline.from_store(store, rag)

    cohort = json.load(open(COHORT))
    records: list[tuple[int, bool, float]] = []
    print(f"{'id':16} {'label':5} {'decision':9} {'conf':6} reason/answer")
    for c in cohort:
        a = pipe.query(c["question"])
        label = 1 if c.get("expect_decision") == "accept" else 0
        abstained = a.decision != "accept"
        records.append((label, abstained, float(a.confidence)))
        detail = (a.notice or "") if abstained else a.answer
        print(f"{c['id']:16} {label:5} {a.decision:9} {a.confidence:6.3f} "
              f"{str(detail)[:44]}")

    payload = calibrate_accept_threshold(records, max_false_accept=MAX_FALSE_ACCEPT)
    payload["cohort"] = os.path.basename(COHORT)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"\naccept_threshold = {payload['accept_threshold']} "
          f"(balanced_acc={payload['balanced_accuracy']}, "
          f"recall={payload['recall']}, false_accept={payload['false_accept']}, "
          f"ceiling_met={payload['false_accept_ceiling_met']})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
