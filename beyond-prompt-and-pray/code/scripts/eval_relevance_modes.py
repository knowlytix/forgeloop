#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Stage E helper: compare the relevance-gate modes on the labeled query cohort.

Reports, per mode, recall (grounded queries accepted) and false-accept
(out-of-corpus queries accepted) so the gate is chosen on data, not by default.
Modes: geometric u-veto (query-calibrated) vs the Qwen3-4B LLM judge.

Run on spark-ef84:
    python scripts/eval_relevance_modes.py --store-path data/gms_policy_store_cap
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)

from scripts.calibrate_policy_rag import QUERY_PHRASINGS, NEGATIVES

POSITIVES = [q for qs in QUERY_PHRASINGS.values() for q in qs]


def _measure(store_path):
    from agentlab.capstone.policy_rag import PolicyRagRetriever
    rag = PolicyRagRetriever(store_path=store_path)

    def accepted(q):
        ans = rag.pipe.query(q, generate=False)
        return ans.decision != "abstain" and bool(ans.sources)

    pos_acc = [q for q in POSITIVES if accepted(q)]
    neg_acc = [q for q in NEGATIVES if accepted(q)]
    recall = len(pos_acc) / len(POSITIVES)
    fa = len(neg_acc) / len(NEGATIVES)
    return recall, fa, neg_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", default="data/gms_policy_store_cap")
    args = ap.parse_args()

    for mode, uveto in [("geometric", "1"), ("llm", "0")]:
        os.environ["AGENTLAB_RAG_RELEVANCE_MODE"] = mode
        os.environ["AGENTLAB_RAG_UVETO"] = uveto
        # fresh process-state: PolicyRagRetriever reads env at construction
        for m in list(sys.modules):
            if m.startswith("agentlab.capstone.policy_rag"):
                del sys.modules[m]
        recall, fa, neg_acc = _measure(args.store_path)
        label = f"{mode}{' +u-veto' if uveto == '1' else ''}"
        print(f"\n### relevance={label}: recall={recall:.2f} "
              f"false_accept={fa:.2f} (n_pos={len(POSITIVES)} n_neg={len(NEGATIVES)})")
        if neg_acc:
            print("   leaked negatives:")
            for q in neg_acc:
                print(f"     - {q!r}")


if __name__ == "__main__":
    main()
