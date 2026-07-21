#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""What the capstone verifier catches end-to-end vs at the checker level.

(1) numeric tamper through the full verifier: a fabricated fee figure is
    contradicted against the register (exact path).
(2) the fused value-polarity checker directly (as Beyond Chunk and Pray Ch11
    demonstrates it): a reversed stance is contradicted, a synonym supported.
"""
from __future__ import annotations
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)


def main() -> int:
    from agentlab.capstone.policy_rag import get_default_retriever
    from knowlytix.embedding import FineTunedEmbedding
    from knowlytix.knowledge.rag import PolarityCuts, ValuePolarityChecker

    rag = get_default_retriever()
    sp = str(rag.store.config.store_path) if hasattr(rag.store, "config") else \
        "data/gms_policy_store_cap"
    v = rag.pipe.verifier

    print("== (1) numeric tamper through the full verifier ==")
    for ans in ["The overdraft fee is 35.0 USD.",
                "The overdraft fee is 999.0 USD."]:
        rep = v.verify(ans)
        print(f"  {ans!r}  ok={rep.ok}")
        for cv in rep.verdicts:
            print(f"     {cv.triple.head}/{cv.triple.relation}={cv.triple.tail} "
                  f"-> {cv.status} {cv.detail or ''}")

    print("\n== (2) fused value-polarity checker, direct (Ch11 style) ==")
    vv = FineTunedEmbedding.load(f"{sp}/tuned_encoder")
    uu = FineTunedEmbedding.load(f"{sp}/value_polarity_encoder")
    cuts = PolarityCuts.load(f"{sp}/value_polarity_calibration.json")
    print(f"  cuts: tau_ent={cuts.tau_ent:.3f} tau_contra={cuts.tau_contra:.3f} "
          f"cv_acc={cuts.cv_accuracy:.3f}")
    checker = ValuePolarityChecker(rag.store, vv.encode, uu.encode, cuts)
    rule = ("pii_handling", "has_unencrypted_channel_pii", "forbidden")
    for val in ["forbidden", "prohibited", "banned", "permitted", "allowed"]:
        print(f"  SSN over unencrypted email is {val:11s} -> "
              f"{checker.check(rule[0], rule[1], val, rule[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
