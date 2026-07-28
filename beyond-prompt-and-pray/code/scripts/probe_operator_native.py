#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Operator-native validation for the banking GEODE-RAG pipeline. Runs grounded,
out-of-vocabulary and no-such-attribute questions through the migrated head_facts
pipeline and reports, per case: the head-bind score, the decision/route, the
answer, and whether it meets expectation. Summarizes recall (grounded -> accept),
negative abstain rate, and confident_wrong (any negative ACCEPTED, or a grounded
answer that accepts the wrong number) -- the guarantee the gates must deliver.

Run on spark-ef84 (offline):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH=$HOME/GMS-knowlytix \
      python scripts/probe_operator_native.py --store-path data/gms_policy_store_cap
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)

# (question, kind, expect_substr). kind: "grounded" -> accept (answer should carry
# expect_substr if given); "oov" -> abstain (no entity); "noattr" -> abstain
# (entity holds no such attribute).
CASES = [
    ("What is the overdraft fee?",                          "grounded", "35"),
    ("How long do I have to dispute a charge?",             "grounded", "60"),
    ("How long does a dispute investigation take?",         "grounded", "10"),
    ("How much can a representative reverse without approval?", "grounded", "35"),
    ("How much notice before the bank closes my account?",  "grounded", "30"),
    ("Can my social security number be sent over email?",    "grounded", None),
    ("What is the UDAAP harm threshold?",                   "grounded", "500"),
    ("Why was I charged a $35 overdraft fee?",              "grounded", "35"),
    ("I want to dispute a charge I never authorized.",      "grounded", None),
    ("My social security number was emailed unencrypted.",  "grounded", None),
    ("Escalate this unfair fee to compliance.",            "grounded", None),
    # out-of-vocabulary -> must abstain (head-bind floor)
    ("What is the capital of France?",                     "oov", None),
    ("What is the weather forecast for tomorrow?",          "oov", None),
    ("Who is the CEO of the bank?",                        "oov", None),
    # no-such-attribute -> must abstain (select synthesis)
    ("What is the overdraft interest rate?",               "noattr", None),
    ("What is the interest rate on a savings account?",     "noattr", None),
]


def main() -> int:
    import torch
    from knowlytix.embedding import FineTunedEmbedding
    from knowlytix.knowledge.config import DocGMSConfig
    from knowlytix.knowledge.store import GMSExpertStore
    from knowlytix.knowledge.rag.query_triples import GeometricQueryParser

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", default="data/gms_policy_store_cap")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # head-name scorer (for the reported bind score)
    cfg = DocGMSConfig(store_path=str(args.store_path), ingest_mode="regex", loss_mode="cap")
    s2 = GMSExpertStore(cfg, dev)
    s2.load()
    from agentlab.capstone.policy_rag import PolicyRagRetriever
    PolicyRagRetriever._inject_aliases(s2, Path(args.store_path))  # match production head binding
    v_enc = FineTunedEmbedding.load(str(Path(args.store_path) / "tuned_encoder"))
    parser = GeometricQueryParser(s2, encoder=v_enc.encode)

    rag = PolicyRagRetriever(store_path=str(args.store_path))
    floor = rag.pipe.rag.head_bind_floor
    print(f"store={args.store_path}  LLM={rag.llm.model_name}  "
          f"retrieve_mode={rag.pipe.rag.retrieve_mode}  head_bind_floor={floor:.4f}  "
          f"accept_threshold={rag.accept_threshold:.4f}")

    recall_hit = recall_tot = 0
    neg_abstain = neg_tot = 0
    confident_wrong = 0
    for q, kind, exp in CASES:
        cand = parser.head_candidates(q, top_k=1)
        hb = cand[0] if cand else ("-", 0.0)
        ans = rag.pipe.query(q)
        acc = ans.decision == "accept"
        atxt = (ans.answer or "").replace("\n", " ")[:90]
        ok = True
        if kind == "grounded":
            recall_tot += 1
            ok = acc and (exp is None or exp in (ans.answer or ""))
            recall_hit += ok
            if acc and exp is not None and exp not in (ans.answer or ""):
                confident_wrong += 1            # accepted a wrong number
        else:
            neg_tot += 1
            ok = not acc
            neg_abstain += ok
            if acc:
                confident_wrong += 1            # accepted an out-of-scope question
        mark = "OK " if ok else "XX "
        print(f"  [{mark}] {kind:8s} head={hb[0]!r}({hb[1]:.2f}) "
              f"-> {ans.decision}/{ans.route} verified={ans.verified}")
        print(f"        Q: {q!r}")
        print(f"        A: {atxt!r}")
    print(f"\nrecall (grounded->accept): {recall_hit}/{recall_tot}   "
          f"negatives abstained: {neg_abstain}/{neg_tot}   "
          f"CONFIDENT_WRONG: {confident_wrong}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
