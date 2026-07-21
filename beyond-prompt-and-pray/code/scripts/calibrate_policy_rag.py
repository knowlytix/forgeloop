#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Stage E (operator-native): calibrate the two decision gates of the banking
GEODE-RAG pipeline from a labeled cohort, under a false-accept ceiling, and
persist the operating points the retriever reads back. The embedding relevance
gate is RETIRED -- relations are operators, so there is no relation-phrase cut to
fit. Two gates remain:

  1. HEAD-BIND FLOOR -- the operator-native gate that REPLACES the relevance gate.
     A question whose best head-name cosine is below the floor names no policy
     entity, so the pipeline abstains. Calibrated from head-name scores:
     positives = answerable cohort questions (each names a real entity), negatives
     = out-of-vocabulary questions that name nothing. Recall-first: when the two
     clouds separate, the floor is the gap midpoint; else a balanced-accuracy
     sweep under the ceiling. Written to head_bind_calibration.json.
  2. ACCEPT THRESHOLD -- a confidence gate fit by ``calibrate_accept_threshold``
     over (label, abstained, confidence) collected by running the cohort through
     the production retriever with the head-bind floor set and the accept gate OFF
     (accept_threshold=0). Spurious binds that retrieve a head holding no
     answering fact abstain at select-and-answer synthesis, not here. Written to
     rag_gate_calibration.json.

Run on spark-ef84 (offline; model cached):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH=$HOME/jupyterlab/GMS-knowlytix \
      python scripts/calibrate_policy_rag.py --store-path data/gms_policy_store_cap
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)

_MAX_FALSE_ACCEPT = 0.05

# Answerable policy questions: each names a real policy entity AND a held attribute
# -> head binds, a retrieved fact answers, accept.
POSITIVES = [
    "What is the overdraft fee?",
    "How much is the NSF fee?",
    "What's the charge for a domestic wire transfer?",
    "What does an international wire cost?",
    "What is the stop payment fee?",
    "How much is the paper statement fee?",
    "What is the late payment fee on my credit card?",
    "How long do I have to dispute a charge?",
    "What's the deadline to file a transaction dispute?",
    "How long does a dispute investigation take?",
    "Is provisional credit issued during a dispute?",
    "How much can a representative reverse without approval?",
    "How much can a manager reverse?",
    "What is the supervisor reversal cap?",
    "How much notice before the bank closes my account?",
    "Does account closure require identity verification?",
    "Can my social security number be sent over email?",
    "How quickly must you notify me of a data breach?",
    "Is redaction required for personal data?",
    "What is the UDAAP harm threshold?",
    "How much is the late fee on loan servicing?",
    "Why was I charged a $35 overdraft fee?",
]
# Out-of-vocabulary: name NO policy entity -> the head-bind floor must reject them.
OOV_NEGATIVES = [
    "What is the capital of France?",
    "What is the weather forecast for tomorrow?",
    "How do I reset my online banking password?",
    "Who is the CEO of the bank?",
    "Tell me about cryptocurrency investing.",
    "What is management's outlook for next year?",
]
# No-such-attribute: name a real entity but ask for an attribute it does not hold
# -> the head binds, but select-and-answer synthesis must abstain (NOT the floor).
NOATTR_NEGATIVES = [
    "What is the overdraft interest rate?",
    "What is the annual fee for a checking account?",
    "What is the interest rate on a savings account?",
    "What is the policy?",
]


def _calibrate_floor(pos, neg, ceil=_MAX_FALSE_ACCEPT):
    """Recall-first head-bind floor: gap midpoint when the clouds separate, else a
    balanced-accuracy sweep under the false-accept ceiling."""
    pos, neg = sorted(pos), sorted(neg)
    if pos and neg and min(pos) > max(neg):
        return round(0.5 * (min(pos) + max(neg)), 4)
    grid = sorted({round(x, 3) for x in list(pos) + list(neg)} | {0.0})
    best, tau = None, 0.0
    for t in grid:
        recall = sum(s >= t for s in pos) / len(pos) if pos else 0.0
        far = sum(s >= t for s in neg) / len(neg) if neg else 0.0
        cand = (0.5 * (recall + (1 - far)), far <= ceil, -t)
        if best is None or cand > best:
            best, tau = cand, t
    return round(tau, 4)


def main() -> int:
    import torch
    from knowlytix.embedding import FineTunedEmbedding
    from knowlytix.knowledge.config import DocGMSConfig
    from knowlytix.knowledge.store import GMSExpertStore
    from knowlytix.knowledge.rag import calibrate_accept_threshold
    from knowlytix.knowledge.rag.query_triples import GeometricQueryParser

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", default="data/gms_policy_store_cap")
    args = ap.parse_args()
    sp = Path(args.store_path)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = DocGMSConfig(store_path=str(sp), ingest_mode="regex", loss_mode="cap")
    store = GMSExpertStore(cfg, dev)
    if not store.load():
        raise SystemExit(f"FAIL: no store at {sp}")
    # Inject the corpus alias table for operator-native head binding (same as the
    # production retriever) so the floor is calibrated on the WITH-alias scores.
    from agentlab.capstone.policy_rag import PolicyRagRetriever
    n_al = PolicyRagRetriever._inject_aliases(store, sp)
    print(f"injected {n_al} has_alias edges for head binding")
    v_enc = FineTunedEmbedding.load(str(sp / "tuned_encoder"))

    # --- 1) HEAD-BIND FLOOR: head-name scores, positives vs OOV ------------------
    parser = GeometricQueryParser(store, encoder=v_enc.encode)

    def top_score(q):
        c = parser.head_candidates(q, top_k=1)
        return c[0][1] if c else 0.0

    pos_s = [top_score(q) for q in POSITIVES]
    neg_s = [top_score(q) for q in OOV_NEGATIVES]
    floor = _calibrate_floor(pos_s, neg_s)
    recall = sum(s >= floor for s in pos_s) / len(pos_s)
    far = sum(s >= floor for s in neg_s) / len(neg_s)
    print(f"[head-bind] floor={floor:.4f}  recall={recall:.3f}  "
          f"false_accept={far:.3f}  ({len(pos_s)} pos / {len(neg_s)} OOV)")
    print(f"  pos head scores: {[round(x,3) for x in sorted(pos_s)]}")
    print(f"  OOV head scores: {[round(x,3) for x in sorted(neg_s)]}")
    floor_payload = {"method": "head_bind_floor_balanced_acc",
                     "head_bind_floor": floor, "max_false_accept": _MAX_FALSE_ACCEPT,
                     "recall": round(recall, 4), "false_accept": round(far, 4),
                     "n_pos": len(pos_s), "n_neg": len(neg_s)}
    (sp / "head_bind_calibration.json").write_text(
        json.dumps(floor_payload, indent=2) + "\n")

    # --- 2) ACCEPT THRESHOLD: run the cohort through the head_facts pipeline ------
    os.environ["AGENTLAB_RAG_HEAD_BIND_FLOOR"] = str(floor)
    os.environ["AGENTLAB_RAG_ACCEPT_THRESHOLD"] = "0.0"   # gate off for measurement
    from agentlab.capstone.policy_rag import PolicyRagRetriever
    rag = PolicyRagRetriever(store_path=str(sp))
    print(f"[accept] measuring on the cohort "
          f"(LLM={rag.llm.model_name}, head_facts route, accept gate OFF) ...")

    rows, leaks = [], 0
    for q in POSITIVES:
        ans = rag.pipe.query(q)
        rows.append((1, ans.decision == "abstain", float(ans.confidence)))
    for q in OOV_NEGATIVES + NOATTR_NEGATIVES:
        ans = rag.pipe.query(q)
        abst = ans.decision == "abstain"
        rows.append((0, abst, float(ans.confidence)))
        if not abst:
            leaks += 1
            print(f"   [pre-tau LEAK] neg answered conf={ans.confidence:.3f} "
                  f"verified={ans.verified}: {q!r}")
    cal = calibrate_accept_threshold(rows, max_false_accept=_MAX_FALSE_ACCEPT)
    print(f"[accept] tau={cal['accept_threshold']:.4f}  "
          f"balanced_acc={cal['balanced_accuracy']:.3f}  recall={cal['recall']:.3f}  "
          f"false_accept={cal['false_accept']:.3f} "
          f"(ceiling {_MAX_FALSE_ACCEPT} "
          f"{'met' if cal['false_accept_ceiling_met'] else 'UNMET'})  "
          f"n={cal['cohort_n']}  pre-tau leaks={leaks}")

    payload = {
        "store_path": str(sp),
        "calibrated_at": date.today().isoformat(),
        "method": "operator_native_headfloor+accept_v1",
        "head_bind_floor": floor,
        **cal,
    }
    (sp / "rag_gate_calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {sp/'head_bind_calibration.json'}")
    print(f"wrote {sp/'rag_gate_calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
