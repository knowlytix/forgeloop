# SPDX-License-Identifier: Apache-2.0
"""Calibrate the existing GEODE-RAG gates for the Annual Report store and verify.

This adds NO decision logic — every decision is already a calibrated gate in the
library (GEODE_RAG_DESIGN.md §14). It fits each gate's operating point from the
store, persists it as JSON beside the store (the form the runtime gates read),
and verifies the gate decides correctly.

Gates calibrated here (no LLM needed):
  * groundedness  — GMSJudge.calibrate() -> store/calibration.json (tau_geo),
                    read by GraphVerifier / HallucinationOracle.
  * relevance     — calibrate_relevance_thresholds() -> store/relevance_calibration.json
                    (tau_accept v-accept floor), read by GeometricRelevanceGate.

The accept-vs-abstain gate (calibrate_accept_threshold -> rag_gate_calibration.json)
needs a pipeline run over the cohort (Qwen) and lives in calibrate_accept_gate.py.

Run:  python scripts/calibrate_gates.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import REPO_ROOT, use_branch_library  # noqa: E402

use_branch_library()

import torch  # noqa: E402

from knowlytix.core.config import GeometryConfig  # noqa: E402
from knowlytix.core.graph.encoders import encode_texts  # noqa: E402
from knowlytix.knowledge.config import DocGMSConfig  # noqa: E402
from knowlytix.knowledge.store import GMSExpertStore  # noqa: E402
from knowlytix.knowledge.rag.query_triples import GeometricQueryParser  # noqa: E402
from knowlytix.knowledge.rag.relevance import (  # noqa: E402
    GeometricRelevanceGate, calibrate_relevance_thresholds,
)
from knowlytix.harness.testing.judge import GMSJudge  # noqa: E402
from knowlytix.harness.testing.hallucination import HallucinationOracle  # noqa: E402

STORE = os.path.join(REPO_ROOT, "data", "gms_annual_report_store")

# Build-time relation phrasings: a few natural ways each relation is asked. The
# v-accept floor is the lowest cosine a genuine phrasing has to its own relation.
RELATION_PHRASINGS = {
    "has_revenue":  ["revenue", "how much revenue", "topline", "sales"],
    "has_headcount": ["headcount", "how many staff", "number of employees", "staff count"],
    "has_division": ["which division", "what division it belongs to", "division"],
    "has_region":   ["which region", "region it operates in", "where it is located"],
    "has_head":     ["who heads it", "who leads it", "the head of", "its leader"],
    "has_amount":   ["the amount", "how much", "the dollar amount", "the figure in dollars"],
    "has_fy2024":   ["the fiscal 2024 figure", "value in 2024", "fy2024 amount"],
    "has_fy2025":   ["the fiscal 2025 figure", "value in 2025", "fy2025 amount"],
    "has_value":    ["the value", "the figure", "the number"],
}


def load_store():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GMSExpertStore(
        DocGMSConfig(store_path=STORE,
                     geometry=GeometryConfig(d_v=64, d_u=64, m=32, d=32)),
        device=dev)
    assert store.load(), "store failed to load"
    return store


def calibrate_groundedness(store) -> dict:
    print("\n== groundedness: GMSJudge.calibrate() -> calibration.json ==")
    judge = GMSJudge(store)
    judge.calibrate(seed=42)
    target = judge.save(STORE)
    payload = json.load(open(os.path.join(target, "calibration.json")))
    tau = payload.get("thresholds", {}).get("geodesic")
    print(f"  tau_geo (geodesic) = {tau}")
    return payload


def calibrate_relevance(store) -> dict:
    print("\n== relevance: calibrate_relevance_thresholds() -> relevance_calibration.json ==")
    rels = set(getattr(store.adapter, "relation_to_idx", {}))
    phrasings = {r: ws for r, ws in RELATION_PHRASINGS.items() if r in rels}
    # No contradiction-tuned u-encoder for this store (0 contradiction pairs),
    # so the u-veto is left uncalibrated and the gate runs accept-only (v floor).
    out = calibrate_relevance_thresholds(phrasings, encode_texts, encode_texts)
    out["u_veto"] = "disabled (no contradiction encoder; 0 contradiction pairs)"
    path = os.path.join(STORE, "relevance_calibration.json")
    json.dump(out, open(path, "w"), indent=2, sort_keys=True)
    print(f"  tau_accept = {out['tau_accept']}  (wrote {os.path.basename(path)})")
    return out


def verify_groundedness(store) -> None:
    print("\n== verify groundedness gate (HallucinationOracle reads calibration.json) ==")
    oracle = HallucinationOracle(store=store)  # loads tau_geo from calibration.json
    triples = list(store.doc_graph.triples)
    h, r, t = next((x for x in triples if x[1] != "in_section"), triples[0])
    import random
    rng = random.Random(0)
    ents = [e for e in store.adapter.entity_to_idx]
    fake = rng.choice([e for e in ents if e != t])
    good = oracle.assess_claim(h, r, t)
    bad = oracle.assess_claim(h, r, fake)
    print(f"  real  ({h},{r},{t}) -> passed={good.passed}")
    print(f"  fake  ({h},{r},{fake}) -> passed={bad.passed}")
    print(f"  VERDICT: {'PASS' if good.passed and not bad.passed else 'CHECK'}")


def verify_relevance(store, tau_accept: float) -> None:
    print(f"\n== verify relevance gate at calibrated tau_accept={tau_accept} ==")
    parser = GeometricQueryParser(store, encoder=encode_texts)
    gate = GeometricRelevanceGate(store, v_encoder=encode_texts,
                                  tau_accept=tau_accept)
    cases = [
        ("IN ", "accept", "What was Consumer division revenue?"),
        ("IN ", "accept", "What is the total revenue?"),
        ("IN ", "accept", "How many people work in Retail?"),
        ("OUT", "abstain", "What is the capital of France?"),
        ("OUT", "abstain", "How do I bake sourdough bread?"),
        ("OUT", "abstain", "Who won the 2014 World Cup?"),
    ]
    ok = 0
    for tag, want, q in cases:
        tris = parser.extract(q)
        head = next((t.head for t in tris if not t.head.startswith("?")), None)
        matched = gate.relevant_relation(q, head)[0] if head else None
        got = "accept" if matched else "abstain"
        good = got == want
        ok += good
        print(f"  [{tag}] want={want:7} got={got:7} {'ok' if good else 'MISS'}  {q}")
    print(f"  VERDICT: {ok}/{len(cases)} correct")


def main() -> None:
    store = load_store()
    calibrate_groundedness(store)
    rel = calibrate_relevance(store)
    verify_groundedness(store)
    verify_relevance(store, rel["tau_accept"])


if __name__ == "__main__":
    main()
