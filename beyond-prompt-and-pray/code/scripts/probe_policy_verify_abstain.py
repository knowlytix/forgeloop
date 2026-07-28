#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Stage D (Chunk-and-Pray Ch10-11): self-verification, abstention and coverage
for the banking RAG.

Three checks on the new store:
  1. coverage_report -- which policy regions are triple-mediated and which are
     blind spots (prose with no triples), so a question about a blind spot can
     abstain rather than guess.
  2. self-verification (geometric, LLM-free) -- a faithful answer verifies; a
     tampered figure is caught as a contradiction against the graph.
  3. abstention -- out-of-corpus / ungrounded questions return no grounded
     evidence (search -> []), with the pipeline's decision/notice surfaced.

Run on spark-ef84:
    python scripts/probe_policy_verify_abstain.py --store-path data/gms_policy_store_cap
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)

from agentlab.capstone.policy_rag import PolicyRagRetriever
from knowlytix.knowledge.rag import coverage_report

# (answer text, should_verify). The tampered figure must be caught.
VERIFY_CASES = [
    ("The overdraft fee is $35.", True),
    ("The overdraft fee is $999.", False),     # tampered -> contradicted
]

# Questions with no grounded answer in the corpus -> must abstain.
ABSTAIN_QS = [
    "What is the weather forecast for tomorrow?",
    "Who is the CEO of the bank?",
    "What is management's outlook for next year?",
    "What is the interest rate on a savings account?",   # not in corpus
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", default="data/gms_policy_store_cap")
    args = ap.parse_args()

    rag = PolicyRagRetriever(store_path=args.store_path)
    print(f"RAG LLM: {rag.llm.model_name}  tau={rag.accept_threshold}  "
          f"verifier={getattr(rag.pipe.verifier, 'mode', None)}\n")

    # 1) coverage
    cov = coverage_report(rag.store)
    print("=== coverage ===")
    print(f"  coverage_ratio = {cov.coverage_ratio:.2f}")
    for rg in cov.regions:
        flag = "  <-- BLIND SPOT" if rg.blind_spot else ""
        print(f"   [{'x' if rg.covered else ' '}] {rg.title} "
              f"({rg.triple_count} triples, {rg.body_lines} body lines){flag}")

    # 2) self-verification (geometric)
    print("\n=== self-verification (geometric, claims vs graph) ===")
    v_ok = 0
    for text, expect_ok in VERIFY_CASES:
        rep = rag.pipe.verifier.verify(text)
        ok = bool(getattr(rep, "ok", False))
        passed = (ok == expect_ok)
        v_ok += passed
        print(f"   [{'OK ' if passed else 'XX '}] verify({text!r}) -> ok={ok} "
              f"(expected {expect_ok})")
        for vd in getattr(rep, "verdicts", []):
            print(f"        {vd.triple} -> {vd.status} | {vd.detail}")
    print(f"   verification correct: {v_ok}/{len(VERIFY_CASES)}")

    # 3) abstention on ungrounded questions
    print("\n=== abstention (out-of-corpus / ungrounded -> no answer) ===")
    abst = 0
    for q in ABSTAIN_QS:
        res = rag.search(q)
        ans = rag.pipe.query(q)            # decision/notice for the audit trail
        abstained = (not res)
        abst += abstained
        print(f"   [{'ABSTAIN' if abstained else 'ANSWER '}] {q!r}")
        print(f"        decision={ans.decision} notice={ans.notice!r}")
        if res:
            print(f"        !! answered: {res[0]['answer']!r} "
                  f"facts={res[0]['query_facts']}")
    print(f"\n   abstained on {abst}/{len(ABSTAIN_QS)} ungrounded questions")


if __name__ == "__main__":
    main()
