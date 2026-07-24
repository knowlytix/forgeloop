#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Stage C (Chunk-and-Pray Ch9): grounded synthesis on Qwen3-4B.

Runs the banking RAG retriever (search_policy's engine) over a handful of policy
questions and prints, for each, the synthesized answer alongside the grounded
``(h,r,t)`` facts and their provenance spans. The check is that every figure in
the answer appears in a retrieved span -- the model verbalizes the evidence, it
does not invent fees or deadlines -- and that an out-of-corpus question abstains
rather than guess.

Run on spark-ef84:
    PYTHONPATH=$HOME/GMS-knowlytix \
      python scripts/probe_policy_synthesis.py --store-path data/gms_policy_store_cap
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# Make the repo root (agentlab) and the knowlytix branch importable regardless of
# how the script is launched / shell quoting.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_KNOW = os.environ.get("KNOWLYTIX_SRC",
                       os.path.expanduser("~/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)

from agentlab.capstone.policy_rag import PolicyRagRetriever

# Grounded policy questions (answers are in the corpus) + an out-of-corpus one.
QUESTIONS = [
    "What is the overdraft fee?",
    "How many days do I have to file a transaction dispute?",
    "How much can a representative reverse without manager approval?",
    "Within how many days must the bank notify me before closing my account?",
    "What is the weather forecast for tomorrow?",   # out of corpus -> abstain
]

_NUM = re.compile(r"\d+(?:\.\d+)?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", default="data/gms_policy_store_cap")
    args = ap.parse_args()

    rag = PolicyRagRetriever(store_path=args.store_path)
    print(f"RAG LLM: {rag.llm.model_name}   store: {args.store_path}")
    print(f"accept_threshold(tau)={rag.accept_threshold}  tuned_encoder={rag.tuned_encoder}\n")

    grounded_ok = 0
    for q in QUESTIONS:
        res = rag.search(q)
        print("=" * 78)
        print("Q:", q)
        if not res:
            print("  -> ABSTAIN (no grounded evidence survived the gates)")
            continue
        r = res[0]
        spans = r["text"]
        ans = r["answer"]
        print(f"  decision={r['decision']} route={r['route']} verified={r['verified']} "
              f"score={r['score']:.3f}")
        print(f"  policy id: {r['id']}   policies: {r['policies']}")
        print(f"  grounded facts: {r['query_facts']}")
        print(f"  provenance spans:\n      " + spans.replace('\n', '\n      '))
        print(f"  ANSWER: {ans}")
        # numbers-in-answer must appear in the grounded spans (no invented figures)
        ans_nums = {f"{float(x):g}" for x in _NUM.findall(ans)}
        span_nums = {f"{float(x):g}" for x in _NUM.findall(spans)}
        invented = ans_nums - span_nums
        print(f"  numeric check: answer_nums={sorted(ans_nums)} "
              f"span_nums={sorted(span_nums)} invented={sorted(invented) or 'NONE'}")
        grounded_ok += not invented

    print("\n" + "=" * 78)
    print(f"grounded answers with no invented number: {grounded_ok}/{len(QUESTIONS)-1} "
          f"(the 5th is the out-of-corpus abstain check)")


if __name__ == "__main__":
    main()
