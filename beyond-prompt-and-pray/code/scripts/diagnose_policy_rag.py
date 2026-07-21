#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Per-stage diagnostic for the banking GEODE-RAG pipeline. For each question it
walks the real pipeline stage by stage and prints the internal signals so a
failure is localized to ONE stage:

  1. PARSE   -- the geometric parser's query triples.
  2. BIND    -- the bound (head, relation, tail) + bound flag (entity binding).
  3. RELEVANCE -- for every bound head: the head's asserted relations, the FULL
     per-relation v-cosine(question, relation-phrase) and u-tension, the gate's
     accept floor (tau_accept) and per-relation u-veto cut (tau_contra), and the
     gate verdict. This is the exact computation in GeometricRelevanceGate, with
     every candidate exposed (not just the chosen best), so we can see whether the
     correct relation was even the nearest, and whether a cut -- not the signal --
     is what rejects it.
  4. RETRIEVE -- the grounded facts that came back + confidence.
  5. VERIFY   -- the self-verification verdicts.
  6. FINAL    -- decision / route / notice.

Read the raw v-cosine / u-tension numbers, not the verdict: the cuts on the store
may be stale, but the raw signal tells us whether the stage CAN separate.

Run on spark-ef84 (offline):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH=$HOME/jupyterlab/GMS-knowlytix \
      python scripts/diagnose_policy_rag.py --store-path data/gms_policy_store_cap
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)

# (question, expected_head, note). expected_head=None for out-of-corpus negatives
# (should abstain). The set mixes known-failing routes, passing controls and
# clear negatives so each row's stage breakdown is interpretable.
CASES = [
    # --- known-failing routes -------------------------------------------------
    ("When does Regulation E apply to my dispute?",       "reg_e",       "FAIL route"),
    ("What is the threshold for Regulation Z disclosures?", "reg_z",      "FAIL route"),
    ("Does Regulation X cover my mortgage servicing?",     "reg_x",       "FAIL route"),
    ("My social security number was emailed unencrypted.", "pii_handling", "FAIL route"),
    ("Someone leaked my personal information.",            "pii_handling", "FAIL route"),
    ("Escalate this unfair fee to compliance.",            "regulatory_escalation", "FAIL route"),
    # --- passing controls -----------------------------------------------------
    ("Why was I charged a $35 overdraft fee?",             "overdraft",   "PASS control"),
    ("I want to dispute a charge I never authorized.",     "disputes",    "PASS control"),
    ("How long do I have to dispute a charge?",            "disputes",    "PASS control"),
    # --- out-of-corpus negatives (must abstain) -------------------------------
    ("What is the capital of France?",                     None,          "NEG abstain"),
    ("What is the overdraft interest rate?",               None,          "NEG no-attr"),
]


def _fmt(x, n=3):
    try:
        return f"{float(x):.{n}f}"
    except Exception:  # noqa: BLE001
        return str(x)


def main() -> int:
    import torch
    from knowlytix.knowledge.rag.relevance import _strip_has, _tension

    os.environ.setdefault("AGENTLAB_RAG_UVETO", "1")        # expose the u-encoder
    os.environ.setdefault("AGENTLAB_RAG_ACCEPT_THRESHOLD", "0.0")  # don't pre-filter
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", default="data/gms_policy_store_cap")
    args = ap.parse_args()

    from agentlab.capstone.policy_rag import PolicyRagRetriever
    rag = PolicyRagRetriever(store_path=str(args.store_path))
    pipe = rag.pipe
    gate = pipe.relevance
    print(f"store={args.store_path}  LLM={rag.llm.model_name}  "
          f"u-veto={'ON' if getattr(gate, '_u', None) is not None else 'OFF'}")
    print(f"tau_accept={_fmt(gate.tau_accept,4)}  "
          f"default_tau_contra={_fmt(gate.tau_contra,4)}  "
          f"per-relation cuts={len(gate.tau_contra_per_relation)}")

    def relevance_breakdown(question: str, head: str) -> None:
        rels = gate.head_relations(head)
        if not rels:
            print(f"      head {head!r} has NO content relations")
            return
        phrases = [_strip_has(r).replace("_", " ") for r in rels]
        v = gate._enc("v", [question] + phrases)
        sims = (v[1:] @ v[0]).tolist()
        # u-tension for every relation (the gate computes it only for the best;
        # we show all so a wrongly-vetoed correct relation is visible).
        tens = []
        for ph in phrases:
            u = gate._enc("u", [question, ph])
            tens.append(_tension(float(u[0] @ u[1])))
        order = sorted(range(len(rels)), key=lambda i: sims[i], reverse=True)
        bi = order[0]
        print(f"      head={head!r}  rels={len(rels)}  "
              f"v-best={rels[bi]} ({_fmt(sims[bi])})  "
              f"{'>= ' if sims[bi] >= gate.tau_accept else '< '}tau_accept "
              f"{_fmt(gate.tau_accept)}")
        for i in order:
            tau_c = gate.tau_contra_per_relation.get(rels[i], gate.tau_contra)
            veto = " <-VETO" if (i == bi and tens[i] > tau_c) else ""
            star = " *" if i == bi else "  "
            print(f"       {star} {rels[i]:34s} v={_fmt(sims[i])}  "
                  f"u-tension={_fmt(tens[i])}  tau_contra={_fmt(tau_c)}{veto}")
        matched, detail = gate.relevant_relation(question, head)
        print(f"      gate verdict: matched={matched!r}  ({detail})")

    for q, exp, note in CASES:
        print("\n" + "=" * 78)
        print(f"Q: {q!r}   [exp={exp!r} | {note}]")

        # 1-2. parse + bind
        ex = pipe.extract(q)
        bound = list(getattr(ex, "bound_facts", []) or [])
        print(f"  PARSE/BIND: is_bound={ex.is_bound}  bound_facts={bound}")
        heads = []
        for h, r, t in bound:
            if h and not h.startswith("?") and h not in heads:
                heads.append(h)

        # 3. relevance internals per bound head
        if heads:
            print("  RELEVANCE:")
            for h in heads:
                relevance_breakdown(q, h)
        else:
            print("  RELEVANCE: (no concrete bound head to judge)")

        # 4-6. retrieve / verify / final (full pipeline, no synthesis)
        ans = pipe.query(q, generate=False)
        srcs = [(s.head, s.relation, s.tail, round(float(getattr(s, "score", 0.0)), 3))
                for s in (ans.sources or [])]
        print(f"  RETRIEVE: confidence={_fmt(ans.confidence)}  sources={srcs}")
        print(f"  VERIFY:   {ans.verification or {}}")
        print(f"  FINAL:    decision={ans.decision}  route={ans.route}  "
              f"verified={ans.verified}  notice={ans.notice!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
