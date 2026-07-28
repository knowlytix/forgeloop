#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Stage E (Chunk-and-Pray Ch12): calibrate the GEOMETRIC relevance gate on a
labeled query cohort WITH negatives -- no LLM judge.

The geometric gate (GeometricRelevanceGate) accepts a question for a bound head H
when ``best_cos = max_r cos(question, phrase(r))`` over H's relations clears
``tau_accept``, then vetoes when u-space tension to the accepted relation exceeds
a per-relation cut. The build's ``calibrate_relevance_thresholds`` sets both
recall-first from POSITIVES ONLY, so ``tau_accept`` pins at 0.05 and out-of-corpus
questions (force-bound to a real edge) slip through.

This recalibrates ``tau_accept`` by max-separation of positive vs negative
``best_cos`` under a false-accept ceiling -- the same discipline as the accept and
cap gates -- and recomputes per-relation u-veto cuts that keep positives and
reject the negatives that bind to each relation. Writes relevance_calibration.json.

Run on spark-ef84:
    python scripts/calibrate_policy_relevance_geo.py --store-path data/gms_policy_store_cap
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)

from scripts.calibrate_policy_rag import QUERY_PHRASINGS, NEGATIVES

_MAX_FALSE_ACCEPT = 0.05
_ACCEPT_MARGIN = 0.02   # nudge tau below the lowest kept positive, above negatives


def _sweep_tau_accept(pos, neg, ceiling):
    """Largest tau (most selective) with false-accept <= ceiling that keeps the
    most positives; max balanced accuracy as the objective."""
    cands = sorted(set(pos + neg))
    best = None
    for c in cands:
        tau = c
        recall = sum(p >= tau for p in pos) / len(pos)
        fa = sum(n >= tau for n in neg) / len(neg)
        bal = (recall + (1 - fa)) / 2
        key = (fa <= ceiling, bal, recall, -fa)
        if best is None or key > best[0]:
            best = (key, tau, recall, fa, bal)
    return best[1], best[2], best[3], best[4]


def main() -> int:
    import torch  # noqa: F401
    from knowlytix.embedding import FineTunedEmbedding
    from knowlytix.knowledge.rag.relevance import GeometricRelevanceGate, _strip_has, _tension

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", default="data/gms_policy_store_cap")
    args = ap.parse_args()
    sp = Path(args.store_path)

    ft = FineTunedEmbedding.load(str(sp / "tuned_encoder"))
    u_enc = FineTunedEmbedding.load(str(sp / "contradiction_encoder"))

    # Bind each query to its head via the production parse-bind, then score it the
    # way the gate does (best_cos over the head's relations + u-tension to best_r).
    from agentlab.capstone.policy_rag import PolicyRagRetriever
    os.environ["AGENTLAB_RAG_RELEVANCE"] = "0"     # bind only; we score relevance here
    rag = PolicyRagRetriever(store_path=str(sp))
    gate = GeometricRelevanceGate(rag.store, ft.encode, u_enc.encode)

    def score(question):
        ex = rag.extract(question)
        facts = ex["query_facts"]
        if not facts:
            return None                      # did not bind -> abstains anyway
        head = facts[0][0]
        rels = gate.head_relations(head)
        if not rels:
            return None
        phrases = [_strip_has(r).replace("_", " ") for r in rels]
        v = gate._enc("v", [question] + phrases)
        sims = (v[1:] @ v[0])
        bi = int(sims.argmax())
        best_r, best_cos = rels[bi], float(sims[bi])
        u = gate._enc("u", [question, phrases[bi]])
        ten = _tension(float(u[0] @ u[1]))
        return head, best_r, best_cos, ten

    positives = [q for qs in QUERY_PHRASINGS.values() for q in qs]
    rows = {"pos": [], "neg": []}
    print("=== per-query relevance scores (best_cos = v-accept signal) ===")
    for label, qs in (("pos", positives), ("neg", NEGATIVES)):
        for q in qs:
            s = score(q)
            if s is None:
                # unbound -> treated as abstain (best_cos = -1 so any tau rejects)
                rows[label].append((q, None, None, -1.0, None))
                print(f"  [{label}] UNBOUND (abstains)                         {q!r}")
                continue
            head, best_r, best_cos, ten = s
            rows[label].append((q, head, best_r, best_cos, ten))
            print(f"  [{label}] cos={best_cos:.3f} ten={ten:.3f} "
                  f"{head}/{best_r:<28} {q!r}")

    pos_cos = [r[3] for r in rows["pos"]]
    neg_cos = [r[3] for r in rows["neg"]]
    tau_raw, recall, fa, bal = _sweep_tau_accept(pos_cos, neg_cos, _MAX_FALSE_ACCEPT)
    tau_accept = round(max(0.05, tau_raw - _ACCEPT_MARGIN), 4)
    print(f"\n[v-accept] swept tau_accept={tau_accept} "
          f"(raw cut {tau_raw:.3f} - margin {_ACCEPT_MARGIN})  "
          f"recall={recall:.2f} false_accept={fa:.2f} balanced_acc={bal:.2f} "
          f"(ceiling {_MAX_FALSE_ACCEPT})")

    # Per-relation u-veto cut: keep positives that bind to R, reject negatives that
    # bind to R. Recall-first: max(pos tension to R) + margin, but never above the
    # min negative tension (so a binding negative is still vetoed when separable).
    per_rel = {}
    for r in sorted({row[2] for row in rows["pos"] + rows["neg"] if row[2]}):
        pos_t = [row[4] for row in rows["pos"] if row[2] == r and row[4] is not None]
        neg_t = [row[4] for row in rows["neg"] if row[2] == r and row[4] is not None]
        if not pos_t:
            continue
        cut = max(pos_t) + 0.02
        if neg_t and min(neg_t) <= cut:      # overlap: clamp toward rejecting negs
            cut = (max(pos_t) + min(neg_t)) / 2
        per_rel[r] = round(cut, 4)
    default_cut = round(sorted(per_rel.values())[len(per_rel) // 2], 4) if per_rel else 1.0

    payload = {
        "method": "geometric_relevance_query_cohort_maxsep_v1",
        "max_false_accept": _MAX_FALSE_ACCEPT,
        "tau_accept": tau_accept,
        "tau_accept_recall": round(recall, 4),
        "tau_accept_false_accept": round(fa, 4),
        "default_tau_contra": default_cut,
        "tau_contra_per_relation": per_rel,
        "overlap": {},
        "cohort_n_pos": len(positives), "cohort_n_neg": len(NEGATIVES),
    }
    (sp / "relevance_calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n[u-veto] per-relation cuts for {len(per_rel)} relations; "
          f"default={default_cut}")
    print(f"wrote {sp/'relevance_calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
