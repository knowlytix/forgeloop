#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Ablation: where does the u-space polarity weakness come from?

policy_rag.py records that the GMS contradiction (u-space) encoder separates
polarity on SHORT value tokens (forbidden<->permitted ~0.95 tension) but goes
out-of-distribution on FULL answer sentences (consistent max 1.27 > flipped min
0.63, no separating cut). Two candidate causes:
  (a) the GMS dual-embedding adapter (full-mode projection) is fit to short-token
      geometry and distorts sentence representations -- the Stiefel hypothesis;
  (b) bi-encoder cosine-tension is simply a weak contradiction signal vs a real
      MNLI cross-encoder that jointly attends premise<->hypothesis.

This isolates them. For each polarity case (a stored policy fact + a CONSISTENT
and a CONTRADICTORY answer) we score contradiction three ways, each on the FULL
answer sentence and on a DECOMPOSED short claim:

  arm1  GMS u-encoder            cosine-tension (2 sin(theta/2), high=contra)
  arm2  raw NLI base bi-encoder  cosine-tension (no GMS adapter)
  arm3  individual MNLI          cross-encoder P(contradiction)

A method "works" if every CONSISTENT score is below every CONTRADICTORY score
(a separating cut exists). We report the consistent-max / contra-min gap per
method x granularity. Run on spark-ef84 (offline):

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=$HOME/GMS-knowlytix \
    python scripts/probe_mnli_vs_uspace.py --store-path data/gms_policy_store_cap
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)

# Each case: a stored policy fact (premise) + a consistent and a flipped answer,
# given as (full sentence, decomposed short claim).
CASES = [
    {
        "fact": "Sending personal data over an unencrypted channel is forbidden.",
        "claim_fact": "unencrypted PII: forbidden",
        "consistent": ("Personal data must not be sent over an unencrypted channel "
                       "under the bank's PII handling policy.",
                       "unencrypted PII: forbidden"),
        "flipped": ("Customers may send their social security number over email "
                    "whenever it is convenient.",
                    "unencrypted PII: permitted"),
    },
    {
        "fact": "Redaction of personal data is required.",
        "claim_fact": "redaction: required",
        "consistent": ("The policy requires personal data to be redacted before "
                       "it is shared.",
                       "redaction: required"),
        "flipped": ("Redaction of personal data is not required and can be skipped.",
                    "redaction: optional"),
    },
    {
        "fact": "Closing an account requires identity verification.",
        "claim_fact": "identity verification: required",
        "consistent": ("An account can only be closed once the customer's identity "
                       "has been verified.",
                       "identity verification: required"),
        "flipped": ("You can close an account without verifying the customer's "
                    "identity.",
                    "identity verification: not required"),
    },
    {
        "fact": "Provisional credit is issued during a dispute investigation.",
        "claim_fact": "provisional credit: issued",
        "consistent": ("While a transaction dispute is investigated, provisional "
                       "credit is issued to the customer.",
                       "provisional credit: issued"),
        "flipped": ("No provisional credit is issued while a dispute is being "
                    "investigated.",
                    "provisional credit: denied"),
    },
]

MNLI_CANDIDATES = [
    "cross-encoder/nli-deberta-v3-base",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "facebook/bart-large-mnli",
    "roberta-large-mnli",
]


def _sep(consistent: list[float], contra: list[float]) -> str:
    cmax, fmin = max(consistent), min(contra)
    ok = "SEPARABLE" if cmax < fmin else "NO CUT"
    return (f"consistent[max={cmax:.3f}] vs contra[min={fmin:.3f}]  "
            f"gap={fmin - cmax:+.3f}  -> {ok}")


def main() -> int:
    import torch
    import torch.nn.functional as F

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", default="data/gms_policy_store_cap")
    args = ap.parse_args()
    sp = Path(args.store_path)

    from knowlytix.embedding import FineTunedEmbedding
    from knowlytix.knowledge.rag.relevance import _tension

    u = FineTunedEmbedding.load(str(sp / "contradiction_encoder"))
    # The raw base encoder behind the adapter (arm 2): encode with the base model
    # only, bypassing the learned full-mode projection.
    base_name = (getattr(u, "base_model", None) or getattr(u, "model_name", None)
                 or getattr(u, "encoder", None) or "sentence-transformers/nli-mpnet-base-v2")
    print(f"u-encoder base model: {base_name}")
    from knowlytix.core.graph.encoders import encode_texts  # knowlytix's encoder loader
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def u_tension(a: str, b: str) -> float:
        v = F.normalize(torch.as_tensor(u.encode([a, b]), dtype=torch.float32), p=2, dim=-1)
        return _tension(float(v[0] @ v[1]))

    def base_tension(a: str, b: str) -> float:
        v = encode_texts([a, b], str(base_name), dev)  # raw base, no GMS adapter
        v = F.normalize(torch.as_tensor(v, dtype=torch.float32), p=2, dim=-1)
        return _tension(float(v[0] @ v[1]))

    # arm 3: a real MNLI cross-encoder -> P(contradiction).
    mnli = None
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    for name in MNLI_CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForSequenceClassification.from_pretrained(name)
            mdl.eval()
            id2label = {i: l.lower() for i, l in mdl.config.id2label.items()}
            ci = next((i for i, l in id2label.items() if "contrad" in l), None)
            if ci is None:
                continue
            mnli = (name, tok, mdl, ci)
            print(f"MNLI cross-encoder: {name}  (contradiction id={ci})")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  (skip {name}: {type(e).__name__})")
    if mnli is None:
        print("MNLI: no cross-encoder available offline -- arm 3 skipped")

    def mnli_contra(premise: str, hypothesis: str) -> float:
        _name, tok, mdl, ci = mnli
        with torch.no_grad():
            x = tok(premise, hypothesis, return_tensors="pt", truncation=True)
            p = F.softmax(mdl(**x).logits, dim=-1)[0]
        return float(p[ci])

    arms = [("arm1 GMS u-encoder tension", u_tension),
            ("arm2 raw NLI base tension ", base_tension)]
    if mnli is not None:
        arms.append(("arm3 MNLI P(contradiction)", mnli_contra))

    for gran, idx in (("FULL sentence", 0), ("DECOMPOSED claim", 1)):
        print(f"\n################  {gran}  ################")
        for name, fn in arms:
            cons, flip = [], []
            for c in CASES:
                prem = c["fact"] if "MNLI" in name else (
                    c["claim_fact"] if gran.startswith("DECOMP") else c["fact"])
                cons.append(fn(prem, c["consistent"][idx]))
                flip.append(fn(prem, c["flipped"][idx]))
            print(f"  {name}: {_sep(cons, flip)}")
            for c, sc, sf in zip(CASES, cons, flip):
                print(f"      {c['claim_fact']:34s} consistent={sc:.3f}  flipped={sf:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
