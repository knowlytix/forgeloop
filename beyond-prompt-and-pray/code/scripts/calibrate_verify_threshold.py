#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Part A: can a calibrated verifier threshold separate synonyms from reversals?

On a store, score (via score_triple_emb on the tuned v-embedding) a cohort of
value claims per polarity relation: POSITIVES = the consistent value + its
synonyms (should be SUPPORTED), NEGATIVES = the opposite-polarity value + its
synonyms (should be CONTRADICTED). Report the score bands and the best recall-first
threshold tau_verify = max(positive) + margin, plus any false-accepts (a negative
at or below tau). Compare the plain admissibility cap vs a calibrated tau.

Usage: python calibrate_verify_threshold.py <store_path>
"""
import os, sys, json
sys.path.insert(0, os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/GMS-knowlytix")))
import torch
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
from knowlytix.embedding import FineTunedEmbedding

SP = sys.argv[1] if len(sys.argv) > 1 else "data/gms_policy_store_cap_v2"
MARGIN = 0.05

# (head, relation, consistent + synonyms [positives], opposite + synonyms [negatives])
COHORT = [
    ("pii_handling", "has_unencrypted_channel_pii",
     ["forbidden", "prohibited", "banned", "disallowed", "not allowed"],
     ["permitted", "allowed", "acceptable", "fine"]),
    ("pii_handling", "has_redaction",
     ["required", "mandatory", "compulsory", "obligatory"],
     ["optional", "not required", "voluntary", "discretionary"]),
    ("account_closure", "has_identity_verification",
     ["required", "mandatory", "needed", "necessary"],
     ["optional", "not required", "waived", "not needed"]),
    ("account_closure", "has_fraud_notice_exception",
     ["permitted", "allowed", "acceptable"],
     ["forbidden", "prohibited", "disallowed"]),
    ("disputes", "has_provisional_credit",
     ["issued", "granted", "provided", "given"],
     ["denied", "withheld", "refused", "not issued"]),
]

store = GMSExpertStore(DocGMSConfig(store_path=SP, ingest_mode="regex"))
assert store.load()
ft = FineTunedEmbedding.load(os.path.join(SP, "tuned_encoder"))

def sc(h, r, val):
    return store.score_triple_emb(h, r, ft.encode([val])[0])

all_pos, all_neg = [], []
caps = []
print(f"store: {SP}\n")
for h, r, pos, neg in COHORT:
    cap = store.cap_radius(r); caps.append(cap)
    ps = [(v, sc(h, r, v)) for v in pos]
    ns = [(v, sc(h, r, v)) for v in neg]
    all_pos += [s for _, s in ps if s is not None]
    all_neg += [s for _, s in ns if s is not None]
    print(f"{r}  (cap={cap:.3f})")
    print(f"  positives (want supported):   " +
          ", ".join(f"{v}={s:.2f}" for v, s in ps))
    print(f"  negatives (want contradicted): " +
          ", ".join(f"{v}={s:.2f}" for v, s in ns))
    within_cap_pos = sum(s <= cap for _, s in ps if s is not None)
    print(f"  -> positives within plain cap: {within_cap_pos}/{len(ps)} "
          f"(rest would FALSE-FAIL under the cap)")

pmax, nmin = max(all_pos), min(all_neg)
tau = round(pmax + MARGIN, 3)
fa = [s for s in all_neg if s <= tau]
fr = [s for s in all_pos if s > tau]
print(f"\n=== global ===")
print(f"positives band: [{min(all_pos):.3f}, {pmax:.3f}]  "
      f"negatives band: [{nmin:.3f}, {max(all_neg):.3f}]")
print(f"plain cap ~{sum(caps)/len(caps):.3f}: positives within cap = "
      f"{sum(s <= c for s in all_pos for c in [sum(caps)/len(caps)])}/{len(all_pos)}")
print(f"calibrated tau_verify = max(pos)+{MARGIN} = {tau}")
print(f"  separable: {'YES' if pmax < nmin else 'NO (overlap)'}  "
      f"(pos_max={pmax:.3f} {'<' if pmax<nmin else '>='} neg_min={nmin:.3f})")
print(f"  false-accepts (neg <= tau): {len(fa)}   false-rejects (pos > tau): {len(fr)}")
json.dump({"tau_verify": tau, "pos_band": [min(all_pos), pmax],
           "neg_band": [nmin, max(all_neg)], "separable": pmax < nmin,
           "mean_cap": sum(caps)/len(caps)},
          open(os.path.join(SP, "verify_threshold_calibration.json"), "w"), indent=2)
print(f"\nwrote {SP}/verify_threshold_calibration.json")
