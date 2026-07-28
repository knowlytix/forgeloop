#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Does the alias-tuned v-encoder RESOLVE a synonym to its value entity?

The right v-space synonym measure is identity (nearest entity), not cap distance:
if "prohibited" binds to the entity "forbidden", the verifier resolves the asserted
value to the stored one and plausibility is exact. For each synonym, report its
nearest entity (by tuned-encoder v-embedding cosine) and whether it is the intended
value, plus the margin to the runner-up.
"""
import os, sys
sys.path.insert(0, os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/GMS-knowlytix")))
import torch, torch.nn.functional as F
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
from knowlytix.embedding import FineTunedEmbedding

SP = sys.argv[1] if len(sys.argv) > 1 else "data/gms_policy_store_cap_syn"
store = GMSExpertStore(DocGMSConfig(store_path=SP, ingest_mode="regex"))
assert store.load()
ft = FineTunedEmbedding.load(os.path.join(SP, "tuned_encoder"))

ents = sorted(store.adapter.entity_to_idx)
E = F.normalize(torch.as_tensor(ft.encode(ents), dtype=torch.float32), p=2, dim=-1)

# synonym -> intended value entity
CASES = {
    "prohibited": "forbidden", "banned": "forbidden", "disallowed": "forbidden",
    "mandatory": "required", "compulsory": "required", "obligatory": "required",
    "granted": "issued", "provided": "issued", "given": "issued",
    "allowed": "permitted", "acceptable": "permitted",
    # reversals should NOT resolve to the same value as their antonym
    "optional": "(not required)", "denied": "(not issued)",
}

def nearest(tok):
    q = F.normalize(torch.as_tensor(ft.encode([tok]), dtype=torch.float32), p=2, dim=-1)[0]
    sims = E @ q
    top = torch.topk(sims, 3)
    return [(ents[i], float(sims[i])) for i in top.indices]

hits = 0; n = 0
print(f"store: {SP}\n{'synonym':14s} {'-> nearest entity':28s} {'intended':14s} hit  top3")
for tok, want in CASES.items():
    top3 = nearest(tok)
    near, s0 = top3[0]
    margin = s0 - top3[1][1]
    ok = near == want
    if want.startswith("("):
        n += 0  # reversal rows are informational
    else:
        n += 1; hits += ok
    flag = "OK " if (ok and not want.startswith("(")) else ("   " if want.startswith("(") else "XX ")
    print(f"  {tok:12s} {near+f' ({s0:.2f},Δ{margin:.2f})':28s} {want:14s} {flag} "
          + ", ".join(f"{e}={c:.2f}" for e, c in top3))
print(f"\nsynonyms resolved to intended value entity: {hits}/{n}")
