#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Does u-space separate value SYNONYMS (low tension) from ANTONYMS (high)?

The dual split: v-space = identity/synonymy, u-space = polarity/contradiction. A
verifier value check wants: a synonym of the stored value -> CONSISTENT (low
u-tension), an opposite-polarity value -> CONTRADICTORY (high). Test the store's
contradiction (u) encoder on value tokens directly. If it separates cleanly, the
verifier uses u-tension (synonym-tolerant by construction); if not, the u-encoder
needs SFT on value-polarity pairs (it was trained on RELATION phrasings).
"""
import os, sys
sys.path.insert(0, os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/GMS-knowlytix")))
import torch
import torch.nn.functional as F
from knowlytix.embedding import FineTunedEmbedding
from knowlytix.knowledge.rag.relevance import _tension

SP = sys.argv[1] if len(sys.argv) > 1 else "data/gms_policy_store_cap_actor"
u = FineTunedEmbedding.load(os.path.join(SP, "contradiction_encoder"))

# value: (synonyms [want LOW tension vs value], antonyms [want HIGH])
GROUPS = [
    ("forbidden", ["prohibited", "banned", "disallowed", "not allowed"],
                  ["permitted", "allowed", "acceptable"]),
    ("required",  ["mandatory", "compulsory", "obligatory"],
                  ["optional", "voluntary", "discretionary"]),
    ("issued",    ["granted", "provided", "given"],
                  ["denied", "withheld", "refused"]),
    ("permitted", ["allowed", "acceptable", "fine"],
                  ["forbidden", "prohibited", "disallowed"]),
]

def ten(a, b):
    v = F.normalize(torch.as_tensor(u.encode([a, b]), dtype=torch.float32), p=2, dim=-1)
    return _tension(float(v[0] @ v[1]))

print(f"u-encoder: {SP}/contradiction_encoder\n")
syn_all, ant_all = [], []
for val, syns, ants in GROUPS:
    sy = [(s, ten(val, s)) for s in syns]
    an = [(a, ten(val, a)) for a in ants]
    syn_all += [t for _, t in sy]
    ant_all += [t for _, t in an]
    print(f"{val}:")
    print(f"  synonyms (want LOW):  " + ", ".join(f"{s}={t:.2f}" for s, t in sy))
    print(f"  antonyms (want HIGH): " + ", ".join(f"{a}={t:.2f}" for a, t in an))

smax, amin = max(syn_all), min(ant_all)
print(f"\n=== global ===")
print(f"synonym tension band: [{min(syn_all):.3f}, {smax:.3f}]")
print(f"antonym tension band: [{amin:.3f}, {max(ant_all):.3f}]")
print(f"separable: {'YES' if smax < amin else 'NO (overlap)'}  "
      f"(syn_max={smax:.3f} {'<' if smax < amin else '>='} ant_min={amin:.3f})")
if smax < amin:
    print(f"  -> u-space cut tau ~ {(smax+amin)/2:.3f} separates synonym from antonym")
