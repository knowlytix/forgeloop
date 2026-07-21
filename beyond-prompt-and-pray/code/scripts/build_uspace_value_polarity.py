#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""u-space SFT for VALUE polarity (not relation phrasings).

The stock contradiction encoder is trained on relation phrasings, so on value
tokens it only weakly separates synonyms from antonyms. Here we finetune a u-space
encoder whose groups are per-VALUE stance clusters (a stored value + its synonyms);
same-group (synonyms) -> consistent (low tension), cross-group (antonyms + other
values) -> contradictory (high). Synonyms are generated for the stance values with
the actor (the "synonyms for entities" step), then fed to finetune_contradiction.
Saves value_polarity_encoder/ and validates synonym(low) vs antonym(high) separation.

Run on spark-ef84.
"""
import os, sys, json
sys.path.insert(0, os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix")))
import torch
import torch.nn.functional as F
from pathlib import Path
from knowlytix.embedding import EmbeddingSFTConfig, finetune_contradiction
from knowlytix.knowledge.geode.alias_gen import generate_entity_aliases
from knowlytix.knowledge.geode import QWEN_4B
from knowlytix.knowledge.rag.relevance import _tension

SP = sys.argv[1] if len(sys.argv) > 1 else "data/gms_policy_store_cap_actor"
# Stance poles present in the bank policy store (+ their negations as contrast).
POLES = ["forbidden", "permitted", "required", "optional", "issued", "denied"]
# antonym pairs for VALIDATION (not used in training; training is group membership).
ANTONYM = {"forbidden": "permitted", "permitted": "forbidden", "required": "optional",
           "optional": "required", "issued": "denied", "denied": "issued"}

print("generating value synonyms with the actor ...", flush=True)
syns = generate_entity_aliases(POLES, llm=QWEN_4B, n_aliases=6, device="cuda",
                               max_tokens=96)
groups = {v: sorted({v, *syns.get(v, [])}) for v in POLES}
for v, g in groups.items():
    print(f"  {v}: {g}")

cfg = EmbeddingSFTConfig(rank=8, mode="full", objective="contradiction",
                         epochs=400, margin=1.3, device="cuda")
print("\nfinetune_contradiction on value-polarity groups ...", flush=True)
u = finetune_contradiction(groups, cfg)
out = Path(SP) / "value_polarity_encoder"
u.save(out)
print(f"saved {out}", flush=True)

def ten(a, b):
    v = F.normalize(torch.as_tensor(u.encode([a, b]), dtype=torch.float32), p=2, dim=-1)
    return _tension(float(v[0] @ v[1]))

print("\n=== validation: synonym(low) vs antonym(high) ===")
syn_all, ant_all = [], []
for v in POLES:
    sy = [(s, ten(v, s)) for s in groups[v] if s != v]
    an = [(a, ten(v, a)) for a in groups.get(ANTONYM[v], [ANTONYM[v]])]
    syn_all += [t for _, t in sy]; ant_all += [t for _, t in an]
    print(f"{v}: syn " + ", ".join(f"{s}={t:.2f}" for s, t in sy) +
          " | ant " + ", ".join(f"{a}={t:.2f}" for a, t in an))
smax, amin = max(syn_all), min(ant_all)
tau = round((smax + amin) / 2, 3)
print(f"\nsynonym band [{min(syn_all):.3f},{smax:.3f}]  antonym band [{amin:.3f},{max(ant_all):.3f}]")
print(f"separable: {'YES' if smax < amin else 'NO'}  tau~{tau}")
json.dump({"tau_polarity": tau, "syn_band": [min(syn_all), smax],
           "ant_band": [amin, max(ant_all)], "separable": smax < amin},
          open(Path(SP) / "value_polarity_calibration.json", "w"), indent=2)
