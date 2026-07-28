#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""u-space polarity SFT from per-pole LLM-materialized synonyms (Qwen 4B).

The LLM is a build-time materializer (like the DOE rephraser): one focused call per
stance pole yields clean synonyms. The 6 poles become 6 DISJOINT groups (pole +
synonyms); a token appearing in >1 group is dropped (ambiguous supervision). The
antonym contrast is inherent -- opposite poles are different groups, so
finetune_contradiction drives them to high tension and synonyms within a pole to
low. Pure geometry at runtime.
"""
import os, sys, re
sys.path.insert(0, os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/GMS-knowlytix")))
import torch, torch.nn.functional as F
from pathlib import Path
from collections import Counter
from knowlytix.knowledge.llm_backend import LocalTransformersBackend
from knowlytix.knowledge.geode import QWEN_4B
from knowlytix.embedding import EmbeddingSFTConfig, finetune_contradiction
from knowlytix.knowledge.rag.relevance import _tension

SP = sys.argv[1] if len(sys.argv) > 1 else "data/gms_policy_store_cap_syn"
# pole -> (gloss for disambiguation, opposite pole for validation)
POLES = {
    "forbidden": ("not allowed / prohibited", "permitted"),
    "permitted": ("allowed / authorized", "forbidden"),
    "required": ("mandatory / must be done", "optional"),
    "optional": ("not mandatory / voluntary", "required"),
    "issued": ("granted / provided", "denied"),
    "denied": ("refused / withheld", "issued"),
}
llm = LocalTransformersBackend(QWEN_4B, device="cuda")

def synonyms(term, gloss):
    raw = llm.call(
        system="You list precise single-word or short-phrase synonyms for a bank-"
               "policy stance term. Output only a comma-separated list, no numbering.",
        user=f"Synonyms of the policy stance '{term}' (meaning: {gloss}). 6 items.",
        max_tokens=80)
    out = []
    for tok in re.split(r"[,\n]", raw):
        s = tok.strip(" -*\t.").strip().lower()
        if 2 <= len(s) <= 28 and s not in out:
            out.append(s)
    return out[:6]

raw_groups = {}
for t, (gloss, _) in POLES.items():
    syn = synonyms(t, gloss)
    raw_groups[t] = sorted({t, *syn})
    print(f"  {t}: {raw_groups[t]}")

# Drop tokens appearing in >1 group (ambiguous -> would teach contradictory labels).
counts = Counter(tok for g in raw_groups.values() for tok in g)
groups = {t: [tok for tok in g if counts[tok] == 1] for t, g in raw_groups.items()}
groups = {t: (g if t in g else [t] + g) for t, g in groups.items()}  # keep the pole itself
dropped = [tok for tok, c in counts.items() if c > 1]
print(f"\ndropped ambiguous (in >1 group): {dropped}")

cfg = EmbeddingSFTConfig(rank=8, mode="full", objective="contradiction",
                         epochs=400, margin=1.3, device="cuda")
u = finetune_contradiction(groups, cfg)
u.save(Path(SP) / "value_polarity_encoder")
print(f"saved {SP}/value_polarity_encoder ({len(groups)} disjoint stance groups)")

def ten(a, b):
    v = F.normalize(torch.as_tensor(u.encode([a, b]), dtype=torch.float32), p=2, dim=-1)
    return _tension(float(v[0] @ v[1]))
syn_t, ant_t = [], []
print("\nvalidation (synonym=LOW vs antonym=HIGH):")
for t, (_, opp) in POLES.items():
    sy = [ten(t, s) for s in groups[t] if s != t]
    an = [ten(t, s) for s in groups.get(opp, [opp]) if s != opp] + [ten(t, opp)]
    syn_t += sy; ant_t += an
    print(f"  {t}: syn[{min(sy):.2f},{max(sy):.2f}] vs ant({opp})[{min(an):.2f},{max(an):.2f}]")
print(f"\nGLOBAL synonym [{min(syn_t):.2f},{max(syn_t):.2f}]  antonym [{min(ant_t):.2f},{max(ant_t):.2f}]  "
      f"separable={max(syn_t) < min(ant_t)}")
