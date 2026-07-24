#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Build + calibrate the fused value-polarity verifier artifacts for the capstone
policy store, then validate the checker.

Produces two artifacts inside the store passed as argv[1] (default
data/gms_policy_store_cap):

  * ``value_polarity_encoder/`` -- u-space polarity encoder. Qwen 4B materializes
    per-pole synonyms at build time (like the DOE rephraser); the 6 poles become 6
    disjoint stance groups; ``finetune_contradiction`` drives opposite poles apart
    and same-pole synonyms together. Runtime is pure geometry.
  * ``value_polarity_calibration.json`` -- the 3-class u-tension cuts
    (tau_ent, tau_contra), CV-gated via ``PolarityCuts.calibrate``.

Then loads the store's tuned v-encoder + the new u-encoder + the cuts into a
``ValuePolarityChecker`` and prints the PII-forbidden verdicts as a smoke test.

GPU (cuda) required for the SFT; run on spark-ef84.
"""
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.environ.get(
    "KNOWLYTIX_SRC", os.path.expanduser("~/GMS-knowlytix")))

import torch
import torch.nn.functional as F
from knowlytix.embedding import EmbeddingSFTConfig, finetune_contradiction, FineTunedEmbedding
from knowlytix.knowledge.geode import QWEN_4B
from knowlytix.knowledge.llm_backend import LocalTransformersBackend
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
from knowlytix.knowledge.rag import (PolarityCuts, ValuePolarityChecker,
                                     polarity_tension_cohort)
from knowlytix.knowledge.rag.relevance import _tension

STORE = sys.argv[1] if len(sys.argv) > 1 else "data/gms_policy_store_cap"


def log(*a):
    print(*a, flush=True)


# pole -> (gloss for disambiguation, opposite pole for validation)
POLES = {
    "forbidden": ("not allowed / prohibited", "permitted"),
    "permitted": ("allowed / authorized", "forbidden"),
    "required": ("mandatory / must be done", "optional"),
    "optional": ("not mandatory / voluntary", "required"),
    "issued": ("granted / provided", "denied"),
    "denied": ("refused / withheld", "issued"),
}

log(f"[1/4] materializing per-pole synonyms with {QWEN_4B} ...")
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
    log(f"  {t}: {raw_groups[t]}")

# Drop tokens appearing in >1 group (ambiguous -> contradictory supervision).
counts = Counter(tok for g in raw_groups.values() for tok in g)
groups = {t: [tok for tok in g if counts[tok] == 1] for t, g in raw_groups.items()}
groups = {t: (g if t in g else [t] + g) for t, g in groups.items()}  # keep the pole
dropped = [tok for tok, c in counts.items() if c > 1]
log(f"  dropped ambiguous (in >1 group): {dropped}")

log("[2/4] SFT u-space polarity encoder (contradiction objective) ...")
cfg = EmbeddingSFTConfig(rank=8, mode="full", objective="contradiction",
                         epochs=400, margin=1.3, device="cuda")
u = finetune_contradiction(groups, cfg)
u.save(Path(STORE) / "value_polarity_encoder")
log(f"  saved {STORE}/value_polarity_encoder ({len(groups)} disjoint stance groups)")


def ten(a, b):
    v = F.normalize(torch.as_tensor(u.encode([a, b]), dtype=torch.float32), p=2, dim=-1)
    return _tension(float(v[0] @ v[1]))


syn_t, ant_t = [], []
for t, (_, opp) in POLES.items():
    sy = [ten(t, s) for s in groups[t] if s != t]
    an = [ten(t, s) for s in groups.get(opp, [opp]) if s != opp] + [ten(t, opp)]
    syn_t += sy
    ant_t += an
log(f"  SFT check: synonym[{min(syn_t):.2f},{max(syn_t):.2f}] "
    f"antonym[{min(ant_t):.2f},{max(ant_t):.2f}] separable={max(syn_t) < min(ant_t)}")

log("[3/4] calibrating 3-class polarity cuts (5-fold CV gated) ...")
# Each axis pairs opposite stances; the lists are same-stance synonyms.
axes = [
    ("forbidden", ["prohibited", "banned", "disallowed", "outlawed"],
     "permitted", ["allowed", "authorized", "approved"]),
    ("required", ["mandatory", "compulsory", "obligatory", "enforced"],
     "optional", ["voluntary", "discretionary"]),
    ("issued", ["granted", "provided", "awarded", "conferred"],
     "denied", ["withheld", "refused", "rejected"]),
]
tension, labels = polarity_tension_cohort(u.encode, axes)
cuts = PolarityCuts.calibrate(tension, labels)
if cuts is None:
    log("  DEGENERATE: CV gate not cleared; no cuts fit. Aborting.")
    sys.exit(1)
CAL = os.path.join(STORE, "value_polarity_calibration.json")
cuts.save(CAL)
log(f"  cohort samples : {len(tension)}")
log(f"  tau_ent        : {cuts.tau_ent:.4f}")
log(f"  tau_contra     : {cuts.tau_contra:.4f}")
log(f"  cv_accuracy    : {cuts.cv_accuracy}")
log(f"  CV gate (>=0.60) cleared: {cuts.cv_accuracy is not None and cuts.cv_accuracy >= 0.60}")
log(f"  saved {CAL}")

log("[4/4] validating ValuePolarityChecker on the store ...")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
store = GMSExpertStore(DocGMSConfig(store_path=STORE, ingest_mode="regex"), device=dev)
assert store.load(), f"failed to load store {STORE}"
v = FineTunedEmbedding.load(os.path.join(STORE, "tuned_encoder"))
checker = ValuePolarityChecker(store, v.encode, u.encode, PolarityCuts.load(CAL))
rule = ("pii_handling", "has_unencrypted_channel_pii", "forbidden")
log(f"  rule = {rule} (stored stance: forbidden)")
for val in ["forbidden", "prohibited", "permitted", "allowed", "optional"]:
    log(f"    {val:11s} -> {checker.check(rule[0], rule[1], val, rule[2])}")
log("DONE.")
