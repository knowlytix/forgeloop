#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Uncertainty-aware fusion of the three geometric value signals, with bootstrap CIs.

Signals per asserted value v (relation r, stored value s):
  d_cap = score_triple_emb(h,r,enc(v)) vs cap_radius(r)   (plausibility, v-space)
  res   = nearest entity to v + margin (top1-top2)        (identity, v-space)
  u_ten = u-tension(v, s) via tuned value_polarity_encoder (polarity, u-space)

Cascade (use the decisive measure; abstain when none is):
  within cap                              -> supported  (exact value, cap-certain)
  res decisive & near != s                -> contradicted (wrong value)
  res decisive & near == s & u low        -> supported  (synonym: same value+polarity)
  res decisive & near == s & u high       -> contradicted (reversal onto stored value)
  else if u high                          -> contradicted
  else if u low & near == s               -> supported
  else                                    -> UNCERTAIN (defer / human review)

Reports route-B-alone vs fused accuracy with 95% bootstrap CIs, and fused coverage.
"""
import os, sys
sys.path.insert(0, os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/GMS-knowlytix")))
import numpy as np
import torch, torch.nn.functional as F
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
from knowlytix.embedding import FineTunedEmbedding
from knowlytix.knowledge.rag.relevance import _tension

SP = sys.argv[1] if len(sys.argv) > 1 else "data/gms_policy_store_cap_syn"
TAU_U = float(sys.argv[2]) if len(sys.argv) > 2 else 0.62
RES_MARGIN, U_BAND = 0.08, 0.10   # decisive bands

store = GMSExpertStore(DocGMSConfig(store_path=SP, ingest_mode="regex"))
assert store.load()
v_ft = FineTunedEmbedding.load(os.path.join(SP, "tuned_encoder"))
u_ft = FineTunedEmbedding.load(os.path.join(SP, "value_polarity_encoder"))
ents = sorted(store.adapter.entity_to_idx)
E = F.normalize(torch.as_tensor(v_ft.encode(ents), dtype=torch.float32), p=2, dim=-1)

def resolve(tok):
    q = F.normalize(torch.as_tensor(v_ft.encode([tok]), dtype=torch.float32), p=2, dim=-1)[0]
    s = E @ q; top = torch.topk(s, 2)
    return ents[int(top.indices[0])], float(top.values[0] - top.values[1])
def uten(a, b):
    x = F.normalize(torch.as_tensor(u_ft.encode([a, b]), dtype=torch.float32), p=2, dim=-1)
    return _tension(float(x[0] @ x[1]))

COHORT = [
    ("pii_handling", "has_unencrypted_channel_pii", "forbidden",
     ["forbidden", "prohibited", "banned", "disallowed"], ["permitted", "allowed", "acceptable"]),
    ("pii_handling", "has_redaction", "required",
     ["required", "mandatory", "compulsory", "obligatory"], ["optional", "voluntary", "discretionary"]),
    ("account_closure", "has_identity_verification", "required",
     ["required", "mandatory", "needed"], ["optional", "waived", "not needed"]),
    ("account_closure", "has_fraud_notice_exception", "permitted",
     ["permitted", "allowed", "acceptable"], ["forbidden", "prohibited", "disallowed"]),
    ("disputes", "has_provisional_credit", "issued",
     ["issued", "granted", "provided"], ["denied", "withheld", "refused"]),
]

rows = []  # (want, routeB_verdict, fused_verdict)
detail = []
for h, r, s, pos, neg in COHORT:
    cap = store.cap_radius(r)
    for v, want in [(p, "supported") for p in pos] + [(n, "contradicted") for n in neg]:
        d = store.score_triple_emb(h, r, v_ft.encode([v])[0])
        near, margin = resolve(v)
        ut = uten(v, s)
        rb = "supported" if (d is not None and d <= cap) else "contradicted"
        # uncertainty-aware cascade
        if d is not None and d <= cap:
            fz = "supported"
        else:
            res_dec = margin >= RES_MARGIN
            u_low, u_high = ut <= TAU_U - U_BAND, ut >= TAU_U + U_BAND
            if res_dec and near != s:
                fz = "contradicted"
            elif res_dec and near == s:
                fz = "supported" if u_low else ("contradicted" if u_high else "uncertain")
            elif u_high:
                fz = "contradicted"
            elif u_low and near == s:
                fz = "supported"
            else:
                fz = "uncertain"
        rows.append((want, rb, fz))
        detail.append((v, want, rb, fz, near, margin, ut, d))

def boot_acc(mask_fn, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.array([i for i, r in enumerate(rows) if mask_fn(r)])
    if len(idx) == 0:
        return (float("nan"),) * 3
    corr = np.array([rows[i][1 if mask_fn is RB else 2] == rows[i][0] for i in idx])
    accs = [corr[rng.integers(0, len(idx), len(idx))].mean() for _ in range(B)]
    return corr.mean(), np.percentile(accs, 2.5), np.percentile(accs, 97.5)

RB = "routeB-marker"
n = len(rows)
rb_corr = np.mean([v == w for w, v, _ in rows])
# bootstrap on full cohort for route-B
rng = np.random.default_rng(0)
rb_boot = [np.mean([(rows[i][1] == rows[i][0]) for i in rng.integers(0, n, n)]) for _ in range(4000)]
conf = [(w, fz) for w, _, fz in rows if fz != "uncertain"]
fz_corr = np.mean([fz == w for w, fz in conf]) if conf else float("nan")
fz_idx = [i for i, (w, _, fz) in enumerate(rows) if fz != "uncertain"]
fz_boot = [np.mean([(rows[i][2] == rows[i][0]) for i in rng.choice(fz_idx, len(fz_idx))]) for _ in range(4000)] if fz_idx else [float("nan")]

print(f"store={SP}  tau_u={TAU_U}  n={n}\n")
print(f"{'value':12s} {'want':12s} {'routeB':12s} {'fused':12s}  near(Δ) / u / d_cap")
for v, w, rb, fz, near, mg, ut, d in detail:
    flag = "" if fz == w else ("  ~uncertain" if fz == "uncertain" else "  MISS")
    print(f"  {v:10s} {w:12s} {rb:12s} {fz:12s}  {near}(Δ{mg:.2f})/u{ut:.2f}/d{(d if d is not None else -1):.2f}{flag}")

print(f"\nroute-B-alone:  acc={rb_corr:.3f}  95% CI [{np.percentile(rb_boot,2.5):.3f}, {np.percentile(rb_boot,97.5):.3f}]  (n={n}, coverage=1.00)")
cov = len(conf) / n
print(f"fused+uncert:   acc={fz_corr:.3f}  95% CI [{np.percentile(fz_boot,2.5):.3f}, {np.percentile(fz_boot,97.5):.3f}]  (confident n={len(conf)}, coverage={cov:.2f})")
unc = [v for v, w, rb, fz, *_ in detail if fz == "uncertain"]
print(f"abstained (uncertain -> human review): {len(unc)}  {unc}")
