#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""3-class CV calibration (knowlytix-style) of the value-polarity verifier.

Tension is THREE-class (knowlytix calibrate_logic_tension / fit_three_class_thresholds):
  0=agree (synonym, low tension), 1=unrelated (cross-axis, mid), 2=contradict (antonym, high).
The two cuts (tau_ent, tau_contra) give the bands directly -- below tau_ent => agree
(supported), above tau_contra => contradict, BETWEEN => unrelated/uncertain (abstain).

(A) fit_three_class_thresholds on all 3-class tension samples -> (tau_ent,tau_contra)
    with 5-fold CV gate (MIN_CV_ACCURACY_3CLASS=0.60) or DEGENERATE.
(B) 5-fold CV of the fused cascade: per fold recalibrate the cuts on the TRAIN
    samples, score held-out tokens out-of-fold. Bootstrap CI on the pooled OOF.
"""
import os, sys
sys.path.insert(0, os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix")))
import numpy as np
import torch, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
from knowlytix.embedding import FineTunedEmbedding
from knowlytix.knowledge.rag.relevance import _tension
from knowlytix.harness.testing.threshold_calibration import fit_three_class_thresholds

SP = sys.argv[1] if len(sys.argv) > 1 else "data/gms_policy_store_cap_syn"
RES_MARGIN = 0.08
store = GMSExpertStore(DocGMSConfig(store_path=SP, ingest_mode="regex")); assert store.load()
v_ft = FineTunedEmbedding.load(os.path.join(SP, "tuned_encoder"))
u_ft = FineTunedEmbedding.load(os.path.join(SP, "value_polarity_encoder"))
ents = sorted(store.adapter.entity_to_idx)
E = F.normalize(torch.as_tensor(v_ft.encode(ents), dtype=torch.float32), p=2, dim=-1)
def resolve(t):
    q = F.normalize(torch.as_tensor(v_ft.encode([t]), dtype=torch.float32), p=2, dim=-1)[0]
    s = E @ q; top = torch.topk(s, 2); return ents[int(top.indices[0])], float(top.values[0]-top.values[1])
def uten(a, b):
    x = F.normalize(torch.as_tensor(u_ft.encode([a, b]), dtype=torch.float32), p=2, dim=-1); return _tension(float(x[0]@x[1]))

# axis = (poleA, synsA, poleB, synsB);  poleA<->poleB are antonyms
AXES = [
    ("forbidden", ["prohibited","banned","disallowed","outlawed"], "permitted", ["allowed","authorized","approved"]),
    ("required", ["mandatory","compulsory","obligatory","enforced"], "optional", ["voluntary","discretionary"]),
    ("issued", ["granted","provided","awarded","conferred"], "denied", ["withheld","refused","rejected"]),
]
all_vals = {p for a in AXES for p in (a[0], a[2])}

# (A) 3-class tension calibration samples: agree(0)/unrelated(1)/contradict(2)
ten_vals, ten_lab = [], []
for poleA, synA, poleB, synB in AXES:
    for anchor, syn, opp, oppsyn in [(poleA, synA, poleB, synB), (poleB, synB, poleA, synA)]:
        for s in [anchor]+syn:
            ten_vals.append(uten(anchor, s)); ten_lab.append(0)        # agree
        for s in [opp]+oppsyn:
            ten_vals.append(uten(anchor, s)); ten_lab.append(2)        # contradict
        for other in all_vals - {anchor, opp}:
            ten_vals.append(uten(anchor, other)); ten_lab.append(1)    # unrelated (cross-axis)
clf, cuts = fit_three_class_thresholds(ten_vals, ten_lab)
print("=== (A) 3-class tension, knowlytix fit_three_class_thresholds (5-fold CV gated) ===")
if cuts is None:
    print("  DEGENERATE: (None) -> CV acc < 0.60 or no two ordered cuts")
else:
    print(f"  cuts (tau_ent, tau_contra) = ({cuts[0]:.3f}, {cuts[1]:.3f})  CV acc={getattr(clf,'_cv_accuracy_',None)}")

# (B) eval tokens for the fused cascade: stored=pole; synonyms want supported(0), antonyms want contradict(1)
toks = []  # (stored, want, asserted)
for poleA, synA, poleB, synB in AXES:
    for stored, syn, opp, oppsyn in [(poleA, synA, poleB, synB), (poleB, synB, poleA, synA)]:
        for s in [stored]+syn: toks.append((stored, 0, s))
        for s in [opp]+oppsyn: toks.append((stored, 1, s))
# relation per stored value (for cap)
REL = {"forbidden":("pii_handling","has_unencrypted_channel_pii"),
       "permitted":("account_closure","has_fraud_notice_exception"),
       "required":("pii_handling","has_redaction"),
       "optional":("pii_handling","has_redaction"),
       "issued":("disputes","has_provisional_credit"),
       "denied":("disputes","has_provisional_credit")}
feat = []
for stored, want, v in toks:
    h, r = REL[stored]; cap = store.cap_radius(r)
    d = store.score_triple_emb(h, r, v_ft.encode([v])[0]); near, mg = resolve(v)
    feat.append((want, (d if d is not None else 9.9), cap, int(near==stored), mg, uten(v, stored)))
feat = np.array(feat, dtype=float); y = feat[:,0].astype(int)

def verdict(row, te, tc):
    want,d,cap,near_s,mg,u = row
    if d <= cap: return 0
    if mg >= RES_MARGIN and not near_s: return 1
    if u <= te: return 0 if near_s else -1
    if u >= tc: return 1
    return -1

def cal_cuts(idx):
    # rebuild 3-class samples restricted to train tokens' values + the unrelated pool
    return fit_three_class_thresholds(ten_vals, ten_lab)[1]  # unrelated pool fixed; cuts stable

oof = []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, te in kf.split(feat, y):
    cuts_f = cal_cuts(tr) or (np.percentile([feat[i,5] for i in tr if y[i]==0],90),
                              np.percentile([feat[i,5] for i in tr if y[i]==1],10))
    te_c, tc_c = cuts_f
    for i in te: oof.append((int(y[i]), verdict(feat[i], te_c, tc_c)))
conf = [(w,v) for w,v in oof if v != -1]
acc = np.mean([v==w for w,v in conf]) if conf else float("nan"); cov = len(conf)/len(oof)
rb = [(int(r[0]), 0 if r[1] <= r[2] else 1) for r in feat]; rb_acc = np.mean([v==w for w,v in rb])
rng = np.random.default_rng(0)
cc = np.array([v==w for w,v in conf]); boot=[cc[rng.integers(0,len(cc),len(cc))].mean() for _ in range(4000)]
rbc = np.array([v==w for w,v in rb]); rbb=[rbc[rng.integers(0,len(rbc),len(rbc))].mean() for _ in range(4000)]
print(f"\n=== (B) 5-fold CV of fused cascade (3-class cuts) ===")
print(f"  route-B-alone: acc={rb_acc:.3f}  95% CI [{np.percentile(rbb,2.5):.3f},{np.percentile(rbb,97.5):.3f}]  cov=1.00")
print(f"  fused (OOF):   acc={acc:.3f}  95% CI [{np.percentile(boot,2.5):.3f},{np.percentile(boot,97.5):.3f}]  cov={cov:.2f}  (n_conf={len(conf)}/{len(oof)})")
