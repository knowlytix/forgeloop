#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test of the knowlytix ValuePolarityChecker through AnswerVerifier.

Calibrates PolarityCuts (3-class CV), persists them, builds the checker, wires it
into AnswerVerifier, and runs _check over a synonym/antonym cohort.
"""
import os, sys
sys.path.insert(0, os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix")))
import torch, torch.nn.functional as F
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
from knowlytix.embedding import FineTunedEmbedding
from knowlytix.knowledge.rag.value_polarity import PolarityCuts, ValuePolarityChecker
from knowlytix.knowledge.rag.verify import AnswerVerifier
from knowlytix.knowledge.rag.query_triples import QueryTriple
from knowlytix.knowledge.rag.relevance import _tension

SP = sys.argv[1] if len(sys.argv) > 1 else "data/gms_policy_store_cap_syn"
store = GMSExpertStore(DocGMSConfig(store_path=SP, ingest_mode="regex")); assert store.load()
v_ft = FineTunedEmbedding.load(os.path.join(SP, "tuned_encoder"))
u_ft = FineTunedEmbedding.load(os.path.join(SP, "value_polarity_encoder"))

def uten(a, b):
    x = F.normalize(torch.as_tensor(u_ft.encode([a, b]), dtype=torch.float32), p=2, dim=-1)
    return _tension(float(x[0] @ x[1]))

AXES = [
    ("forbidden", ["prohibited","banned","disallowed","outlawed"], "permitted", ["allowed","authorized","approved"]),
    ("required", ["mandatory","compulsory","obligatory","enforced"], "optional", ["voluntary","discretionary"]),
    ("issued", ["granted","provided","awarded","conferred"], "denied", ["withheld","refused","rejected"]),
]
all_vals = {p for a in AXES for p in (a[0], a[2])}
# 3-class calibration cohort
tv, tl = [], []
for pA, sA, pB, sB in AXES:
    for anc, syn, opp, osy in [(pA,sA,pB,sB),(pB,sB,pA,sA)]:
        for s in [anc]+syn: tv.append(uten(anc,s)); tl.append(0)
        for s in [opp]+osy: tv.append(uten(anc,s)); tl.append(2)
        for o in all_vals-{anc,opp}: tv.append(uten(anc,o)); tl.append(1)
cuts = PolarityCuts.calibrate(tv, tl)
print(f"calibrated cuts: {cuts}")
if cuts: cuts.save(os.path.join(SP, "value_polarity_calibration.json"))

checker = ValuePolarityChecker(store, v_ft.encode, u_ft.encode, cuts)
ver = AnswerVerifier(store, None, v_encoder=v_ft.encode, value_checker=checker)

REL = {"forbidden":("pii_handling","has_unencrypted_channel_pii"),
       "permitted":("account_closure","has_fraud_notice_exception"),
       "required":("pii_handling","has_redaction"),
       "issued":("disputes","has_provisional_credit")}
CASES = [
    ("forbidden","prohibited","supported"), ("forbidden","banned","supported"),
    ("forbidden","permitted","contradicted"), ("forbidden","allowed","contradicted"),
    ("required","mandatory","supported"), ("required","compulsory","supported"),
    ("required","optional","contradicted"), ("required","voluntary","contradicted"),
    ("permitted","allowed","supported"), ("permitted","forbidden","contradicted"),
    ("issued","granted","supported"), ("issued","denied","contradicted"),
]
print(f"\n{'stored':10s} {'asserted':12s} {'verdict':13s} {'want':13s}")
hit = 0
for stored, val, want in CASES:
    h, r = REL[stored]
    vd = ver._check(QueryTriple(h, r, val))
    ok = (vd.status == want) or (vd.status == "uncertain")  # uncertain = safe defer
    hit += vd.status == want
    mark = "OK" if vd.status == want else ("~defer" if vd.status == "uncertain" else "MISS")
    print(f"  {stored:10s} {val:12s} {vd.status:13s} {want:13s} {mark}  ({vd.detail})")
print(f"\nexact verdicts: {hit}/{len(CASES)}  (uncertain counted as safe deferral)")
