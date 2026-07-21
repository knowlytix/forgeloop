#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Verify-layer route B: AnswerVerifier._check returns geometric verdicts for
categorical/polarity claims (synonym-tolerant, out-of-vocab) and leaves the numeric
path exact. Run on the consistent store (out_dim==d_v)."""
import os, sys
sys.path.insert(0, os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix")))
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
from knowlytix.embedding import FineTunedEmbedding
from knowlytix.knowledge.rag.verify import AnswerVerifier
from knowlytix.knowledge.rag.query_triples import QueryTriple

SP = sys.argv[1] if len(sys.argv) > 1 else "data/gms_policy_store_cap_v2"
store = GMSExpertStore(DocGMSConfig(store_path=SP, ingest_mode="regex"))
assert store.load()
ft = FineTunedEmbedding.load(os.path.join(SP, "tuned_encoder"))
ver = AnswerVerifier(store, None, v_encoder=ft.encode)

# (head, relation, tail, expected, note)
CASES = [
    ("pii_handling", "has_redaction", "required", "supported", "consistent (categorical)"),
    ("pii_handling", "has_redaction", "mandatory", "supported", "SYNONYM (string-match would fail)"),
    ("pii_handling", "has_redaction", "optional", "contradicted", "reversal, out-of-vocab"),
    ("pii_handling", "has_unencrypted_channel_pii", "forbidden", "supported", "consistent"),
    ("pii_handling", "has_unencrypted_channel_pii", "permitted", "contradicted", "reversal"),
    ("pii_handling", "has_unencrypted_channel_pii", "prohibited", "supported", "SYNONYM of forbidden"),
    ("account_closure", "has_identity_verification", "optional", "contradicted", "reversal, out-of-vocab"),
    ("overdraft", "has_fee_amount", "35", "supported", "NUMERIC -> exact path"),
    ("overdraft", "has_fee_amount", "999", "contradicted", "NUMERIC reversal -> exact path"),
]

print(f"{'head/relation':45s} {'tail':12s} {'verdict':13s} {'exp':13s} {'':3s} note")
hits = 0
for h, r, t, exp, note in CASES:
    vd = ver._check(QueryTriple(h, r, t))
    good = vd.status == exp
    hits += good
    print(f"  {h+'.'+r:43s} {t:12s} {vd.status:13s} {exp:13s} {'OK ' if good else 'XX '} {note}")
    print(f"      -> {vd.detail}")
print(f"\n{hits}/{len(CASES)} verdicts as expected")
