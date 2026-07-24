#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Route B end-to-end on the consistent store: polarity reversal via operator/cap.

(1) Consistency: tuned_encoder out_dim == model d_v, and v_embed == the warm-start
    vectors (cos ~1, frozen). (2) Route B: for each polarity fact, score the
    CONSISTENT tail and the FLIPPED tail against the relation cap. In-vocabulary
    tails go through score_triple (index); OUT-OF-VOCAB tails (optional/denied) go
    through score_triple_emb, embedding the token with the tuned encoder -- the
    fix for the gap that made 3/5 unscorable before. Route B works if every
    consistent tail is within cap and every flipped tail is outside it.
"""
import os, sys
sys.path.insert(0, os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/GMS-knowlytix")))
import torch, torch.nn.functional as F
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
from knowlytix.embedding import FineTunedEmbedding

SP = sys.argv[1] if len(sys.argv) > 1 else "data/gms_policy_store_cap_v2"

POLARITY = [
    ("pii_handling", "has_unencrypted_channel_pii", "forbidden", "permitted"),
    ("pii_handling", "has_redaction", "required", "optional"),
    ("account_closure", "has_identity_verification", "required", "optional"),
    ("account_closure", "has_fraud_notice_exception", "permitted", "forbidden"),
    ("disputes", "has_provisional_credit", "issued", "denied"),
]

store = GMSExpertStore(DocGMSConfig(store_path=SP, ingest_mode="regex"))
assert store.load(), "load failed"
ft = FineTunedEmbedding.load(os.path.join(SP, "tuned_encoder"))
de = store.model.dual_emb
d_v = de.d_v
ents = set(store.adapter.entity_to_idx)

# (1) consistency
enc = ft.encode(["forbidden"])
enc_dim = torch.as_tensor(enc).reshape(len(["forbidden"]), -1).shape[1]
warm = torch.load(os.path.join(SP, "v_emb.pt"), map_location="cpu")
vemb = de.v_embed.weight.detach().float().cpu()
cos = []
for e in ("forbidden", "permitted", "required", "issued"):
    i = store.adapter.entity_to_idx.get(store._canon_entity(e))
    if i is not None and e in warm:
        a = F.normalize(vemb[i], dim=-1); b = F.normalize(torch.as_tensor(warm[e], dtype=torch.float32), dim=-1)
        cos.append(float(a @ b))
print(f"[consistency] d_v={d_v}  tuned encode dim={enc_dim}  "
      f"(match={enc_dim == d_v})  v_embed-vs-warm cos={[round(c,3) for c in cos]}")

def emb(token):
    v = ft.encode([token])
    return torch.as_tensor(v, dtype=torch.float32).reshape(-1)

def score(h, r, tail):
    if tail in ents:
        return store.score_triple(h, r, tail), "idx"
    return store.score_triple_emb(h, r, emb(tail)), "emb"

print(f"\n{'relation':38s} {'cap':>6} {'consistent':>18} {'flipped':>18}  verdict")
ok = 0
for h, r, t, flip in POLARITY:
    cap = store.cap_radius(r)
    sc, mc = score(h, r, t)
    sf, mf = score(h, r, flip)
    good = sc is not None and sf is not None and sc <= cap < sf
    ok += good
    print(f"  {r:36s} {cap:>6.3f} {f'{sc:.3f}({mc})' if sc is not None else 'None':>18} "
          f"{f'{sf:.3f}({mf})' if sf is not None else 'None':>18}  "
          f"{'OK' if good else 'FAIL'}  ({t}->{flip})")
print(f"\nroute B separates {ok}/{len(POLARITY)} polarity facts "
      f"(consistent within cap < flipped outside)")
