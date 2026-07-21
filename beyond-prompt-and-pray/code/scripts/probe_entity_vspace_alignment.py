#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Are the GMS entity embeddings the v-encoder (text) embeddings?

If yes (entities inserted/frozen from the v-encoder), the operator/cap geometry is
defined over v-space, so route B generalizes to ANY asserted tail: embed its text
with the v-encoder and score it against the operator image, even when the token is
not a trained entity. If no (learned-from-scratch / drifted), a v-encoder embedding
is in a different frame and cannot be fed into the geometry.

Test: for several trained entities, cosine(GMS model entity vector, v-encoder
embedding of the entity name). High & uniform => text-aligned (B generalizes).

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    KNOWLYTIX_SRC=$HOME/jupyterlab/GMS-knowlytix PYTHONPATH=$HOME/jupyterlab/GMS-knowlytix \
    python scripts/probe_entity_vspace_alignment.py --store-path data/gms_policy_store_cap
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)

PROBE_ENTS = ["forbidden", "permitted", "required", "issued",
              "checking_account", "overdraft", "pii_handling", "disputes"]


def main() -> int:
    import torch
    import torch.nn.functional as F
    from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
    from knowlytix.embedding import FineTunedEmbedding

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", default="data/gms_policy_store_cap")
    args = ap.parse_args()
    sp = Path(args.store_path)

    store = GMSExpertStore(DocGMSConfig(store_path=str(sp), ingest_mode="regex"))
    if not store.load():
        print("load failed"); return 1
    model, adapter = store.model, store.adapter

    # GMS model entity v-embedding table (warm-started from the v-encoder).
    emb = model.dual_emb.v_embed.weight.detach().float().cpu()
    print(f"model v_embed table: {tuple(emb.shape)}  (d_v={model.dual_emb.d_v})")

    # The actual warm-start vectors (256-d, name-keyed) the model was seeded from.
    warm = torch.load(str(sp / "v_emb.pt"), map_location="cpu")
    print(f"v_emb.pt: {len(warm)} warm-start vectors, dim "
          f"{tuple(next(iter(warm.values())).shape)}")

    def vvec(name):
        v = warm.get(name)
        return None if v is None else F.normalize(
            torch.as_tensor(v, dtype=torch.float32), p=2, dim=-1)

    print(f"\n{'entity':20s} {'in store?':>9} {'dim(model/v)':>14} {'cos(model,v)':>13}")
    cur = store
    e2i = adapter.entity_to_idx
    for e in PROBE_ENTS:
        ce = cur._canon_entity(e) if hasattr(cur, "_canon_entity") else e
        idx = e2i.get(ce)
        if idx is None or idx >= emb.shape[0]:
            print(f"  {e:18s} {'no':>9}")
            continue
        mv = F.normalize(emb[idx], p=2, dim=-1)
        vv = vvec(ce) if vvec(ce) is not None else vvec(e)
        if vv is None:
            print(f"  {e:18s} {'yes':>9} {'-':>14} {'not in v_emb.pt':>13}")
            continue
        dims = f"{mv.shape[0]}/{vv.shape[0]}"
        if mv.shape[0] != vv.shape[0]:
            print(f"  {e:18s} {'yes':>9} {dims:>14} {'DIM MISMATCH':>13}")
            continue
        cos = float(mv @ vv)
        print(f"  {e:18s} {'yes':>9} {dims:>14} {cos:>13.3f}")

    print("\nhigh & uniform cos => entity geometry is v-space (route B generalizes);"
          "\nlow / mismatched   => learned/drifted frame (cannot reuse v-embeddings).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
