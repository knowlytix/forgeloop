#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Pinpoint the production train_gms hang: time warm-start vs the epoch loop.

The full build stalls in store_from_triples before printing epoch 1. This
isolates that step: load the corrected triples from the existing cap store, then
run train_gms with timing around _maybe_init_embeddings (the EmbeddingConfig
Mode B warm-start) and the epoch loop, at a small epoch count so it returns fast.
"""
import os, sys, time
sys.path.insert(0, os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix")))

import torch
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
from knowlytix.core.config import CapLossConfig

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/gms_policy_store_cap"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30

dev = torch.device("cuda")
# Existing store -> corrected triples + the warm-start vectors path.
st = GMSExpertStore(DocGMSConfig(store_path=SRC, ingest_mode="regex"))
assert st.load(), "load failed"
triples = list(st.doc_graph.triples)
print(f"loaded {len(triples)} triples from {SRC}")

cfg = DocGMSConfig(ingest_mode="regex", store_path="/tmp/time_train_store")
cfg.train.epochs = EPOCHS
cfg.train.device = str(dev)
cfg.loss_mode = "cap"
cfg.cap = CapLossConfig()
vpath = os.path.join(SRC, "v_emb.pt")
if os.path.exists(vpath):
    cfg.embedding.v_vectors_path = vpath
    print(f"warm-start v_vectors_path = {vpath}")

# Time the warm-start step alone by patching it.
import knowlytix.core.train_finstructbench as T
_orig = T._maybe_init_embeddings
def _timed(model, adapter, embedding):
    t = time.time()
    _orig(model, adapter, embedding)
    print(f"  [_maybe_init_embeddings warm-start] {time.time()-t:.1f}s", flush=True)
T._maybe_init_embeddings = _timed

from knowlytix.knowledge.geode.rag import store_from_triples
print(f"calling store_from_triples (epochs={EPOCHS}) ...", flush=True)
t0 = time.time()
store = store_from_triples("data/banking_policy_full.md", triples, cfg, device=dev)
print(f"TOTAL store_from_triples: {time.time()-t0:.1f}s", flush=True)
