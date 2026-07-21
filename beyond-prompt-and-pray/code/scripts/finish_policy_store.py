#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Complete the consistent policy store in a FRESH process.

The monolithic build wedges at store_from_triples (an in-process CUDA/state issue
after the GEODE loop + embed/contradiction SFT cycles); the identical call runs
in seconds in a clean process. This stage reuses the encoders the build already
saved (tuned_encoder/, contradiction_encoder/, v_emb.pt) and produces only the
missing production GMS: run the geometry-only GEODE loop for corrected triples,
then train + save the production store, warm-started from the build's v_emb.pt so
v_embed == the tuned-encoder frame (frozen).
"""
import os, sys, time
KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix"))
sys.path.insert(0, KNOW)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
from knowlytix.core.config import CapLossConfig
from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.geode.loop import GeodeLoop
from knowlytix.knowledge.geode.rag import store_from_triples
from build_geode_rag_store import _make_compat_trainer, _NOISE_RELATIONS

DOC = sys.argv[1] if len(sys.argv) > 1 else "data/banking_policy_full.md"
STORE = sys.argv[2] if len(sys.argv) > 2 else "data/gms_policy_store_cap_v2"
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 800
dev = torch.device("cuda")

config = DocGMSConfig(ingest_mode="regex", store_path=STORE)
config.train.epochs = EPOCHS
config.train.device = str(dev)
config.loss_mode = "cap"
config.cap = CapLossConfig()
vpath = os.path.join(STORE, "v_emb.pt")
assert os.path.exists(vpath), f"missing warm-start vectors {vpath} (run the build's embed-SFT first)"
config.embedding.v_vectors_path = vpath
print(f"warm-start from {vpath}", flush=True)

t = time.time()
loop = GeodeLoop(_make_compat_trainer(dev, epochs=EPOCHS), device=dev, llm=None, max_iters=8)
loop_res = loop.run(DOC)
kept = [(h, r, t2) for h, r, t2 in loop_res.triples if r not in _NOISE_RELATIONS]
print(f"GEODE loop: converged={loop_res.converged}, {len(kept)} kept triples "
      f"({time.time()-t:.1f}s)", flush=True)

t = time.time()
store = store_from_triples(DOC, kept, config, device=dev, drop_relations=_NOISE_RELATIONS)
print(f"store_from_triples done in {time.time()-t:.1f}s; "
      f"{store.adapter.num_entities} entities, "
      f"{len(store.doc_graph.triples)} triples", flush=True)
print("STORE COMPLETE:", STORE, flush=True)
