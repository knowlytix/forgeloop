# SPDX-License-Identifier: Apache-2.0
"""Binder Bake-Off, Arm B: build the finetuned SLM binder via the library.

Thin caller of ``knowlytix.knowledge.rag.compiler.build_query_compiler`` -- datagen
(DoE-varied, held-out-level + held-out-fact aware) -> LoRA SFT Qwen3-4B -> adapter
saved into ``store/query_compiler/`` and auto-wired by ``RagConfig.from_store_dir``.

The manifest (store/query_compiler/compiler_meta.json) carries n_train, the held-out
counts, the factor set and, with evaluate=True, accept-cohort metrics. Arm B is later
measured on BOTH surfaces: ``CompiledQueryParser.compile_raw`` (the SLM's raw learned
emission -> mis_bind surface) and ``.extract`` (store-walked -> safe abstention).

Run on GPU:  python scripts/nl2triple_experiment/build_arm_b.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
from forgeloop import data_path  # noqa: E402  (resolves the installed book data)

import torch  # noqa: E402

from knowlytix.core.config import GeometryConfig  # noqa: E402
from knowlytix.knowledge.config import DocGMSConfig  # noqa: E402
from knowlytix.knowledge.llm_backend import LocalTransformersBackend  # noqa: E402
from knowlytix.knowledge.rag.compiler import QWEN_4B, build_query_compiler  # noqa: E402
from knowlytix.knowledge.store import GMSExpertStore  # noqa: E402

STORE = os.environ.get("GMS_STORE", str(data_path("gms_annual_report_store")))


def load_store(store_path, dev):
    g = json.load(open(os.path.join(store_path, "model_dims.json")))["geometry"]
    cfg = DocGMSConfig(store_path=store_path, geometry=GeometryConfig(
        d_v=g["d_v"], d_u=g["d_u"], m=g["m"], d=g["d"]))
    store = GMSExpertStore(cfg, device=dev)
    assert store.load(), f"store not found at {store_path}"
    return store


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    store = load_store(STORE, dev)
    llm = LocalTransformersBackend(QWEN_4B, device=dev)
    print(f"[arm-b] store={STORE} dev={dev} triples={len(store.triples)}", flush=True)
    manifest = build_query_compiler(
        STORE, llm, store=store,
        variants_per_base=12,        # DoE surface breadth per base question
        max_multi_hop=150,
        negatives_per_head=1,
        heldout_fact_frac=0.15,      # reserve novel (head,relation) facts for novel_entity_score
        device=dev, evaluate=True, seed=42,
    )
    print("[arm-b] manifest:")
    print(json.dumps(manifest, indent=2))
    print(f"[arm-b] adapter dir: {manifest.get('compiler_dir')}")


if __name__ == "__main__":
    main()
