#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Focused routing check on the REAL artifact path (RagPipeline.extract) for the
de-collided regulation entities and the PII case -- the queries the isolated
binding probe does not cover. Geometric parse, no LLM judge.

Run on spark-ef84 (offline):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$HOME/jupyterlab/GMS-knowlytix \
      python scripts/probe_policy_extract_routing.py --store-path data/gms_policy_store_cap
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.store import GMSExpertStore
from knowlytix.embedding import FineTunedEmbedding
from knowlytix.knowledge.geode import QWEN_4B
from knowlytix.knowledge.llm_backend import LocalTransformersBackend
from knowlytix.knowledge.rag import RagConfig, RagPipeline

# (message, expected head substring) -- reg de-collision + pii + a few controls.
CASES = [
    ("When does Regulation E apply to my dispute?",        "reg_e"),
    ("What is the threshold for Regulation Z disclosures?", "reg_z"),
    ("Does Regulation X cover my mortgage servicing?",      "reg_x"),
    ("My social security number was emailed unencrypted.",  "pii_handling"),
    ("Someone leaked my personal information.",             "pii_handling"),
    ("Escalate this unfair fee to compliance.",             "regulatory_escalation"),
    ("I want to dispute a charge I never authorized.",      "disputes"),
    ("Why was I charged a $35 overdraft fee?",              "overdraft"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", default="data/gms_policy_store_cap")
    ap.add_argument("--encoder-dir", default=None,
                    help="tuned-encoder dir (default <store>/tuned_encoder)")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = DocGMSConfig(store_path=args.store_path, ingest_mode="regex", loss_mode="cap")
    store = GMSExpertStore(cfg, dev)
    if not store.load():
        raise SystemExit(f"FAIL: no store at {args.store_path}")
    enc_dir = args.encoder_dir or str(Path(args.store_path) / "tuned_encoder")
    ft = FineTunedEmbedding.load(enc_dir)
    llm = LocalTransformersBackend(QWEN_4B, device=str(dev))
    rc = RagConfig(llm=llm, binding="embedding", encoder=ft.encode,
                   query_parse_mode="geometric", relevance_gate=False,
                   dense_fallback=False)
    pipe = RagPipeline.from_store(store, rc)

    hits = 0
    print("=== extract routing (real pipeline, geometric parse) ===")
    for msg, exp in CASES:
        ex = pipe.extract(msg)
        facts = list(getattr(ex, "bound_facts", []) or [])
        heads = sorted({h for h, r, t in facts})
        ok = any(exp in h for h in heads)
        hits += ok
        print(f"   [{'OK ' if ok else 'XX '}] {msg!r}")
        print(f"        heads={heads}  facts={facts}  (exp ~{exp!r})")
    print(f"\nrouting: {hits}/{len(CASES)}")


if __name__ == "__main__":
    main()
