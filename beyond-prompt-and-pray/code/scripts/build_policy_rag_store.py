"""Build a GMS-backed entity index for Graph-RAG policy search.

Ingests data/bank_policies.md (Chapter 32's per-policy attribute compendium)
into a GMS store at data/gms_policy_store/. The store is separate from
data/gms_banking_store/, which serves the Chapter 16 substrate role
(plausibility gate, ENM, tension). The two have different schemas and
different consumers.

Run after editing bank_policies.md or whenever the policy library grows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from knowlytix.harness.governance.memory import ingest_document
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    doc_path = root / "data" / "bank_policies.md"
    store_path = root / "data" / "gms_policy_store"

    if not doc_path.exists():
        print(f"FAIL: missing {doc_path}", file=sys.stderr)
        return 1

    config = DocGMSConfig(
        ingest_mode="regex",
        store_path=str(store_path),
    )
    config.train.epochs = 200
    config.train.batch_size = 256
    config.train.lr = 5e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    store = GMSExpertStore(config, device=device)
    result = ingest_document(
        store, str(doc_path), llm=None, config=config, device=device,
    )
    print(f"ingest result: {result}")
    store.save()
    print(f"saved to {store_path}")
    print(f"entities: {len(store.adapter.entity_to_idx)}")
    print()
    print("--- entity sample (lower-cased, first 40) ---")
    for e in sorted(store.adapter.entity_to_idx.keys())[:40]:
        print(f"  {e!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
