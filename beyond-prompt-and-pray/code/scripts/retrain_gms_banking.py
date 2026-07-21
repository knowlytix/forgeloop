"""Retrain the banking GMS store with schema-declared functional relations.

Reads data/banking_policy.md (now containing `## Schema Declarations`
bullets), re-runs the docgms ingest pipeline, and overwrites the trained
store at data/gms_banking_store/. After this script, tension_energy on
heads sharing a functional relation should rise from ~1.2 to ~1.7-2.0.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
from knowlytix.harness.governance.memory import ingest_document
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    doc_path = root / "data" / "banking_policy.md"
    store_path = root / "data" / "gms_banking_store"
    backup_path = root / "data" / "gms_banking_store.pre_schema"

    if not doc_path.exists():
        print(f"FAIL: missing {doc_path}", file=sys.stderr)
        return 1

    if store_path.exists() and not backup_path.exists():
        print(f"backing up existing store to {backup_path}")
        shutil.copytree(store_path, backup_path)

    config = DocGMSConfig(
        ingest_mode="regex",
        store_path=str(store_path),
    )
    config.train.epochs = 800
    config.train.batch_size = 256
    config.train.lr = 5e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"loss_mode: {config.loss_mode}")
    print(f"lambda_tension: {config.loss.lambda_tension}")
    print(f"epochs: {config.train.epochs}")

    store = GMSExpertStore(config, device=device)
    print(f"ingesting {doc_path} ...")
    result = ingest_document(
        store, str(doc_path), llm=None, config=config, device=device,
    )
    print(f"ingest result: {result}")
    print(f"triples: {len(store.adapter.heads) if store.adapter else 0}")
    print(f"enm:     {len(store.enm) if store.enm else 0}")
    store.save()
    print(f"saved to {store_path}")

    # Quick probe to verify tension supervision activated
    print("\n--- tension probes (post-retrain) ---")
    for a, b in [
        ("representative", "supervisor"),
        ("supervisor", "manager"),
        ("35.0", "100.0"),
        ("100.0", "500.0"),
        ("overdraft", "wire_international"),
    ]:
        te = store.tension_energy(a, b)
        print(f"  tension({a!r}, {b!r}) = {te}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
