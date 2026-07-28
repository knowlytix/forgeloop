#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the demo GMS store the test-suite fixture loads.

`tests/conftest.py`'s `gms_store` fixture reads a trained store from
`data/gms_demo_store`, built from the committed corpus
`data/model_risk_assessment.md`. No script built it, so the fixture always
skipped and the GMS-backed adapter tests (gates, memory, judge — 14 of them)
never ran even with a working licensed knowlytix installed.

Same shape as retrain_gms_banking.py: regex ingest + train, no LLM, CPU is fine.

    python scripts/build_demo_store.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from knowlytix.harness.governance.memory import ingest_document
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    doc_path = root / "data" / "model_risk_assessment.md"
    store_path = root / "data" / "gms_demo_store"

    if not doc_path.exists():
        print(f"FAIL: missing corpus {doc_path}", file=sys.stderr)
        return 1

    config = DocGMSConfig(ingest_mode="regex", store_path=str(store_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\ningesting {doc_path} ...")

    store = GMSExpertStore(config, device=device)
    result = ingest_document(store, str(doc_path), llm=None, config=config, device=device)
    print(f"ingest result: {result}")
    print(f"triples: {len(store.adapter.heads) if store.adapter else 0}")
    # ExactNumericalMemory is not Sized in current knowlytix — count its keys.
    print(f"enm:     {len(list(store.enm.keys())) if store.enm else 0}")

    store.save()
    print(f"saved to {store_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
