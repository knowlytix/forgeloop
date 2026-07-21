"""Build + train the GMS regulatory-guard store for flag_regulatory.

Ingests data/regulatory_guard.md (flag -> evidence-entity graph, plus
customer-vocabulary aliases) into a GMS store at data/gms_regulatory_store/
and trains the geometric embeddings so score_triple / link_predict are
meaningful. This store is the deterministic guard behind the flag_regulatory
tool: Qwen proposes regulations, this graph verifies and corrects them.

Separate from data/gms_policy_store/ (Graph-RAG retrieval) and
data/gms_banking_store/ (Chapter-16 substrate); different schema, different
consumer. Run after editing regulatory_guard.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from knowlytix.harness.governance.memory import ingest_document
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    doc_path = root / "data" / "regulatory_guard.md"
    store_path = root / "data" / "gms_regulatory_store"

    if not doc_path.exists():
        print(f"FAIL: missing {doc_path}", file=sys.stderr)
        return 1

    config = DocGMSConfig(ingest_mode="regex", store_path=str(store_path))
    config.train.epochs = 600
    config.train.batch_size = 256
    config.train.lr = 5e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    store = GMSExpertStore(config, device=device)
    result = ingest_document(store, str(doc_path), llm=None, config=config, device=device)
    print(f"ingest result: {result}")
    store.save()
    print(f"saved to {store_path}")

    triples = store.triples
    rels = sorted({r for _, r, _ in triples})
    print(f"entities: {len(store.adapter.entity_to_idx)}  triples: {len(triples)}")
    print(f"relations: {rels}")
    print("\n--- has_evidence triples ---")
    for h, r, t in triples:
        if r == "has_evidence":
            print(f"  {h} -> {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
