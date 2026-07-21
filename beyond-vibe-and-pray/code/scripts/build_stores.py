"""Build one GMS store per FFIEC sub-topic for the ReasonLoop POC.

Per topic, the overview and examination-procedures markdown are concatenated
into one document, then a single persisted GMSExpertStore is built via the
knowlytix `ingest_document` pipeline (convert -> LLM parse -> train -> populate).

Extraction uses a local Qwen3-4B on the GB10 GPU as the parse LLM. GEODE's
`build_rag_store` was tried first but its regex extraction seed is built for
structured/tabular documents and returns zero triples on this regulatory prose;
the LLM extraction path is the appropriate reuse for prose sources.

    ~/cluster/spark-venv/bin/python scripts/build_stores.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.ingest import ingest_document
from knowlytix.knowledge.llm_backend import LocalTransformersBackend
from knowlytix.knowledge.store import GMSExpertStore

DATA = CODE / "data"
FFIEC = DATA / "ffiec"
STORES = DATA / "stores"

MODEL = "Qwen/Qwen3-4B-Instruct-2507"

TOPICS = {
    "cip": ["cip/01.md", "cip/01_ep.md"],
    "sar": ["sar/04.md", "sar/04_ep.md"],
}


def combine(topic: str, parts: list[str]) -> Path:
    out = FFIEC / topic / "combined.md"
    text = "\n\n".join((FFIEC / p).read_text(encoding="utf-8") for p in parts)
    out.write_text(text, encoding="utf-8")
    return out


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    STORES.mkdir(parents=True, exist_ok=True)

    print(f"loading extraction LLM: {MODEL} ...")
    t0 = time.time()
    llm = LocalTransformersBackend(MODEL, device="cuda")
    smoke = llm.call("You are a terse assistant.", "Reply with the single word OK.")
    print(f"  loaded in {time.time()-t0:.0f}s; smoke: {smoke[:40]!r}")

    for topic, parts in TOPICS.items():
        md = combine(topic, parts)
        store_path = STORES / topic
        print(f"\n=== building '{topic}'  ({md.stat().st_size} bytes)  ->  {store_path} ===")
        config = DocGMSConfig(store_path=str(store_path))
        store = GMSExpertStore(config, device=device)
        t0 = time.time()
        res = ingest_document(store, str(md), llm, config, device)
        store.save(str(store_path))
        prov = store_path / "provenance.json"
        print(
            f"  triples={res.new_triples}  entities={res.new_entities}  enm={res.new_enm}  "
            f"epochs={res.training_epochs}  first={res.is_first}  {time.time()-t0:.0f}s"
        )
        print(f"  stats: {store.stats()}")
        print(f"  provenance.json: {'yes' if prov.exists() else 'no (rebuildable from store markdown)'}")
    print("\nDONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
