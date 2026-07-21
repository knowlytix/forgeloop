#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Stage B (Chunk-and-Pray Ch6-8) validation for the banking-policy store:
held-out paraphrase binding + parse-and-bind extraction.

The store's ``has_alias`` edges hold the customer vocabulary the parse-bind
encoder was tuned on. This probe binds *held-out* customer phrasings (NOT in the
alias table) to the canonical policy entity through the real
``TripleBinder(mode="embedding")`` path, so a hit is generalization rather than
memorization. It reports binding accuracy for the baseline MiniLM encoder and
for the tuned encoder side by side, then confirms ``RagPipeline.extract`` binds
a handful of natural customer messages to policy triples (geometric parse, no
LLM) -- the artifact ``extract_facts`` / ``search_policy`` consume.

Run on spark-ef84:
    PYTHONPATH=$HOME/jupyterlab/GMS-knowlytix \
      python scripts/probe_policy_binding.py --store-path data/gms_policy_store_cap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.store import GMSExpertStore
from knowlytix.knowledge.rag import TripleBinder
from knowlytix.knowledge.rag.query_triples import QueryTriple

# (held-out paraphrase, expected canonical policy entity). Every surface form is
# absent from the store's has_alias table, so a correct bind is generalization.
PROBES = [
    ("I overdrew my checking account",      "overdraft"),
    ("my account went into the red",         "overdraft"),
    ("I didn't make this charge",            "disputes"),
    ("contest a transaction on my card",     "disputes"),
    ("please credit the fee back to me",     "fee_reversal"),
    ("reverse this charge as a courtesy",    "fee_reversal"),
    ("shut down my account",                 "account_closure"),
    ("cancel my checking account",           "account_closure"),
    ("my personal data was exposed",         "pii_handling"),
    ("someone leaked my private information","pii_handling"),
    ("this fee is unfair and deceptive",     "regulatory_escalation"),
    ("escalate this to compliance",          "regulatory_escalation"),
    ("my monthly loan bill",                 "loan_servicing"),
    ("the payment on my car loan",           "loan_servicing"),
]

# Natural customer messages for the parse-and-bind extraction check.
MESSAGES = [
    "I want to dispute a charge I never authorized.",
    "Why was I hit with an overdraft fee of $35?",
    "Can you waive the fee on my account as a goodwill gesture?",
    "I need to close my checking account.",
    "My social security number was sent over an unencrypted email.",
]


def _load_store(store_path: str, device) -> GMSExpertStore:
    cfg = DocGMSConfig(store_path=store_path, ingest_mode="regex", loss_mode="cap")
    store = GMSExpertStore(cfg, device)
    if not store.load():
        raise SystemExit(f"FAIL: no store at {store_path}")
    return store


def _alias_map(store) -> dict:
    """alias-node -> canonical policy entity, read off the has_alias edges."""
    m = {}
    for h, r, t in store.triples:
        if r == "has_alias":
            m[t] = h          # tail is the customer alias, head is canonical
    return m


def _resolve(head, amap):
    """Resolve an alias node to its canonical policy entity (one hop)."""
    return amap.get(head, head)


def _binding_accuracy(store, encoder, label, amap, bind_threshold=0.5, bind_margin=0.05):
    binder = TripleBinder(store, mode="embedding", encoder=encoder,
                          bind_threshold=bind_threshold, bind_margin=bind_margin)
    hits, rows = 0, []
    for surf, exp in PROBES:
        b = binder.bind(QueryTriple(surf, "has_policy", "?"))
        resolved = _resolve(b.head, amap) if b.head else None
        ok = (resolved == exp)
        hits += ok
        rows.append((surf, exp, b.head, resolved, ok))
    print(f"\n=== {label}: {hits}/{len(PROBES)} = {hits/len(PROBES):.2f} (alias-resolved) ===")
    for surf, exp, raw, resolved, ok in rows:
        via = f" via {raw!r}" if raw != resolved else ""
        print(f"   [{'OK ' if ok else 'XX '}] {surf!r:42s} -> {str(resolved)!r:22s}{via}  (exp {exp!r})")
    return hits / len(PROBES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", default="data/gms_policy_store_cap")
    ap.add_argument("--encoder-dir", default=None,
                    help="tuned-encoder dir (default <store>/tuned_encoder)")
    ap.add_argument("--bind-threshold", type=float, default=0.5)
    ap.add_argument("--bind-margin", type=float, default=0.05)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    store = _load_store(args.store_path, device)
    amap = _alias_map(store)

    bt, bm = args.bind_threshold, args.bind_margin
    print(f"(bind_threshold={bt} bind_margin={bm})")
    # baseline MiniLM (encoder=None) vs the alias-tuned encoder saved with the store
    base = _binding_accuracy(store, None, "baseline MiniLM", amap, bt, bm)

    from knowlytix.embedding import FineTunedEmbedding
    enc_dir = args.encoder_dir or str(Path(args.store_path) / "tuned_encoder")
    ft = FineTunedEmbedding.load(enc_dir)
    tuned = _binding_accuracy(store, ft.encode, f"tuned ({Path(enc_dir).name})", amap, bt, bm)
    print(f"\nbaseline={base:.2f}  tuned={tuned:.2f}  delta={tuned-base:+.2f}")

    # parse-and-bind extraction: the grounded fact/query extract_facts hands to
    # search_policy. Geometric parse uses the tuned encoder; the Qwen3-4B backend
    # satisfies RagConfig and serves as the extraction fallback.
    from knowlytix.knowledge.geode import QWEN_4B
    from knowlytix.knowledge.llm_backend import LocalTransformersBackend
    from knowlytix.knowledge.rag import RagConfig, RagPipeline
    llm = LocalTransformersBackend(QWEN_4B, device=str(device))
    cfg = RagConfig(
        llm=llm, binding="embedding", encoder=ft.encode,
        query_parse_mode="geometric", relevance_gate=False,
        dense_fallback=False,
    )
    pipe = RagPipeline.from_store(store, cfg)
    print("\n=== RagPipeline.extract parse-and-bind (alias-resolved policy) ===")
    bound_ct = 0
    for msg in MESSAGES:
        ex = pipe.extract(msg)
        facts = list(getattr(ex, "bound_facts", []) or [])
        policies = sorted({_resolve(h, amap) for h, r, t in facts})
        bound_ct += bool(facts)
        print(f"   {msg!r}")
        print(f"      is_bound={ex.is_bound}  policies={policies}  bound_facts={facts}")
    print(f"\nmessages with >=1 bound fact: {bound_ct}/{len(MESSAGES)}")


if __name__ == "__main__":
    main()
