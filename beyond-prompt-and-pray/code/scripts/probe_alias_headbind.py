#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Cheap test: would keeping has_alias (as entity-alias NAMES for head binding)
rescue the low-scoring positives? Loads the store, reads the corpus has_alias
table, injects those alias edges into the in-memory doc_graph, rebuilds the
GeometricQueryParser, and recomputes head_candidates scores WITH vs WITHOUT
aliases for the calibration cohort. No LLM, no rebuild.

Run on spark-ef84 (offline):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH=$HOME/jupyterlab/GMS-knowlytix \
      python scripts/probe_alias_headbind.py --store-path data/gms_policy_store_cap \
        --corpus data/banking_policy_full.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)

POSITIVES = [
    ("What is the overdraft fee?", "overdraft"),
    ("How long do I have to dispute a charge?", "disputes"),
    ("What's the deadline to file a transaction dispute?", "disputes"),
    ("I want to dispute a charge I never authorized.", "disputes"),
    ("Can my social security number be sent over email?", "pii_handling"),
    ("My social security number was emailed unencrypted.", "pii_handling"),
    ("Is redaction required for personal data?", "pii_handling"),
    ("Escalate this unfair fee to compliance.", "regulatory_escalation"),
    ("What is the UDAAP harm threshold?", "regulatory_escalation"),
    ("How much can a representative reverse without approval?", "fee_reversal"),
    ("How much notice before the bank closes my account?", "account_closure"),
]
OOV = ["What is the capital of France?", "What is the weather forecast for tomorrow?",
       "Who is the CEO of the bank?", "Tell me about cryptocurrency investing.",
       "What is management's outlook for next year?",
       "How do I reset my online banking password?"]


def _corpus_aliases(md_path):
    """Read the Policy Aliases table -> {head: [alias, ...]}."""
    import re
    out = {}
    in_tbl = False
    for ln in Path(md_path).read_text().splitlines():
        if ln.strip().lower().startswith("## policy aliases"):
            in_tbl = True
            continue
        if in_tbl and ln.startswith("## "):
            break
        m = re.match(r"\|\s*([a-z0-9_]+)\s*\|\s*([^|]+?)\s*\|", ln)
        if in_tbl and m and m.group(1) != "policy_id":
            out.setdefault(m.group(1), []).append(m.group(2).strip())
    return out


def main() -> int:
    import torch
    from knowlytix.embedding import FineTunedEmbedding
    from knowlytix.knowledge.config import DocGMSConfig
    from knowlytix.knowledge.store import GMSExpertStore
    from knowlytix.knowledge.rag.query_triples import GeometricQueryParser

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", default="data/gms_policy_store_cap")
    ap.add_argument("--corpus", default="data/banking_policy_full.md")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = DocGMSConfig(store_path=str(args.store_path), ingest_mode="regex", loss_mode="cap")
    store = GMSExpertStore(cfg, dev)
    store.load()
    v_enc = FineTunedEmbedding.load(str(Path(args.store_path) / "tuned_encoder"))

    base = GeometricQueryParser(store, encoder=v_enc.encode)

    # inject has_alias edges into the doc_graph, rebuild a second parser
    aliases = _corpus_aliases(args.corpus)
    n_added = 0
    for h, al in aliases.items():
        for a in al:
            try:
                store.doc_graph.add_triple(h, "has_alias", a)
                n_added += 1
            except Exception:  # noqa: BLE001
                pass
    withal = GeometricQueryParser(store, encoder=v_enc.encode)
    print(f"injected {n_added} has_alias edges; "
          f"entities with aliases now: {len(withal._aliases)}\n")

    def score(p, q):
        c = p.head_candidates(q, top_k=1)
        return (c[0][0], c[0][1]) if c else ("-", 0.0)

    print("=== POSITIVES: head bind WITHOUT vs WITH aliases ===")
    for q, exp in POSITIVES:
        h0, s0 = score(base, q)
        h1, s1 = score(withal, q)
        f0 = "OK" if h0 == exp else "XX"
        f1 = "OK" if h1 == exp else "XX"
        print(f"  exp={exp:22s} | base [{f0}] {h0:18s} {s0:.3f}  ->  "
              f"alias [{f1}] {h1:18s} {s1:.3f}  | {q!r}")
    print("\n=== OOV (must stay low / bind nothing useful) ===")
    for q in OOV:
        h0, s0 = score(base, q)
        h1, s1 = score(withal, q)
        print(f"  base {h0:18s} {s0:.3f}  ->  alias {h1:18s} {s1:.3f}  | {q!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
