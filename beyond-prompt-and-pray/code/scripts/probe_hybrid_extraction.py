#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Extraction-quality probe: hybrid ingest (regex + Qwen3-4B) of the policy corpus.

Dumps every extracted triple + ENM entry grouped by head, then checks coverage
against the target fact set (the values every policy fact must carry). Run this
BEFORE building the store, so extraction accuracy is verified first.

Run on spark-ef84:
    KNOWLYTIX_SRC=$HOME/GMS-knowlytix PYTHONPATH=$HOME/GMS-knowlytix \
      python scripts/probe_hybrid_extraction.py data/banking_policy_full.md
"""
from __future__ import annotations
import os, sys
_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)

DOC = sys.argv[1] if len(sys.argv) > 1 else "data/banking_policy_full.md"
MODE = sys.argv[2] if len(sys.argv) > 2 else "hybrid"

# Target values each policy fact must carry (head-concept -> list of expected
# tail values, as strings). Coverage = does an extracted triple for a plausibly
# matching head carry this value? Relation surface may differ (prose extraction
# names its own relations), so we check value presence per head concept.
TARGETS = {
    "overdraft": ["35", "per_occurrence", "1", "regulation_e"],
    "nsf": ["35", "per_event"],
    "late_payment": ["25", "per_event"],
    "wire_domestic": ["30", "per_transaction"],
    "wire_international": ["45", "per_transaction"],
    "stop_payment": ["30", "per_request"],
    "paper_statement": ["3", "per_month"],
    "disputes": ["60", "10", "issued", "regulation_e", "regulation_z"],
    "fee_reversal": ["35", "12", "0", "manager"],
    "account_closure": ["30", "required", "permitted"],
    "pii_handling": ["24", "forbidden", "required"],
    "regulatory_escalation": ["1", "500", "12_usc_5531", "regulation_x"],
    "loan_servicing": ["25", "regulation_z"],
    "representative": ["35"], "supervisor": ["100"], "manager": ["500"],
    "compliance_officer": ["99999"],
    "reg_e": ["unauthorized_electronic_transfer", "dispute_unit", "50"],
    "reg_x": ["mortgage_servicing_issue", "compliance", "0"],
    "reg_z": ["credit_card_billing_error", "dispute_unit", "0"],
    "udaap": ["unfair_or_abusive_fee", "compliance", "500"],
}


def _norm(s: str) -> str:
    return "".join(c for c in str(s).lower() if c.isalnum())


def _val_in(target: str, tail: str) -> bool:
    t, x = _norm(target), _norm(tail)
    if t and (t in x or x in t):
        return True
    try:
        return abs(float(str(target).replace(",", "")) -
                   float(str(tail).replace(",", "").rstrip("."))) < 1e-6
    except ValueError:
        return False


def main() -> int:
    from knowlytix.benchmark.ingest import ingest_markdown
    from knowlytix.knowledge.geode import QWEN_4B
    from knowlytix.knowledge.llm_backend import LocalTransformersBackend
    from knowlytix.knowledge.ingest import _llm_backend_to_callable

    cb = None
    if MODE in ("hybrid", "llm_only"):
        llm = LocalTransformersBackend(QWEN_4B, device="cuda")
        cb = _llm_backend_to_callable(llm)
    print(f"ingesting {DOC} mode={MODE} llm={QWEN_4B if cb else None} ...", flush=True)
    g = ingest_markdown(DOC, mode=MODE, llm_callable=cb)

    triples = list(g.triples)
    from collections import defaultdict
    by = defaultdict(list)
    for h, r, t in triples:
        by[h].append((r, t))
    print(f"\n=== extracted: {len(triples)} triples, "
          f"{len(g.enm)} ENM, {len(by)} heads ===")
    for h in sorted(by):
        print(f"\n{h}:")
        for r, t in sorted(by[h]):
            print(f"    {r} -> {t}")

    print("\n\n=== COVERAGE vs target facts (value present for a matching head?) ===")
    hit = miss = 0
    for concept, vals in TARGETS.items():
        # candidate heads whose name overlaps the concept
        cand = [h for h in by if _norm(concept) in _norm(h) or _norm(h) in _norm(concept)]
        pool = [(r, t) for h in cand for r, t in by[h]]
        for v in vals:
            ok = any(_val_in(v, t) for _, t in pool)
            hit += ok; miss += (not ok)
            if not ok:
                print(f"  MISS  {concept:24s} value={v!r}  (heads tried: {cand})")
    tot = hit + miss
    print(f"\ncoverage: {hit}/{tot} target values present "
          f"({100*hit/max(1,tot):.0f}%);  {miss} missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
