#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Regenerate a store's provenance.json to point each fact at its source SENTENCE.

The KG (triples) is unchanged; only provenance is re-derived against the realistic
prose corpus, with prefer_prose so a fact binds to the policy statement that
asserts it (readable) rather than the table cell it was extracted from. Also
refreshes the store's documents/combined.md so query-time reads the same corpus.

Pure text alignment (no GPU / no model retrain).

    python scripts/regen_prose_provenance.py \
        --store data/gms_policy_store_cap --corpus data/banking_policy_full.md
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import Counter

_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)


def _triples(store: str):
    raw = json.load(open(os.path.join(store, "triples.json")))
    out = []
    for t in raw:
        if isinstance(t, dict):
            out.append((t["head"], t["relation"], t["tail"]))
        else:
            out.append(tuple(t[:3]))
    return out


def main() -> int:
    from knowlytix.knowledge.geode.provenance import ProvenanceLedger, is_consistent

    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="data/gms_policy_store_cap")
    ap.add_argument("--corpus", default="data/banking_policy_full.md")
    ap.add_argument("--write", action="store_true", help="persist provenance.json + combined.md")
    args = ap.parse_args()

    triples = _triples(args.store)
    text = open(args.corpus, encoding="utf-8").read()
    led = ProvenanceLedger.from_text(text, "<report>", prefer_prose=True)
    cov = led.coverage(triples)
    records = led.to_records(triples)

    by_method = Counter(r["method"] for r in records)
    inconsistent = [(r["head"], r["relation"], r["tail"], r["method"])
                    for r, p in zip(records, cov["provenances"].values())
                    if not is_consistent(p)]
    print(f"triples: {len(triples)}   methods: {dict(by_method)}")
    print(f"unaligned: {cov['unaligned']}  {cov['unaligned_triples'][:8]}")
    print(f"inconsistent (value not in span): {len(inconsistent)}  {inconsistent[:8]}")

    print("\n-- sample prose-sentence provenance --")
    shown = 0
    for r in records:
        if r["method"] == "prose_sentence" and shown < 8:
            print(f"  ({r['head']}, {r['relation']}, {r['tail']})")
            print(f"     -> {r['raw']!r}")
            shown += 1
    print("\n-- the SSN/PII stance fact --")
    for r in records:
        if r["head"] == "pii_handling" and "unencrypted" in r["relation"]:
            print(f"  ({r['head']}, {r['relation']}, {r['tail']}) [{r['method']}]")
            print(f"     -> {r['raw']!r}")

    if args.write:
        with open(os.path.join(args.store, "provenance.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        os.makedirs(os.path.join(args.store, "documents"), exist_ok=True)
        with open(os.path.join(args.store, "documents", "combined.md"), "w",
                  encoding="utf-8") as f:
            f.write(text)
        print(f"\nWROTE provenance.json ({len(records)} recs) + documents/combined.md")
    else:
        print("\n(dry run; pass --write to persist)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
