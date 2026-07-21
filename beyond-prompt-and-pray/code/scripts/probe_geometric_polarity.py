#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Route B: can a polarity reversal be caught GEOMETRICALLY via the operator/cap?

Once an answer is decomposed to a claim triple (h, r, asserted_value), a polarity
reversal is just a WRONG categorical tail. The deciding question: does the GMS
operator geometry place the CONSISTENT tail (the stored value) inside the relation
cap and the FLIPPED tail (opposite polarity) outside it -- the same admissibility
check that catches numeric tampering, now on a categorical polarity tail?

For each polarity fact (h, r, t) we report:
  - is t a scorable tail entity? (route B needs the polarity tokens in the graph)
  - score_triple(h, r, t)            consistent  -- want <= cap_radius(r)
  - score_triple(h, r, flipped)      reversed    -- want >  cap_radius(r)
  - cap_radius(r)                    the calibrated admissibility boundary

"SEPARABLE" if every consistent score is within cap and every flipped score is
outside it. Run on spark-ef84 (offline):

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    KNOWLYTIX_SRC=$HOME/jupyterlab/GMS-knowlytix PYTHONPATH=$HOME/jupyterlab/GMS-knowlytix \
    python scripts/probe_geometric_polarity.py --store-path data/gms_policy_store_cap
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)

# polarity facts in the bank policy store + the opposite-polarity token.
POLARITY = [
    ("pii_handling", "has_unencrypted_channel_pii", "forbidden", "permitted"),
    ("pii_handling", "has_redaction", "required", "optional"),
    ("account_closure", "has_identity_verification", "required", "optional"),
    ("account_closure", "has_fraud_notice_exception", "permitted", "forbidden"),
    ("disputes", "has_provisional_credit", "issued", "denied"),
]


def main() -> int:
    from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", default="data/gms_policy_store_cap")
    args = ap.parse_args()

    cfg = DocGMSConfig(store_path=args.store_path, ingest_mode="regex")
    store = GMSExpertStore(cfg)
    if not store.load():
        print(f"failed to load store at {args.store_path}")
        return 1
    g = store.doc_graph

    ents = {e.lower() for e in (getattr(g, "entities", None) or
                                {h for h, _r, _t in g.triples} |
                                {t for _h, _r, t in g.triples})}
    print(f"store: {len(g.triples)} triples; polarity tokens present as entities?")
    for tok in ("forbidden", "permitted", "required", "optional", "issued", "denied"):
        print(f"  {tok:12s} {'yes' if tok in ents else 'NO'}")

    def cap(r):
        try:
            return store.cap_radius(r)
        except Exception as e:  # noqa: BLE001
            return f"ERR:{type(e).__name__}"

    def score(h, r, t):
        try:
            s = store.score_triple(h, r, t)
            return None if s is None else float(s)
        except Exception as e:  # noqa: BLE001
            return f"ERR:{type(e).__name__}"

    print(f"\n{'relation':40s} {'cap':>8} {'consistent':>11} {'flipped':>9}  verdict")
    cons_ok = flip_ok = total = 0
    for h, r, t, flip in POLARITY:
        cr = cap(r)
        sc = score(h, r, t)
        sf = score(h, r, flip)
        total += 1
        if isinstance(cr, float) and isinstance(sc, float) and isinstance(sf, float):
            ci = sc <= cr
            fo = sf > cr
            cons_ok += ci
            flip_ok += fo
            verdict = ("OK" if (ci and fo) else
                       f"{'cons>cap ' if not ci else ''}{'flip<=cap' if not fo else ''}")
            crs, scs, sfs = f"{cr:.3f}", f"{sc:.3f}", f"{sf:.3f}"
        else:
            verdict = "unscorable"
            crs, scs, sfs = str(cr), str(sc), str(sf)
        print(f"  {r:38s} {crs:>8} {scs:>11} {sfs:>9}  ({t}->{flip}) {verdict}")

    print(f"\nconsistent within cap: {cons_ok}/{total} | "
          f"flipped outside cap: {flip_ok}/{total}")
    print("=> route B viable" if cons_ok == total == flip_ok else
          "=> route B does NOT cleanly separate as-is")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
