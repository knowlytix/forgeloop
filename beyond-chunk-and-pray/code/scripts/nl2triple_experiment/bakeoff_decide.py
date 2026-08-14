# SPDX-License-Identifier: Apache-2.0
"""Binder Bake-Off decision rule (spec Section 6, gate G4) -- thin caller.

Reads the crossover report written by run_bakeoff.py and applies the shared G4
decision rule from knowlytix.knowledge.rag.bakeoff.decide: admissibility against the
(un-ratified placeholder) ceilings, the preference for the binder that generalizes,
recusal when none clears the ceilings, and the audit record (I4). All logic lives in
the library; this script only loads, calls decide(), prints, and persists.
"""
from __future__ import annotations

import json
import os

from forgeloop import data_path   # resolves the installed book data
from knowlytix.knowledge.rag.bakeoff import DEFAULT_CEILINGS, decide

REPORT = os.environ.get("BAKEOFF_REPORT", str(data_path("enrichment", "bakeoff_ABCD.json")))
OUT = str(data_path("enrichment", "bakeoff_decision.json"))
# The runtime has no clock in the library; stamp the audit from the caller.
STAMP = os.environ.get("BAKEOFF_STAMP", "2026-07-30")


def main():
    report = json.load(open(REPORT))
    out = decide(report, ceilings=DEFAULT_CEILINGS, ratified=False, timestamp=STAMP)

    print(f"[decide] ceilings (un-ratified placeholders): {out['ceilings']}")
    print(f"[decide] overall accuracy leader: "
          f"{out['audit']['overall_accuracy_leader']}\n")
    for name, a in out["arms"].items():
        m, ck = a["metrics"], a["checks"]
        flags = " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in ck.items())
        print(f"  {name:12s} acc={m['accuracy']} holdout={m['holdout_accuracy']} "
              f"mis_bind={m['mis_bind_rate']} gap={m['synthetic_vs_real_gap']} "
              f"cov={m['conformal_coverage']} novel={m['novel_entity_score']} "
              f"patch={m['patch_cost']}({m['patch_mechanism']}) | "
              f"admissible={a['admissible']} [{flags}]")
    print(f"\n[decide] recommended: {out['recommended']}  (recused={out['recused']})")
    print(f"[decide] rationale: {out['rationale']}")
    print("\n[decide] per-regime leaders (real-language crossover):")
    for f, levels in out["by_regime"].items():
        leaders = ", ".join(f"{lv}:{arm}" for lv, arm in levels.items())
        print(f"    {f:16s} {leaders}")

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[decide] wrote {OUT}")


if __name__ == "__main__":
    main()
