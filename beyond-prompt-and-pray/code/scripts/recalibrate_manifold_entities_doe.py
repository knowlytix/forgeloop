"""Recalibrate the ManifoldEntityExtractor flag thresholds against the DoE set.

The cases.json calibration (20 rows, almost no hard negatives) picked tau=1.30,
which the guard A/B then showed is far too loose: recall soared but ~32% of the
60 benign ``none`` rows falsely escalated. This refits udaap/reg_x tau against the
negative-rich DoE set (60 ``none`` rows as negatives) and --- to keep the chosen
threshold honest rather than circular --- reports it under 2-fold cross-validation
before persisting a tau fit on all in-scope rows.

Operates at the entity-path *firing* level (does udaap/reg_x fire on the message),
which is what the threshold controls; the full-guard escalation effect is then
re-measured by scripts/eval_manifold_entity_flags.py against the persisted tau.

    python scripts/recalibrate_manifold_entities_doe.py            # CV report + persist
    python scripts/recalibrate_manifold_entities_doe.py --dry-run  # report only
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

from agentlab.capstone.manifold_entities import ManifoldEntityExtractor
from agentlab.capstone.manifold_evidence import _DEFAULT_ARTIFACT

_CSV = Path("data/capstone_doe_results.csv")
_OUT = _DEFAULT_ARTIFACT / "entity_calibration.json"
_FACTOR_TO_ENTITY = {"UDAAP": "udaap", "Reg_X": "reg_x"}
_IN_SCOPE = {"UDAAP", "Reg_X", "none"}
_GRID = [round(0.05 * k, 2) for k in range(10, 41)]   # 0.50 .. 2.00


def _best_tau(rows: list[tuple[float, bool]]) -> tuple[float, float]:
    """tau maximizing F1 over the grid for one entity; (tau, f1)."""
    pos = sum(1 for _, g in rows if g)
    best = (1.30, -1.0)
    if pos == 0:
        return best
    for tau in _GRID:
        tp = sum(1 for d, g in rows if g and d <= tau)
        fp = sum(1 for d, g in rows if not g and d <= tau)
        fn = pos - tp
        f1 = 0.0 if tp == 0 else tp / (tp + 0.5 * (fp + fn))
        if f1 > best[1]:
            best = (tau, f1)
    return best


def _confusion(rows: list[tuple[float, bool]], tau: float) -> tuple[int, int, int, int]:
    tp = sum(1 for d, g in rows if g and d <= tau)
    fp = sum(1 for d, g in rows if not g and d <= tau)
    fn = sum(1 for d, g in rows if g and d > tau)
    tn = sum(1 for d, g in rows if not g and d > tau)
    return tp, fp, fn, tn


def main() -> int:
    dry = "--dry-run" in sys.argv
    rows = [r for r in csv.DictReader(_CSV.open()) if r["regulatory"] in _IN_SCOPE]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ext = ManifoldEntityExtractor.load(default_tau=1.30)

    # Per-entity: list of (min_geodesic_distance, is_gold) aligned to rows.
    dist: dict[str, list[tuple[float, bool]]] = {"udaap": [], "reg_x": []}
    fold: list[int] = []
    for i, r in enumerate(rows):
        ed = ext.entity_distances(r["message"])
        gold_ent = _FACTOR_TO_ENTITY.get(r["regulatory"])
        for e in ("udaap", "reg_x"):
            dist[e].append((ed[e], gold_ent == e))
        fold.append(i % 2)                              # deterministic 2-fold split

    print(f"in-scope rows: {len(rows)} "
          f"(UDAAP={sum(r['regulatory']=='UDAAP' for r in rows)}, "
          f"Reg_X={sum(r['regulatory']=='Reg_X' for r in rows)}, "
          f"none={sum(r['regulatory']=='none' for r in rows)})\n")

    fitted: dict[str, float] = {}
    for e in ("udaap", "reg_x"):
        d = dist[e]
        # 2-fold CV: fit on one fold, score the held-out one; pool the held-out confusion.
        held = [0, 0, 0, 0]
        for f in (0, 1):
            train = [d[i] for i in range(len(d)) if fold[i] != f]
            test = [d[i] for i in range(len(d)) if fold[i] == f]
            tau, _ = _best_tau(train)
            c = _confusion(test, tau)
            held = [held[j] + c[j] for j in range(4)]
        tp, fp, fn, tn = held
        rec = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        # Final tau fit on all in-scope rows (what we persist).
        full_tau, full_f1 = _best_tau(d)
        fitted[e] = full_tau
        print(f"{e}:  CV held-out  recall={rec:.0%} ({tp}/{tp+fn})  "
              f"precision={prec:.0%} ({tp}/{tp+fp})  none-FP={fp}/{tn+fp}")
        print(f"      persist tau={full_tau:.2f} (was 1.30; full-set F1={full_f1:.2f})\n")

    payload = {
        "entity_thresholds": fitted,
        "source": str(_CSV),
        "method": ("F1-max over geodesic grid; gold = factors.regulatory; "
                   "60 none rows as hard negatives; tau reported under 2-fold CV, "
                   "persisted value fit on all in-scope rows"),
        "n_cases": len(rows),
    }
    if dry:
        print("--dry-run: not writing.\n" + json.dumps(payload, indent=2))
        return 0
    _OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
