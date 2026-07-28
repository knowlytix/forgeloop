"""Calibrate the regulatory-guard evidence threshold.

The flag_regulatory guard accepts a (flag, has_evidence, entity) link when
``store.score_triple`` is at or below a threshold theta (lower score = more
plausible). This script sweeps theta against a labeled cohort built from the
store itself: every real has_evidence triple is a positive (ALLOW); each
evidence entity paired with a wrong flag is a negative (DENY). It picks the
theta that maximises accuracy subject to a false-allow ceiling and writes
``data/gms_regulatory_store/calibration.json``.

Method mirrors scripts/calibrate_gms_thresholds.py: one-dimensional grid sweep
with a Wilson 95% CI on accuracy. The cohort doubles as a regression fixture if
the store is retrained.

Run after scripts/build_regulatory_guard_store.py.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path

import torch
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore

FLAGS = ["udaap", "reg_e", "reg_z", "reg_x", "fcra"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + (z * z) / n
    centre = phat + (z * z) / (2 * n)
    half = z * math.sqrt(phat * (1 - phat) / n + (z * z) / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    store_path = root / "data" / "gms_regulatory_store"
    out_path = store_path / "calibration.json"
    if not store_path.exists():
        print(f"FAIL: missing {store_path}; run build_regulatory_guard_store.py", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GMSExpertStore(DocGMSConfig(store_path=str(store_path)), device=device)
    if not store.load():
        print(f"FAIL: could not load store at {store_path}", file=sys.stderr)
        return 1

    # Cohort from the store: positives = real has_evidence triples; negatives =
    # the same evidence entity paired with every wrong flag.
    evidence_of: dict[str, str] = {}
    for h, r, t in store.triples:
        if r == "has_evidence":
            evidence_of[t] = h
    cohort: list[tuple[str, str, str, int]] = []
    for entity, true_flag in evidence_of.items():
        cohort.append((true_flag, "has_evidence", entity, 1))
        for flag in FLAGS:
            if flag != true_flag:
                cohort.append((flag, "has_evidence", entity, 0))

    rows: list[tuple[float, int]] = []
    for h, r, t, label in cohort:
        s = store.score_triple(h, r, t)
        if s is None:
            print(f"  WARN: score None for ({h!r},{r!r},{t!r})", file=sys.stderr)
            continue
        rows.append((float(s), label))

    n = len(rows)
    n_allow = sum(1 for _, lbl in rows if lbl == 1)
    n_deny = n - n_allow
    grid = [0.4 + i * 0.01 for i in range(int(round((2.0 - 0.4) / 0.01)) + 1)]

    best = None  # (acc, theta, k_correct)
    for theta in grid:
        tp = sum(1 for s, lbl in rows if lbl == 1 and s <= theta)
        tn = sum(1 for s, lbl in rows if lbl == 0 and s > theta)
        fp = n_deny - tn
        false_allow = fp / n_deny if n_deny else 0.0
        acc = (tp + tn) / n
        if false_allow > 0.05:
            continue
        if best is None or acc > best[0]:
            best = (acc, theta, tp + tn)
    if best is None:  # fall back to max accuracy if ceiling unreachable
        for theta in grid:
            tp = sum(1 for s, lbl in rows if lbl == 1 and s <= theta)
            tn = sum(1 for s, lbl in rows if lbl == 0 and s > theta)
            acc = (tp + tn) / n
            if best is None or acc > best[0]:
                best = (acc, theta, tp + tn)

    acc, theta, k = best
    tp = sum(1 for s, lbl in rows if lbl == 1 and s <= theta)
    tn = sum(1 for s, lbl in rows if lbl == 0 and s > theta)
    ci_lo, ci_hi = wilson_ci(k, n)
    print(f"theta={theta:.3f}  acc={acc:.3f}  [CI {ci_lo:.3f},{ci_hi:.3f}]  "
          f"false_allow={(n_deny-tn)/n_deny:.3f}  false_deny={(n_allow-tp)/n_allow:.3f}  "
          f"n={n} (allow={n_allow}, deny={n_deny})")

    payload = {
        "store_path": str(store_path.relative_to(root)),
        "calibrated_at": date.today().isoformat(),
        "method": "grid_sweep_v1",
        "relation": "has_evidence",
        "max_false_allow": 0.05,
        "evidence_threshold": theta,
        "accuracy": acc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "cohort_n": n,
        "cohort_n_allow": n_allow,
        "cohort_n_deny": n_deny,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
