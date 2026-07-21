"""Calibrate GMS plausibility and contradiction thresholds.

Sweeps theta (plausibility gate) and tau_contra (contradiction write-gate)
against a small labeled cohort and writes the chosen operating points to
``data/gms_banking_store/calibration.json``. The cohort is derived from
the structured tables in ``data/banking_policy.md`` so reviewers can audit
which (head, relation, tail) triples count as legal vs illegal.

Method: grid sweep over [0.1, 2.0] in 0.05 steps. At each threshold,
compute TP, TN, FP, FN, accuracy, false_allow_rate, false_deny_rate, and
Wilson 95% CI on accuracy. Pick the threshold that maximises accuracy
subject to ``false_allow_rate <= max_false_allow``. No DOE here -- the
sweep is one-dimensional and the cohort is small; the cohort doubles as a
regression fixture if the store is retrained.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import torch
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore


# -----------------------------------------------------------------------
# Cohorts
# -----------------------------------------------------------------------

# (head, relation, tail) -> label. label=1 means ALLOW, label=0 means DENY.

PLAUSIBILITY_COHORT: list[tuple[str, str, str, int]] = [
    # Legal in-DAG edges from data/banking_policy.md Workflow Authorizations
    # (the sequential capstone workflow classify->...->draft, plus escalation).
    ("start",            "has_enables", "classify",        1),
    ("classify",         "has_enables", "extract",         1),
    ("extract",          "has_enables", "search_policy",   1),
    ("search_policy",    "has_enables", "flag_regulatory", 1),
    ("flag_regulatory",  "has_enables", "draft_response",  1),
    ("flag_regulatory",  "has_enables", "escalate",        1),
    ("draft_response",   "has_enables", "escalate",        1),
    # Illegal: skip-step
    ("classify",         "has_enables", "search_policy",   0),
    ("extract",          "has_enables", "flag_regulatory", 0),
    ("search_policy",    "has_enables", "draft_response",  0),
    ("classify",         "has_enables", "draft_response",  0),
    ("start",            "has_enables", "draft_response",  0),
    ("start",            "has_enables", "escalate",        0),
    ("extract",          "has_enables", "draft_response",  0),
    # Illegal: reverse direction
    ("draft_response",   "has_enables", "classify",        0),
    ("escalate",         "has_enables", "classify",        0),
    ("flag_regulatory",  "has_enables", "search_policy",   0),
]

CONTRADICTION_COHORT: list[tuple[str, str, str, int]] = [
    # True facts (ENM-backed): allow
    ("representative",      "has_max_reversal", "35.0",     1),
    ("supervisor",          "has_max_reversal", "100.0",    1),
    ("manager",             "has_max_reversal", "500.0",    1),
    ("compliance_officer",  "has_max_reversal", "99999.0",  1),
    ("overdraft",           "has_fee_amount",   "35.0",     1),
    ("late_payment",        "has_fee_amount",   "25.0",     1),
    ("wire_domestic",       "has_fee_amount",   "30.0",     1),
    ("udaap",               "has_threshold",    "500.0",    1),
    ("reg_e",               "has_threshold",    "50.0",     1),
    # Contradicting tails on functional relations: deny
    ("representative",      "has_max_reversal", "100.0",    0),
    ("representative",      "has_max_reversal", "500.0",    0),
    ("supervisor",          "has_max_reversal", "35.0",     0),
    ("supervisor",          "has_max_reversal", "500.0",    0),
    ("manager",             "has_max_reversal", "35.0",     0),
    ("overdraft",           "has_fee_amount",   "25.0",     0),
    ("overdraft",           "has_fee_amount",   "100.0",    0),
    ("late_payment",        "has_fee_amount",   "35.0",     0),
    ("udaap",               "has_threshold",    "50.0",     0),
    ("reg_e",               "has_threshold",    "500.0",    0),
]


# -----------------------------------------------------------------------
# Wilson interval + sweep
# -----------------------------------------------------------------------

@dataclass
class CalibrationResult:
    relation_set: list[str]
    threshold: float
    accuracy: float
    ci_lo: float
    ci_hi: float
    false_allow_rate: float
    false_deny_rate: float
    cohort_n: int
    cohort_n_allow: int
    cohort_n_deny: int


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + (z * z) / n
    centre = phat + (z * z) / (2 * n)
    half = z * math.sqrt(phat * (1 - phat) / n + (z * z) / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def sweep_threshold(
    store: GMSExpertStore,
    cohort: list[tuple[str, str, str, int]],
    *,
    max_false_allow: float = 0.05,
    grid_lo: float = 0.1,
    grid_hi: float = 2.0,
    grid_step: float = 0.05,
) -> CalibrationResult:
    rows: list[tuple[float, int]] = []  # (score, label)
    for h, r, t, label in cohort:
        s = store.score_triple(h, r, t)
        if s is None:
            print(f"  WARN: score None for ({h!r}, {r!r}, {t!r})", file=sys.stderr)
            continue
        rows.append((float(s), label))

    n = len(rows)
    n_allow = sum(1 for _, lbl in rows if lbl == 1)
    n_deny = n - n_allow
    if n_allow == 0 or n_deny == 0:
        raise RuntimeError("cohort needs both ALLOW and DENY entries to sweep")

    grid = [grid_lo + i * grid_step
            for i in range(int(round((grid_hi - grid_lo) / grid_step)) + 1)]

    best: tuple[float, float, int] | None = None  # (acc, theta, k_correct)
    for theta in grid:
        tp = sum(1 for s, lbl in rows if lbl == 1 and s <= theta)  # legal allowed
        tn = sum(1 for s, lbl in rows if lbl == 0 and s > theta)   # illegal denied
        fn = n_allow - tp
        fp = n_deny - tn
        false_allow = fp / n_deny if n_deny else 0.0
        acc = (tp + tn) / n
        if false_allow > max_false_allow:
            continue
        score = (acc, -abs(theta - 1.0))  # secondary: prefer centered theta
        if best is None or score > (best[0], -abs(best[1] - 1.0)):
            best = (acc, theta, tp + tn)

    if best is None:
        # No theta met the false-allow ceiling; fall back to max-accuracy.
        for theta in grid:
            tp = sum(1 for s, lbl in rows if lbl == 1 and s <= theta)
            tn = sum(1 for s, lbl in rows if lbl == 0 and s > theta)
            acc = (tp + tn) / n
            if best is None or acc > best[0]:
                best = (acc, theta, tp + tn)
        assert best is not None

    acc, theta, k_correct = best
    ci_lo, ci_hi = wilson_ci(k_correct, n)
    tp = sum(1 for s, lbl in rows if lbl == 1 and s <= theta)
    tn = sum(1 for s, lbl in rows if lbl == 0 and s > theta)
    return CalibrationResult(
        relation_set=sorted({r for _, r, _, _ in cohort}),
        threshold=theta,
        accuracy=acc,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        false_allow_rate=(n_deny - tn) / n_deny,
        false_deny_rate=(n_allow - tp) / n_allow,
        cohort_n=n,
        cohort_n_allow=n_allow,
        cohort_n_deny=n_deny,
    )


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    store_path = root / "data" / "gms_banking_store"
    out_path = store_path / "calibration.json"
    if not store_path.exists():
        print(f"FAIL: missing {store_path}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GMSExpertStore(DocGMSConfig(store_path=str(store_path)), device=device)
    if not store.load():
        print(f"FAIL: could not load store at {store_path}", file=sys.stderr)
        return 1

    print("Sweeping plausibility threshold (theta) ...")
    theta_result = sweep_threshold(store, PLAUSIBILITY_COHORT, max_false_allow=0.05)
    print(f"  theta={theta_result.threshold:.2f}  acc={theta_result.accuracy:.3f}  "
          f"[CI {theta_result.ci_lo:.3f}, {theta_result.ci_hi:.3f}]  "
          f"false_allow={theta_result.false_allow_rate:.3f}  "
          f"false_deny={theta_result.false_deny_rate:.3f}  "
          f"n={theta_result.cohort_n}")

    print("Sweeping contradiction threshold (tau_contra) ...")
    tau_result = sweep_threshold(store, CONTRADICTION_COHORT, max_false_allow=0.05)
    print(f"  tau_contra={tau_result.threshold:.2f}  acc={tau_result.accuracy:.3f}  "
          f"[CI {tau_result.ci_lo:.3f}, {tau_result.ci_hi:.3f}]  "
          f"false_allow={tau_result.false_allow_rate:.3f}  "
          f"false_deny={tau_result.false_deny_rate:.3f}  "
          f"n={tau_result.cohort_n}")

    # Merge into any existing calibration so sections this script does not
    # manage (e.g. groundedness, written elsewhere) are preserved.
    payload = json.loads(out_path.read_text()) if out_path.exists() else {}
    payload.update({
        "store_path": str(store_path.relative_to(root)),
        "calibrated_at": date.today().isoformat(),
        "method": "grid_sweep_v1",
        "max_false_allow": 0.05,
        "plausibility_gate": asdict(theta_result),
        "contradiction_gate": asdict(tau_result),
    })
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
