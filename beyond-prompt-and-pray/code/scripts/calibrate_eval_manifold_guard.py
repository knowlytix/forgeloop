"""Calibrate the manifold scorer's per-flag thresholds on a held-out DEV split,
then run the end-to-end escalation before/after on the TEST split.

Honest protocol (no eval-set threshold peeking):
  * stratified DEV/TEST split of the in-scope DoE rows (UDAAP, Reg_X, none)
  * per-flag tau calibrated on DEV by Youden's J (positives=class rows, negatives=none)
  * TEST escalation:
      BEFORE = deployed guard (regex evidence + unfairness gate) + real Qwen flagger
      AFTER  = manifold flags (calibrated tau) -> guard.escalation_for_flags
      AFTER+ = manifold UNION raw flagger UDAAP proposal (misleading-clarity complement)
  * writes calibration.json into data/gms_regulatory_cap/

    AGENTLAB_USE_LLM_FLAG=1 python scripts/calibrate_eval_manifold_guard.py
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from agentlab.capstone.manifold_evidence import ManifoldFlagScorer, _DEFAULT_ARTIFACT
from agentlab.capstone.regulatory_guard import (
    get_default_guard, _FLAG_TO_ENTITY, _udaap_supported_by_evidence,
)
from agentlab.models.qwen_flagger import get_default_flagger

_ESC_FLAGS = ["UDAAP", "Reg_X"]   # the high-severity flags that drive escalation


def youden_tau(pos: list[float], neg: list[float]) -> float:
    """Threshold (fire iff dist<=tau) maximizing TPR-FPR on dev."""
    cands = sorted(set(pos + neg))
    best_tau, best_j = (cands[0] - 1e-3 if cands else 0.0), -1.0
    for tau in cands:
        tpr = sum(p <= tau for p in pos) / max(len(pos), 1)
        fpr = sum(n <= tau for n in neg) / max(len(neg), 1)
        if tpr - fpr > best_j:
            best_j, best_tau = tpr - fpr, tau
    return best_tau


def main() -> int:
    guard = get_default_guard()
    scorer = ManifoldFlagScorer.load()
    flagger = get_default_flagger()

    rows = [r for r in csv.DictReader(Path("data/capstone_doe_results.csv").open())
            if r["regulatory"] in ("UDAAP", "Reg_X", "none")]
    dist = scorer.flag_distances_batch([r["message"] for r in rows])  # one encoder pass
    for j, r in enumerate(rows):
        r["_d"] = {f: dist[f][j] for f in scorer.centers}

    # stratified 50/50 dev/test split
    rng = random.Random(42)
    dev, test = [], []
    by_reg: dict[str, list] = {}
    for r in rows:
        by_reg.setdefault(r["regulatory"], []).append(r)
    for reg, grp in by_reg.items():
        rng.shuffle(grp)
        h = len(grp) // 2
        dev += grp[:h]; test += grp[h:]

    # calibrate per-flag tau on DEV (Youden's J: class-positives vs none)
    none_dev = [r for r in dev if r["regulatory"] == "none"]
    taus = {}
    for f in _ESC_FLAGS:
        pos = [r["_d"][f] for r in dev if r["regulatory"] == f]
        neg = [r["_d"][f] for r in none_dev]
        taus[f] = youden_tau(pos, neg)
    scorer.thresholds.update(taus)
    print(f"calibrated tau (dev, Youden's J): "
          + "  ".join(f"{f}={taus[f]:.3f}" for f in _ESC_FLAGS))
    (_DEFAULT_ARTIFACT / "calibration.json").write_text(
        json.dumps({"flag_thresholds": taus, "method": "youden_j_dev_split", "seed": 42}, indent=2))

    def manifold_escalate(r, with_flagger=False):
        fired = {f for f in _ESC_FLAGS if r["_d"][f] <= taus[f]}
        if with_flagger:
            prop = flagger.propose(r["message"]) or []
            fired |= {f for f in prop if f in _ESC_FLAGS}
        return guard.escalation_for_flags(sorted(fired))[0]

    def evidence_supported(message):
        """High-severity flags the context-aware hybrid extractor corroborates
        (guard's evidence path: hybrid evidence -> graph support + UDAAP gate)."""
        ev = guard.evidence_in_message(message)            # hybrid (AGENTLAB_USE_LLM_FLAG=1)
        out = set()
        for f in _ESC_FLAGS:
            if guard._flag_supported(_FLAG_TO_ENTITY[f], ev):
                if f == "UDAAP" and not _udaap_supported_by_evidence(ev):
                    continue
                out.add(f)
        return out

    def manifold_and_evidence(r):
        """Manifold recall INTERSECT context-evidence precision."""
        fired = {f for f in _ESC_FLAGS if r["_d"][f] <= taus[f]}
        fired &= evidence_supported(r["message"])
        return guard.escalation_for_flags(sorted(fired))[0]

    def deployed_escalate(r):
        prop = flagger.propose(r["message"]) or []
        saved = guard.evidence_in_message
        guard.evidence_in_message = guard._regex_evidence  # deployed = regex evidence
        try:
            return guard.verify_and_correct(prop, r["message"])["escalate"]
        finally:
            guard.evidence_in_message = saved

    def tally(rs, fn):
        c = {"UDAAP": [0, 0], "Reg_X": [0, 0], "none": [0, 0]}
        ok = 0
        for r in rs:
            esc = fn(r)
            exp = r["expected_escalation"] == "True"
            c[r["regulatory"]][1] += 1; c[r["regulatory"]][0] += int(esc)
            ok += int(esc == exp)
        return c, ok

    n_te = {k: sum(r["regulatory"] == k for r in test) for k in ("UDAAP", "Reg_X", "none")}
    print(f"\nTEST split: UDAAP={n_te['UDAAP']} Reg_X={n_te['Reg_X']} none={n_te['none']}  "
          f"(dev held out for calibration)\n")
    print(f"{'variant':28s} {'UDAAP':>10} {'Reg_X':>10} {'none-false':>12} {'acc':>8}")
    for name, fn in (("BEFORE deployed (regex+flagger)", deployed_escalate),
                     ("AFTER manifold", lambda r: manifold_escalate(r, False)),
                     ("AFTER manifold + flagger union", lambda r: manifold_escalate(r, True)),
                     ("AFTER manifold AND context-evid", manifold_and_evidence)):
        c, ok = tally(test, fn)
        print(f"{name:28s} {c['UDAAP'][0]:>4}/{c['UDAAP'][1]:<5} {c['Reg_X'][0]:>4}/{c['Reg_X'][1]:<5} "
              f"{c['none'][0]:>5}/{c['none'][1]:<6} {ok}/{len(test)} ({ok/len(test):.0%})")
    print("\n(held-out TEST; tau calibrated on DEV only. UDAAP/Reg_X higher=better recall, "
          "none-false=0 ideal.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
