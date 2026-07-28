"""Calibrate ManifoldEntityExtractor per-entity thresholds against the eval cases
and persist them to ``data/gms_regulatory_cap/entity_calibration.json``.

The eval set (``data/eval_cases/cases.json``) carries *flag*-level ground truth
in ``factors.regulatory`` --- not per-evidence-entity labels --- so we can only
honestly calibrate the entities that ground truth pins down: the high-severity
regulatory flags ``UDAAP`` and ``Reg_X`` (the two flags the cap artifact was
built for, and the signals the manifold path exists to recover). Every other
evidence entity has no gold here, gets no positive, and is left at its default
threshold --- and is *not* written to the file, so the loader falls back to the
default for it rather than persisting an un-calibrated number.

Calibration is in-sample over a 20-case dev set with very few positives, so the
fitted thresholds are coarse by construction. The script prints the per-entity
TP/FP/FN before and after so that is visible, not hidden.

    python scripts/calibrate_manifold_entities.py            # report + persist
    python scripts/calibrate_manifold_entities.py --dry-run  # report only
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

from agentlab.capstone.manifold_entities import ManifoldEntityExtractor
from agentlab.capstone.manifold_evidence import _DEFAULT_ARTIFACT

_CASES = Path("data/eval_cases/cases.json")
_OUT = _DEFAULT_ARTIFACT / "entity_calibration.json"

# The only entities cases.json can supply ground truth for (flag -> entity).
_FACTOR_TO_ENTITY = {"UDAAP": "udaap", "Reg_X": "reg_x"}


def _gold(case: dict) -> set[str]:
    reg = (case.get("factors") or {}).get("regulatory")
    ent = _FACTOR_TO_ENTITY.get(reg)
    return {ent} if ent else set()


def _confusion(rows: list[tuple[str, set[str]]], ext, entity: str, tau: float) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for text, gold in rows:
        d = ext.entity_distances(text)[entity]
        fired, want = d <= tau, entity in gold
        tp += fired and want
        fp += fired and not want
        fn += (not fired) and want
    return tp, fp, fn


def _f1(tp: int, fp: int, fn: int) -> float:
    return 0.0 if tp == 0 else tp / (tp + 0.5 * (fp + fn))


def main() -> None:
    dry = "--dry-run" in sys.argv
    cases = json.loads(_CASES.read_text())
    labeled = [(c["message"], _gold(c)) for c in cases]

    targets = sorted(_FACTOR_TO_ENTITY.values())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Calibrate within the full production vocab so each entity competes in the
        # real geometry; only the flag entities have gold and will actually move.
        ext = ManifoldEntityExtractor.load(default_tau=1.0)

    missing = [e for e in targets if e not in ext.centers]
    if missing:
        print(f"WARNING: target entities absent from store, skipping: {missing}")
    targets = [e for e in targets if e in ext.centers]

    print(f"cases={len(labeled)}  targets={targets}")
    for e in targets:
        npos = sum(1 for _, g in labeled if e in g)
        print(f"  {e}: {npos} positive case(s)")
    print()

    before = {e: ext.thresholds[e] for e in targets}
    ext.calibrate(labeled)                       # fits per-entity tau by F1

    fitted: dict[str, float] = {}
    print(f"{'entity':10s} {'tau':>18s}   {'before (TP/FP/FN,F1)':>26s}   {'after (TP/FP/FN,F1)':>26s}")
    for e in targets:
        b_tau, a_tau = before[e], ext.thresholds[e]
        bt = _confusion(labeled, ext, e, b_tau)
        at = _confusion(labeled, ext, e, a_tau)
        print(f"{e:10s} {b_tau:6.2f} -> {a_tau:6.2f}   "
              f"{str(bt):>14s} f1={_f1(*bt):.2f}   {str(at):>14s} f1={_f1(*at):.2f}")
        fitted[e] = a_tau

    payload = {
        "entity_thresholds": fitted,
        "source": str(_CASES),
        "method": "in-sample F1 max over geodesic-radius grid; gold = factors.regulatory",
        "n_cases": len(labeled),
    }
    if dry:
        print("\n--dry-run: not writing.\n" + json.dumps(payload, indent=2))
        return
    _OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {_OUT} ({len(fitted)} calibrated entit{'y' if len(fitted)==1 else 'ies'})")


if __name__ == "__main__":
    main()
