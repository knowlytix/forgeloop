"""A/B for the windowed ManifoldEntityExtractor flag path in the regulatory guard.

Holds the Qwen flagger and every other guard path constant and toggles ONLY the
new ``_manifold_entity_flags`` recall path (windowed entity-point geometry, on top
of the existing whole-message ``_manifold_flags`` cap path), then measures the
escalation decision against the DoE ground truth in
``data/capstone_doe_results.csv``.

Scope matches eval_guard_evidence_before_after.py: the 29 guard-driven regulatory
rows (UDAAP + Reg_X) plus the 60 ``none`` rows that must not escalate. PII /
prompt-injection escalate via the input gates, not the guard, so are excluded.

    AGENTLAB_USE_LLM_FLAG=1 python scripts/eval_manifold_entity_flags.py
"""

from __future__ import annotations

import collections
import csv
import os
from pathlib import Path

from agentlab.capstone.regulatory_guard import get_default_guard
from agentlab.models.qwen_flagger import get_default_flagger

_CSV = Path("data/capstone_doe_results.csv")
_IN_SCOPE = {"UDAAP", "Reg_X", "none"}


def _run(guard, flagger, rows) -> tuple[dict, dict, dict[str, tuple]]:
    """One pass over the rows; returns accuracy, per-class escalation, per-row flags."""
    acc = [0, 0]                                   # [ok, total]
    by_class: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    by_clarity: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    decisions: dict[str, tuple] = {}
    for r in rows:
        msg, reg, clar = r["message"], r["regulatory"], r["clarity"]
        exp = r["expected_escalation"] == "True"
        proposed = flagger.propose(msg) or []
        v = guard.verify_and_correct(proposed, msg)
        esc = v["escalate"]
        acc[1] += 1; acc[0] += int(esc == exp)
        by_class[reg][1] += 1; by_class[reg][0] += int(esc)
        if reg != "none":
            by_clarity[clar][1] += 1; by_clarity[clar][0] += int(esc)
        decisions[r["row"]] = (esc, tuple(v["flags"]), tuple(v["manifold_flags"]))
    return {"acc": acc, "by_class": dict(by_class), "by_clarity": dict(by_clarity)}, decisions, {}


def _fmt(d: list[int]) -> str:
    return f"{d[0]:>2}/{d[1]:<2} ({d[0]/d[1]:.0%})" if d[1] else "  -"


def main() -> int:
    rows = [r for r in csv.DictReader(_CSV.open()) if r["regulatory"] in _IN_SCOPE]
    guard = get_default_guard()
    flagger = get_default_flagger()

    print(f"in-scope rows: {len(rows)} "
          f"(UDAAP={sum(r['regulatory']=='UDAAP' for r in rows)}, "
          f"Reg_X={sum(r['regulatory']=='Reg_X' for r in rows)}, "
          f"none={sum(r['regulatory']=='none' for r in rows)})\n")

    os.environ["AGENTLAB_USE_MANIFOLD_ENTITIES"] = "0"
    off, dec_off, _ = _run(guard, flagger, rows)
    os.environ["AGENTLAB_USE_MANIFOLD_ENTITIES"] = "1"
    on, dec_on, _ = _run(guard, flagger, rows)

    print(f"{'metric':22s} {'OFF (entity path)':22s} ON")
    print(f"{'escalation accuracy':22s} {_fmt(off['acc']):22s} {_fmt(on['acc'])}")
    print("\nper-class escalation rate (recall for UDAAP/Reg_X; want ~0 for none):")
    for c in ("UDAAP", "Reg_X", "none"):
        print(f"  {c:8s} {_fmt(off['by_class'].get(c,[0,0])):22s} {_fmt(on['by_class'].get(c,[0,0]))}")
    print("\nregulatory recall by clarity (UDAAP+Reg_X):")
    for clar in ("clear", "ambiguous", "misleading"):
        print(f"  {clar:11s} {_fmt(off['by_clarity'].get(clar,[0,0])):22s} {_fmt(on['by_clarity'].get(clar,[0,0]))}")

    flipped = [rid for rid in dec_on if dec_on[rid][0] != dec_off[rid][0]]
    print(f"\nrows whose escalation flipped when the entity path turned ON ({len(flipped)}):")
    by = {r["row"]: r for r in rows}
    for rid in flipped:
        r = by[rid]
        eon, fon, mon = dec_on[rid]
        print(f"  row {rid:>3} reg={r['regulatory']:6s} clar={r['clarity']:10s} "
              f"exp={r['expected_escalation']:5s} -> escalate={eon} flags={list(fon)} "
              f"manifold={list(mon)}\n       msg: {r['message'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
