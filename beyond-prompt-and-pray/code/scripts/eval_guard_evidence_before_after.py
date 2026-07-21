"""Full before/after for the regulatory guard's evidence extraction.

Holds the Qwen flagger constant and toggles ONLY the guard's evidence step ---
regex-alias (before) vs the wired hybrid regex+LLM extractor (after) --- then
measures the escalation decision against the DoE ground truth in
``data/capstone_doe_results.csv``.

Scope: the guard drives the *regulatory* escalations (UDAAP + Reg_X, the 29
rows with ``expected_trigger == 'regulatory'``); the 60 ``none`` rows must not
escalate. PII / prompt-injection rows escalate via the input gates, not the
guard, so they are out of scope and excluded.

    AGENTLAB_USE_LLM_FLAG=1 python scripts/eval_guard_evidence_before_after.py
"""

from __future__ import annotations

import collections
import csv
from pathlib import Path

from agentlab.capstone.regulatory_guard import get_default_guard
from agentlab.models.qwen_flagger import get_default_flagger

_CSV = Path("data/capstone_doe_results.csv")
_IN_SCOPE = {"UDAAP", "Reg_X", "none"}  # guard-driven escalation + true negatives


def _esc(guard, evidence_fn, proposed, message) -> tuple[bool, list[str], list[str]]:
    saved = guard.evidence_in_message
    guard.evidence_in_message = evidence_fn  # type: ignore[method-assign]
    try:
        v = guard.verify_and_correct(proposed, message)
        return v["escalate"], v["flags"], v["evidence"]
    finally:
        guard.evidence_in_message = saved  # type: ignore[method-assign]


def main() -> int:
    rows = [r for r in csv.DictReader(_CSV.open()) if r["regulatory"] in _IN_SCOPE]
    guard = get_default_guard()
    flagger = get_default_flagger()

    regex_fn = guard._regex_evidence          # before: deterministic alias match
    hybrid_fn = guard.evidence_in_message     # after: wired hybrid (regex + LLM)

    # counters[(phase, regulatory)] -> [n_escalated, n_total]
    ctr: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0])
    by_clarity: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0])
    acc: dict[str, list[int]] = {"before": [0, 0], "after": [0, 0]}  # [ok, total]
    changed: list[str] = []

    for r in rows:
        msg, reg, clar = r["message"], r["regulatory"], r["clarity"]
        exp = r["expected_escalation"] == "True"
        proposed = flagger.propose(msg) or []      # shared across before/after

        eb, fb, _ = _esc(guard, regex_fn, proposed, msg)
        ea, fa, ev = _esc(guard, hybrid_fn, proposed, msg)

        for phase, esc in (("before", eb), ("after", ea)):
            ctr[(phase, reg)][1] += 1
            ctr[(phase, reg)][0] += int(esc)
            acc[phase][1] += 1
            acc[phase][0] += int(esc == exp)
            if reg != "none":  # recall by clarity, regulatory rows only
                by_clarity[(phase, clar)][1] += 1
                by_clarity[(phase, clar)][0] += int(esc)
        if eb != ea:
            changed.append(
                f"  {r['seed_case']:9s} reg={reg:6s} clar={clar:10s} exp={exp} "
                f"before={eb} after={ea} proposed={proposed} after_flags={fa} evidence={ev}"
            )

    print(f"in-scope rows: {len(rows)} "
          f"(UDAAP={sum(r['regulatory']=='UDAAP' for r in rows)}, "
          f"Reg_X={sum(r['regulatory']=='Reg_X' for r in rows)}, "
          f"none={sum(r['regulatory']=='none' for r in rows)})\n")

    print(f"{'regulatory':12s} {'before (escalated/total)':26s} after")
    for reg in ("UDAAP", "Reg_X", "none"):
        b, a = ctr[("before", reg)], ctr[("after", reg)]
        tag = "  <- must stay ~0" if reg == "none" else "  <- recall, want up"
        print(f"{reg:12s} {b[0]:>3}/{b[1]:<22} {a[0]:>3}/{a[1]}{tag}")

    print("\nregulatory recall by clarity (UDAAP+Reg_X):")
    for clar in ("clear", "ambiguous", "misleading"):
        b, a = by_clarity[("before", clar)], by_clarity[("after", clar)]
        print(f"  {clar:11s} before {b[0]:>2}/{b[1]:<3} -> after {a[0]:>2}/{a[1]}")

    ab, aa = acc["before"], acc["after"]
    print(f"\nescalation accuracy (all in-scope): "
          f"before {ab[0]}/{ab[1]} ({ab[0]/ab[1]:.0%}) -> after {aa[0]}/{aa[1]} ({aa[0]/aa[1]:.0%})")

    print(f"\nrows whose escalation flipped ({len(changed)}):")
    print("\n".join(changed) if changed else "  (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
