#!/usr/bin/env python
"""Prototype: complaint -> facts -> policy-atom entailment -> policy mapping.

Tests the intermediate-representation architecture (vs direct complaint->policy).
For each complaint, a single structured LLM pass (atoms supplied in the prompt)
extracts complaint FACTS, judges each policy atom's APPLICABILITY (entailed /
contradicted / missing_facts / not_applicable), and emits the primary policy,
missing information, and route -- the auditable decision-support object. Scored
on top-k POLICY recall against expected_policy (accept-set), plus the vague-case
behaviour (a complaint that triggers NO atom should map to NO policy, not a
fabricated one) and whether missing-info is surfaced.

Uses the authenticated `claude` CLI (Sonnet) for the prototype; production would
distil the fact+entailment behaviour into the local Qwen path. Run after the
other arms; writes data/prototype_policy_mapping.json.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
_TOPK = 3


def _atoms_block():
    atoms = yaml.safe_load((_REPO / "data" / "policy_atoms.yaml").read_text())
    lines = []
    for a in atoms:
        trig = " | ".join(a["trigger_conditions"])
        lines.append(f'- {a["policy"]} (atom {a["atom_id"]}, products {a["product_scope"]}): '
                     f'TRIGGER when {trig}. evidence_needed={a["evidence_needed"]}. '
                     f'exclusions={a.get("exclusions", [])}.')
    return "\n".join(lines), sorted({a["policy"] for a in atoms})


_ATOMS, _POLICIES = _atoms_block()
_SYS = (
    "You are a bank complaint-handling assistant. Do NOT match on text similarity; "
    "reason about which policy OBLIGATIONS the alleged FACTS trigger.\n"
    "Policy atoms:\n" + _ATOMS + "\n\n"
    "For each input message return one JSON object with: "
    '"i" (index); "facts" {product, event, allegation, harm, remedy, evidence_present(list), missing_info(list)}; '
    '"applicability" (list of {policy, verdict in [entailed,contradicted,missing_facts,not_applicable]}); '
    '"primary_policy" (the single best-supported policy, or null if NONE is entailed/missing_facts -- '
    "a vague or purely emotional complaint with no concrete triggering event maps to null); "
    '"route". Return ONLY a JSON array of these objects, in input order. No commentary.'
)


def _run_batch(items):
    lines = "\n".join(f"{i}: {m}" for i, m in items)
    prompt = f"{_SYS}\n\nMessages:\n{lines}"
    out = subprocess.run(["claude", "-p", "--model", "sonnet"], input=prompt,
                         capture_output=True, text=True, timeout=400).stdout
    m = re.search(r"\[.*\]", out, re.DOTALL)
    res = {}
    if m:
        try:
            for o in json.loads(m.group(0)):
                res[int(o["i"])] = o
        except Exception:
            pass
    return res


def main() -> int:
    diag = json.loads((_REPO / "data" / "diag_extract_facts.json").read_text())
    rows = diag["rows"]
    cases = json.loads((_REPO / "data" / "eval_cases" / "cases.json").read_text())
    by_id = {c["id"]: c for c in (cases if isinstance(cases, list) else cases["cases"])}

    preds = {}
    B = 8
    for b in range(0, len(rows), B):
        batch = [(i, rows[i]["message"]) for i in range(b, min(b + B, len(rows)))]
        preds.update(_run_batch(batch))
        print(f"  mapped {len(preds)}/{len(rows)}", flush=True)

    # scoring
    pol_hit = pol_tot = 0          # top-k policy recall on cases WITH an expected_policy
    vague_ok = vague_tot = 0       # None-policy cases -> should map to null
    missing_flagged = 0
    unparsed = 0
    detail = []
    for idx, r in enumerate(rows):
        case = by_id[r["seed_case"]]
        exp = case.get("expected_policy")
        acc = set(case.get("acceptable_policies") or ([exp] if exp else []))
        o = preds.get(idx)
        if o is None:
            unparsed += 1
        applic = (o or {}).get("applicability") or []
        # candidate policies = entailed first, then missing_facts (lower confidence)
        ranked = [a.get("policy") for a in applic if a.get("verdict") == "entailed"]
        ranked += [a.get("policy") for a in applic if a.get("verdict") == "missing_facts"]
        primary = (o or {}).get("primary_policy")
        topk = ([primary] if primary else []) + [p for p in ranked if p != primary]
        topk = [p for p in topk if p][:_TOPK]
        if exp:
            pol_tot += 1
            pol_hit += int(bool(acc & set(topk)))
        else:
            vague_tot += 1
            vague_ok += int(primary in (None, "null", "", "none"))
        if (o or {}).get("facts", {}).get("missing_info"):
            missing_flagged += 1
        detail.append({"seed": case["id"], "expected_policy": exp,
                       "primary": primary, "topk": topk,
                       "missing_info": (o or {}).get("facts", {}).get("missing_info")})

    out = {
        "n_rows": len(rows), "unparsed": unparsed,
        "policy_recall_at_%d" % _TOPK: [pol_hit, pol_tot,
                                        round(pol_hit / pol_tot, 3) if pol_tot else None],
        "vague_to_null": [vague_ok, vague_tot,
                          round(vague_ok / vague_tot, 3) if vague_tot else None],
        "missing_info_flagged": missing_flagged,
    }
    print("\n=== prototype: facts -> atom entailment -> policy ===")
    print(json.dumps(out, indent=2))
    print("sample traces:")
    for d in detail[:8]:
        print(f"   [{d['seed']}] exp={d['expected_policy']} -> primary={d['primary']} "
              f"topk={d['topk']} missing={d['missing_info']}")
    (_REPO / "data" / "prototype_policy_mapping.json").write_text(
        json.dumps({"summary": out, "detail": detail}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
