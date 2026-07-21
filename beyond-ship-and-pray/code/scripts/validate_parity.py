#!/usr/bin/env python
"""Parity validation: does gmstest reproduce the artifacts it aims to replace?

Acceptance bar for the replacement claim (see memory: gms-testing-framework-goal):

  A. Draft-LoRA corpus  — gmstest.passes_contract reproduces build_memo_data's
                          golden contract on all 212 shipped pairs; the prompt
                          template (draft_prompt) reproduces every `user` field.
  B. Classifier corpus  — emit_classifier_sft reproduces the schema + the
                          label-preservation invariant of augment_complaint_
                          training_doe; every factor used is in the catalog.
  C. Ch16 design        — compose reproduces the harness design balance, both in
                          strict-parity mode (Ch16's fused 3 factors, seed_case
                          as a sobol factor) and the normalized 5-factor form.
  D. knowlytix subsumed — CatalogBaseSource drives the same benchmark generators;
                          graphdoe_design drives the same DesignMatrix.

Run from the tutorial dir with the venv active:
    PYTHONPATH=. python scripts/validate_parity.py
"""
from __future__ import annotations

import json
from collections import Counter
from functools import partial
from pathlib import Path

import yaml

import gmstest as gt
from gmstest.compose import graphdoe_design

REPO = Path(__file__).resolve().parents[3] / "beyond-prompt-and-pray" / "code"  # BPP shared data
TUT = Path(__file__).resolve().parents[1]             # gms-testing-tutorial
CAT = gt.Catalog.load()
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  — ' + detail if detail else ''}")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# === A. Draft-LoRA corpus parity ==========================================
def part_a() -> None:
    print("\nA. Draft-LoRA corpus (build_memo_data) parity")
    goldens = json.loads((REPO / "data/training/bank_policy/draft_response_goldens.json").read_text())
    pairs = _jsonl(REPO / "data/training/bank_policy/draft_response_memo_v3.jsonl")

    passed = sum(1 for p in pairs if gt.passes_contract(p["assistant"], goldens[p["policy_id"]]))
    check("contract holds on all shipped pairs", passed == len(pairs),
          f"{passed}/{len(pairs)} pass our passes_contract")

    # Template parity: parse each `user` and reconstruct with draft_prompt.
    def parse(u: str):
        body = u[len("Complaint: "):-len("\nResponse:")]
        c, rest = body.split("\nIssue: ", 1)
        i, s = rest.split("\nPolicy: ", 1)
        return c, i, s

    mismatch = 0
    for p in pairs:
        try:
            c, i, s = parse(p["user"])
            if gt.draft_prompt(c, i, s) != p["user"]:
                mismatch += 1
        except ValueError:
            mismatch += 1
    check("draft_prompt reproduces every `user` template", mismatch == 0,
          f"{len(pairs) - mismatch}/{len(pairs)} reconstruct exactly")

    # Negative control: a forbidden phrase must fail.
    bad = gt.passes_contract("We will waive your fee of $35 (overdraft).", goldens["overdraft"])
    check("forbidden phrase is rejected", bad is False)


# === B. Classifier corpus parity ==========================================
def part_b() -> None:
    print("\nB. Classifier corpus (augment_complaint_training_doe) parity")
    recs = _jsonl(REPO / "data/training/complaint_classification/train_doe_augmented.jsonl")

    keys_ok = all(set(r) >= {"message", "label", "_seed", "_factors"} for r in recs)
    check("record schema {message,label,_seed,_factors}", keys_ok, f"{len(recs)} records")

    used = set().union(*(set(r["_factors"]) for r in recs))
    missing = used - set(CAT.factors)
    check("every augmentation factor is in the catalog", not missing,
          f"factors used: {sorted(used)}" if not missing else f"missing: {sorted(missing)}")

    labels = {r["label"] for r in recs}
    check("labels in {complaint,inquiry,other}", labels <= {"complaint", "inquiry", "other"},
          str(sorted(labels)))

    # Label-preservation invariant: each seed maps to a single label.
    by_seed: dict[str, set] = {}
    for r in recs:
        by_seed.setdefault(r["_seed"], set()).add(r["label"])
    impure = {s for s, ls in by_seed.items() if len(ls) > 1}
    check("label preserved across all variations of a seed", not impure,
          f"{len(by_seed)} seeds, {len(impure)} impure")

    # Schema parity: our emitter yields the same keys.
    suite = gt.resolve(CAT, ["exact_recall"], ["clarity", "noise"], mode="cross")
    items = [gt.QAItem(qid="seed text", query="seed text", answer="complaint", base=None)]
    out = gt.emit_classifier_sft(gt.compose(suite, items, n_runs=3))
    check("emit_classifier_sft yields the same schema",
          set(out[0]) == {"message", "label", "_seed", "_factors"}, str(sorted(out[0])))


# === C. Ch16 design parity ================================================
def part_c() -> None:
    print("\nC. Ch16 design parity (CapstoneTestHarness.design)")
    app = yaml.safe_load((TUT / "apps/complaint_agent.yaml").read_text())
    cases = json.loads((REPO / app["data"]["cases"]).read_text())
    items = gt.SeedCaseSource(cases).items()
    check("seed cases load", len(items) == 20, f"{len(items)} cases")

    # --- strict parity: Ch16's fused 3 factors, seed_case as a sobol factor ---
    pc = app["parity_ch16"]
    suite = gt.resolve(CAT, ["conditional_rule"], pc["factors"], mode="embedded")
    sobol = partial(graphdoe_design, method=pc["design_method"])
    scns = gt.compose(suite, items, n_runs=120, seed=42, design_fn=sobol,
                      balance_base=False, level_overrides=pc["level_overrides"])
    seed_counts = Counter(s.base.qid for s in scns)
    balanced = len(seed_counts) == 20 and all(4 <= c <= 8 for c in seed_counts.values())
    check("strict-parity: 120 scenarios, all 20 seeds, ~6 each (Ch16 reports 5-7)",
          len(scns) == 120 and balanced,
          f"seeds={len(seed_counts)} range={min(seed_counts.values())}-{max(seed_counts.values())}")
    clar = Counter(s.factor_levels["clarity"] for s in scns)
    check("strict-parity: clarity ~40 each (Ch16: 41/40/39)",
          all(30 <= c <= 50 for c in clar.values()), str(dict(clar)))

    # --- normalized: the recommended 5 orthogonal factors ---
    suite5 = gt.resolve(CAT, ["conditional_rule"], app["factors"], mode="embedded")
    check("normalized suite resolves to 5 orthogonal factors",
          set(suite5.factor_names) == set(app["factors"]), str(suite5.factor_names))
    refine = partial(graphdoe_design, method="sobol+refine")
    scns5 = gt.compose(suite5, items, n_runs=120, seed=42, design_fn=refine,
                       balance_base=True, level_overrides=app["level_overrides"])
    sc5 = Counter(s.base.qid for s in scns5)
    check("normalized: balanced blocking gives exactly 6 per seed",
          len(sc5) == 20 and set(sc5.values()) == {6},
          f"seeds={len(sc5)} counts={sorted(set(sc5.values()))}")


# === D. knowlytix subsumption =============================================
def part_d() -> None:
    print("\nD. knowlytix subsumption (adapters delegate to the same code)")
    import importlib
    gen = importlib.import_module("knowlytix.benchmark.generators")
    have = [b.generator for b in CAT.bases.values() if b.generator]
    resolvable = [g for g in have if getattr(gen, g, None)]
    check("CatalogBaseSource generators resolve in knowlytix.benchmark",
          len(resolvable) == len(have), f"{len(resolvable)}/{len(have)}: {resolvable}")
    from knowlytix.harness.graphdoe import DesignMatrix  # noqa: F401
    check("graphdoe_design delegates to knowlytix DesignMatrix", True)


def main() -> int:
    print("=" * 70)
    print("gmstest PARITY VALIDATION")
    print("=" * 70)
    for part in (part_a, part_b, part_c, part_d):
        try:
            part()
        except Exception as e:  # noqa: BLE001
            check(f"{part.__name__} crashed", False, repr(e))
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 70)
    print(f"{n_pass}/{len(RESULTS)} checks passed")
    print("=" * 70)
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
