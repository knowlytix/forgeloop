#!/usr/bin/env python
"""Validate the GMS-testing catalogs for completeness and consistency.

Enforces the base<->enrichment contract:

  1. Every factor has gt_invariant: true (presentation must not change truth).
  2. Every base.requires is drawn from the allowed graph-capability vocabulary.
  3. Every factor.applies_to.families references real base families (or "*");
     every applies_to.answer_types references a real base answer_type (or "*").
  4. Names are globally unique and snake_case across both catalogs.
  5. No name appears in both catalogs (the documented boundary/edge_case pair
     uses distinct names, so this must hold).

Also checks that every profile in profiles.yaml references real bases/factors.

Usage:
    python scripts/validate_catalogs.py
Exit code 0 = all checks pass; 1 = violations found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

CATALOGS = Path(__file__).resolve().parent.parent / "catalogs"

FAMILIES = {
    "retrieval_recall",
    "comparison_ranking",
    "reasoning_composition",
    "consistency_integrity",
    "behavioral_operational",
}
REQUIRES_VOCAB = {
    "enm", "triples", "path", "threshold",
    "ordered_sequence", "negatives", "provenance", "policy",
}
ANSWER_TYPES = {
    "float", "int", "bool", "set_str", "str",
    "tuple", "mcq", "list", "decision",
}
SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")


def _load(name: str) -> dict:
    with open(CATALOGS / name) as f:
        return yaml.safe_load(f)


def _iter_bases(base_doc: dict):
    for family, entries in base_doc.items():
        for e in entries:
            yield e


def _iter_factors(factor_doc: dict):
    for section, entries in factor_doc.items():
        if section == "factor_groups":
            continue
        for e in entries:
            yield e


def main() -> int:
    base_doc = _load("base_catalog.yaml")
    factor_doc = _load("factor_catalog.yaml")
    errors: list[str] = []

    bases = list(_iter_bases(base_doc))
    factors = list(_iter_factors(factor_doc))
    base_names = {b["name"] for b in bases}
    factor_names = {f["name"] for f in factors}
    base_answer_types = {b["answer_type"] for b in bases}

    # Rule 1: factors are ground-truth invariant.
    for f in factors:
        if f.get("gt_invariant") is not True:
            errors.append(f"factor '{f['name']}': gt_invariant must be true")

    # Rule 2: base.requires vocabulary; base.family validity.
    for b in bases:
        if b["family"] not in FAMILIES:
            errors.append(f"base '{b['name']}': unknown family '{b['family']}'")
        for cap in b.get("requires", []):
            if cap not in REQUIRES_VOCAB:
                errors.append(f"base '{b['name']}': unknown requires capability '{cap}'")
        if b["answer_type"] not in ANSWER_TYPES:
            errors.append(f"base '{b['name']}': unknown answer_type '{b['answer_type']}'")

    # Rule 3: factor.applies_to references real families / answer_types.
    for f in factors:
        at = f.get("applies_to", {})
        for fam in at.get("families", []):
            if fam != "*" and fam not in FAMILIES:
                errors.append(f"factor '{f['name']}': applies_to.families has unknown '{fam}'")
        for atype in at.get("answer_types", []):
            if atype != "*" and atype not in ANSWER_TYPES:
                errors.append(f"factor '{f['name']}': applies_to.answer_types has unknown '{atype}'")

    # Rule 4: snake_case + uniqueness within each catalog.
    for nm in base_names | factor_names:
        if not SNAKE.match(nm):
            errors.append(f"name '{nm}' is not snake_case")
    if len(base_names) != len(bases):
        errors.append("duplicate base names detected")
    if len(factor_names) != len(factors):
        errors.append("duplicate factor names detected")

    # Rule 5: no overlap between the two catalogs.
    overlap = base_names & factor_names
    if overlap:
        errors.append(f"names appear in BOTH catalogs: {sorted(overlap)}")

    # Profiles reference real things.
    try:
        prof_doc = _load("profiles.yaml")
        groups = set(factor_doc.get("factor_groups", {}))
        for pname, p in prof_doc.get("profiles", {}).items():
            for b in p.get("bases", []):
                if b not in ("*",) and b not in base_names and b not in FAMILIES:
                    errors.append(f"profile '{pname}': unknown base/family '{b}'")
            for fc in p.get("factors", []):
                if fc not in ("*",) and fc not in factor_names and fc not in groups:
                    errors.append(f"profile '{pname}': unknown factor/group '{fc}'")
            if p.get("mode") not in ("cross", "embedded"):
                errors.append(f"profile '{pname}': mode must be cross|embedded")
    except FileNotFoundError:
        pass

    # Report.
    print(f"base categories : {len(bases)} across {len(base_doc)} families")
    print(f"factors         : {len(factors)} across "
          f"{len([k for k in factor_doc if k != 'factor_groups'])} groups")
    print(f"applicable answer_types in base: {sorted(base_answer_types)}")
    if errors:
        print(f"\nFAIL — {len(errors)} violation(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nOK — all consistency checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
