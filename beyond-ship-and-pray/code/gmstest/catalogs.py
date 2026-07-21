"""Load and represent the base / factor / profile catalogs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Default location: the catalogs/ dir alongside this package's parent.
_DEFAULT_CATALOGS = Path(__file__).resolve().parent.parent / "catalogs"

# Section keys in factor_catalog.yaml that are not factor lists.
_NON_FACTOR_SECTIONS = {"factor_groups"}


@dataclass(frozen=True)
class BaseSpec:
    """One base question type — defines content and its ground truth."""
    name: str
    family: str
    category: str
    description: str
    requires: list[str]
    generator: str | None
    answer_type: str
    ground_truth: str
    multiplicity: str
    applicability: list[str]
    status: str
    source: list[str]


@dataclass(frozen=True)
class FactorSpec:
    """One enrichment factor — presentation only, ground-truth invariant."""
    name: str
    section: str
    cardinality: int
    levels: list[str]
    description: str
    gt_invariant: bool
    applies_families: list[str]      # ["*"] means all
    applies_answer_types: list[str]  # ["*"] means all

    def applies_to(self, base: "BaseSpec") -> bool:
        fam_ok = "*" in self.applies_families or base.family in self.applies_families
        at_ok = "*" in self.applies_answer_types or base.answer_type in self.applies_answer_types
        return fam_ok and at_ok


@dataclass(frozen=True)
class Profile:
    """A pick-and-choose bundle: base selection + factor selection + mode."""
    name: str
    description: str
    bases: list[str]
    factors: list[str]
    mode: str  # "cross" | "embedded"


@dataclass
class Catalog:
    bases: dict[str, BaseSpec]
    factors: dict[str, FactorSpec]
    factor_groups: dict[str, list[str]]
    profiles: dict[str, Profile]
    families: dict[str, list[str]] = field(default_factory=dict)

    # -- loading ------------------------------------------------------------
    @classmethod
    def load(cls, catalogs_dir: str | Path | None = None) -> "Catalog":
        d = Path(catalogs_dir) if catalogs_dir else _DEFAULT_CATALOGS

        base_doc = _read_yaml(d / "base_catalog.yaml")
        bases: dict[str, BaseSpec] = {}
        families: dict[str, list[str]] = {}
        for _family, entries in base_doc.items():
            for e in entries:
                spec = BaseSpec(
                    name=e["name"],
                    family=e["family"],
                    category=e.get("category", ""),
                    description=e.get("description", ""),
                    requires=list(e.get("requires", [])),
                    generator=e.get("generator"),
                    answer_type=e["answer_type"],
                    ground_truth=e.get("ground_truth", ""),
                    multiplicity=e.get("multiplicity", ""),
                    applicability=list(e.get("applicability", [])),
                    status=e.get("status", "build"),
                    source=list(e.get("source", [])),
                )
                bases[spec.name] = spec
                families.setdefault(spec.family, []).append(spec.name)

        factor_doc = _read_yaml(d / "factor_catalog.yaml")
        factors: dict[str, FactorSpec] = {}
        for section, entries in factor_doc.items():
            if section in _NON_FACTOR_SECTIONS:
                continue
            for e in entries:
                at = e.get("applies_to", {}) or {}
                spec = FactorSpec(
                    name=e["name"],
                    section=section,
                    cardinality=int(e.get("cardinality", len(e.get("categories", [])))),
                    levels=list(e.get("categories", [])),
                    description=e.get("description", ""),
                    gt_invariant=bool(e.get("gt_invariant", False)),
                    applies_families=list(at.get("families", ["*"])),
                    applies_answer_types=list(at.get("answer_types", ["*"])),
                )
                factors[spec.name] = spec
        factor_groups = {k: list(v["includes"]) for k, v in
                         factor_doc.get("factor_groups", {}).items()}

        profiles: dict[str, Profile] = {}
        prof_path = d / "profiles.yaml"
        if prof_path.exists():
            for pname, p in (_read_yaml(prof_path).get("profiles", {}) or {}).items():
                profiles[pname] = Profile(
                    name=pname,
                    description=p.get("description", ""),
                    bases=list(p.get("bases", [])),
                    factors=list(p.get("factors", [])),
                    mode=p.get("mode", "embedded"),
                )

        return cls(bases=bases, factors=factors, factor_groups=factor_groups,
                   profiles=profiles, families=families)

    # -- convenience --------------------------------------------------------
    def summary(self) -> str:
        n_groups = len({f.section for f in self.factors.values()})
        return (f"{len(self.bases)} base categories in {len(self.families)} families, "
                f"{len(self.factors)} factors in {n_groups} groups, "
                f"{len(self.profiles)} profiles")


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
