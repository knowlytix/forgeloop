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
    """One base question type defining content and its ground truth.

    Attributes:
        name: Unique base-category identifier and the catalog lookup key.
        family: Family this base belongs to, used for family-level selection.
        category: Free-text category label.
        description: Human-readable description of the base question.
        requires: Names of prerequisites the base depends on.
        generator: knowlytix benchmark generator class name, or None when no
            generator exists yet.
        answer_type: Type of the ground-truth answer (for example "str" or "decision").
        ground_truth: Description of how the ground truth is derived.
        multiplicity: Cardinality of the expected answer.
        applicability: Contexts in which the base applies.
        status: Build status, defaulting to "build".
        source: Source references for the base.
    """
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
    """One enrichment factor, presentation only and ground-truth invariant.

    Attributes:
        name: Unique factor identifier and the catalog lookup key.
        section: Factor-catalog section the factor was read from.
        cardinality: Number of levels, defaulting to the number of categories.
        levels: The factor's level vocabulary.
        description: Human-readable description of the factor.
        gt_invariant: Whether varying the factor leaves ground truth unchanged.
        applies_families: Base families the factor applies to; ["*"] means all.
        applies_answer_types: Base answer types the factor applies to; ["*"] means all.
    """
    name: str
    section: str
    cardinality: int
    levels: list[str]
    description: str
    gt_invariant: bool
    applies_families: list[str]      # ["*"] means all
    applies_answer_types: list[str]  # ["*"] means all

    def applies_to(self, base: "BaseSpec") -> bool:
        """Return whether this factor applies to the given base by family and answer type."""
        fam_ok = "*" in self.applies_families or base.family in self.applies_families
        at_ok = "*" in self.applies_answer_types or base.answer_type in self.applies_answer_types
        return fam_ok and at_ok


@dataclass(frozen=True)
class Profile:
    """A selection bundle: chosen bases, chosen factors and a composition mode.

    Attributes:
        name: Profile identifier and the catalog lookup key.
        description: Human-readable description of the profile.
        bases: Base selection tokens (names, family names or "*").
        factors: Factor selection tokens (names, group names or "*").
        mode: Composition mode, either "cross" or "embedded".
    """
    name: str
    description: str
    bases: list[str]
    factors: list[str]
    mode: str  # "cross" | "embedded"


@dataclass
class Catalog:
    """The loaded base, factor and profile catalogs with a family index.

    Attributes:
        bases: Base specs keyed by base name.
        factors: Factor specs keyed by factor name.
        factor_groups: Named factor groups mapping a group name to its factor names.
        profiles: Profile specs keyed by profile name.
        families: Base names grouped by family name.
    """
    bases: dict[str, BaseSpec]
    factors: dict[str, FactorSpec]
    factor_groups: dict[str, list[str]]
    profiles: dict[str, Profile]
    families: dict[str, list[str]] = field(default_factory=dict)

    # -- loading ------------------------------------------------------------
    @classmethod
    def load(cls, catalogs_dir: str | Path | None = None) -> "Catalog":
        """Load the base, factor and profile catalogs from a directory.

        Args:
            catalogs_dir: Directory holding base_catalog.yaml, factor_catalog.yaml
                and optional profiles.yaml; defaults to the package's catalogs/ directory.

        Returns:
            A Catalog populated from the YAML files.
        """
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
        """Return a one-line count of bases, families, factors, groups and profiles."""
        n_groups = len({f.section for f in self.factors.values()})
        return (f"{len(self.bases)} base categories in {len(self.families)} families, "
                f"{len(self.factors)} factors in {n_groups} groups, "
                f"{len(self.profiles)} profiles")


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
