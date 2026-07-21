"""Resolve a profile or an explicit selection into a concrete suite.

Expansion rules:
  bases    : "*" -> all bases; a family name -> all bases in that family;
             otherwise a base name.
  factors  : "*" -> all factors; a factor-group name -> the group's factors;
             otherwise a factor name.

After expansion, factors are filtered to those whose `applies_to` matches at
least one selected base (this is "enrichment enriches the base"). Dropped
factors are recorded for transparency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .catalogs import BaseSpec, Catalog, FactorSpec


@dataclass
class ResolvedSuite:
    bases: list[BaseSpec]
    factors: list[FactorSpec]
    mode: str
    dropped_factors: list[tuple[str, str]] = field(default_factory=list)
    profile: str | None = None

    @property
    def base_names(self) -> list[str]:
        return [b.name for b in self.bases]

    @property
    def factor_names(self) -> list[str]:
        return [f.name for f in self.factors]

    def summary(self) -> str:
        s = (f"suite[{self.profile or 'custom'}]: {len(self.bases)} bases x "
             f"{len(self.factors)} factors, mode={self.mode}")
        if self.dropped_factors:
            s += f" ({len(self.dropped_factors)} factors dropped by applies_to)"
        return s


def _expand_bases(catalog: Catalog, selection: list[str]) -> list[BaseSpec]:
    out: dict[str, BaseSpec] = {}
    for token in selection:
        if token == "*":
            out.update(catalog.bases)
        elif token in catalog.families:
            for name in catalog.families[token]:
                out[name] = catalog.bases[name]
        elif token in catalog.bases:
            out[token] = catalog.bases[token]
        else:
            raise KeyError(f"unknown base/family '{token}'")
    return list(out.values())


def _expand_factors(catalog: Catalog, selection: list[str]) -> list[FactorSpec]:
    out: dict[str, FactorSpec] = {}
    for token in selection:
        if token == "*":
            out.update(catalog.factors)
        elif token in catalog.factor_groups:
            for name in catalog.factor_groups[token]:
                if name in catalog.factors:
                    out[name] = catalog.factors[name]
        elif token in catalog.factors:
            out[token] = catalog.factors[token]
        else:
            raise KeyError(f"unknown factor/group '{token}'")
    return list(out.values())


def applicable_factors(
    factors: list[FactorSpec], bases: list[BaseSpec]
) -> tuple[list[FactorSpec], list[tuple[str, str]]]:
    """Keep only factors that apply to at least one selected base."""
    kept, dropped = [], []
    for f in factors:
        if any(f.applies_to(b) for b in bases):
            kept.append(f)
        else:
            dropped.append((f.name, "applies_to matches no selected base"))
    return kept, dropped


def resolve(
    catalog: Catalog,
    bases: list[str],
    factors: list[str],
    mode: str = "embedded",
    filter_factors: bool = True,
) -> ResolvedSuite:
    if mode not in ("cross", "embedded"):
        raise ValueError(f"mode must be 'cross' or 'embedded', got {mode!r}")
    base_specs = _expand_bases(catalog, bases)
    factor_specs = _expand_factors(catalog, factors)
    dropped: list[tuple[str, str]] = []
    if filter_factors:
        factor_specs, dropped = applicable_factors(factor_specs, base_specs)
    return ResolvedSuite(bases=base_specs, factors=factor_specs, mode=mode,
                         dropped_factors=dropped)


def resolve_profile(catalog: Catalog, name: str, filter_factors: bool = True) -> ResolvedSuite:
    if name not in catalog.profiles:
        raise KeyError(f"unknown profile '{name}'")
    p = catalog.profiles[name]
    suite = resolve(catalog, p.bases, p.factors, p.mode, filter_factors=filter_factors)
    suite.profile = name
    return suite
