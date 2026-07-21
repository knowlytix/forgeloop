"""Compose scenarios by enriching base items with a design matrix.

Two modes (both hold ground truth invariant):

  cross     Option 1 — base x design. The design covers presentation factors
            only; every base item is crossed with every design row.
            #scenarios = #base_items * n_runs
  embedded  Option 2 — base is a factor. `base_question` is added to the design
            as a categorical factor whose levels are the base item ids, sampled
            jointly with the presentation factors (the JASA Table 1 construction).
            #scenarios = n_runs

Design generation is pluggable. `simple_design` is dependency-light and
deterministic (good for tests and offline use). `graphdoe_design` uses the
knowlytix Sobol+refine generator for real space-filling coverage.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from .catalogs import FactorSpec
from .resolve import ResolvedSuite
from .sources import QAItem

# A design function maps (level_specs, n_runs, seed) -> list of rows, where each
# row is {factor_name: level}. level_specs is a list of (name, levels) pairs.
LevelSpec = list[tuple[str, list[str]]]
DesignFn = Callable[[LevelSpec, int, int], list[dict[str, str]]]


@dataclass
class Scenario:
    sid: str
    base: QAItem
    factor_levels: dict[str, str]
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)


# -- design functions -------------------------------------------------------
def simple_design(level_specs: LevelSpec, n_runs: int, seed: int = 42) -> list[dict[str, str]]:
    """Deterministic per-factor random sampling. No external dependencies."""
    if not level_specs:
        return [{}]
    rng = random.Random(seed)
    rows = []
    for _ in range(n_runs):
        rows.append({name: rng.choice(levels) for name, levels in level_specs})
    return rows


def graphdoe_design(level_specs: LevelSpec, n_runs: int, seed: int = 42,
                    method: str = "sobol+refine") -> list[dict[str, str]]:
    """Sobol/Sobol+refine space-filling design via knowlytix graphdoe."""
    from knowlytix.harness.graphdoe import DesignMatrix

    if not level_specs:
        return [{}]
    factors = [{"name": name, "type": "categorical", "categories": levels}
               for name, levels in level_specs]
    df = DesignMatrix(factors, method=method, n_runs=n_runs, seed=seed).generate()
    return df.to_dict(orient="records")


# -- composition ------------------------------------------------------------
def _balanced_assignment(qids: list[str], n_runs: int, seed: int) -> list[str]:
    """Each base id appears ~n_runs/k times, seed-shuffled (blocking factor)."""
    k = len(qids)
    assign = (qids * ((n_runs // k) + 1))[:n_runs]
    random.Random(seed).shuffle(assign)
    return assign


def compose(
    suite: ResolvedSuite,
    base_items: list[QAItem],
    n_runs: int = 64,
    seed: int = 42,
    design_fn: DesignFn = simple_design,
    balance_base: bool = True,
    level_overrides: dict[str, list[str]] | None = None,
) -> list[Scenario]:
    """Compose scenarios from a resolved suite and base items.

    embedded mode: the presentation factors are space-filled by `design_fn`; the
    base question is assigned as a *balanced blocking* factor by default
    (`balance_base=True`) so every base is exercised equally — robust at small
    n, where joint phi_p optimization would starve a high-cardinality
    base_question (the geometric-vs-marginal-coverage tradeoff; see the JASA
    paper). Set `balance_base=False` to put base_question into the design matrix
    as a joint factor (pure Option 2 / JASA Table 1, best at large n).

    `level_overrides` lets an application supply its own level vocabulary for a
    catalog factor (e.g. entity_aliasing -> [canonical, alias]). The general
    catalog stays domain-neutral; the override lives in the application layer.
    """
    if not base_items:
        return []
    by_qid = {it.qid: it for it in base_items}
    overrides = level_overrides or {}
    pres_specs: LevelSpec = [(f.name, overrides.get(f.name, f.levels)) for f in suite.factors]
    scenarios: list[Scenario] = []

    if suite.mode == "cross":
        rows = design_fn(pres_specs, n_runs, seed) if pres_specs else [{}]
        for it in base_items:
            for r, row in enumerate(rows):
                scenarios.append(Scenario(
                    sid=f"{it.qid}~{r:04d}", base=it,
                    factor_levels=dict(row), mode="cross"))
        return scenarios

    # embedded
    if balance_base:
        rows = (design_fn(pres_specs, n_runs, seed) if pres_specs
                else [{} for _ in range(n_runs)])
        assign = _balanced_assignment([it.qid for it in base_items], len(rows), seed)
        for r, (row, qid) in enumerate(zip(rows, assign)):
            it = by_qid[qid]
            scenarios.append(Scenario(
                sid=f"{qid}~{r:04d}", base=it,
                factor_levels=dict(row), mode="embedded"))
    else:
        specs = [("base_question", [it.qid for it in base_items])] + pres_specs
        for r, row in enumerate(design_fn(specs, n_runs, seed)):
            row = dict(row)
            it = by_qid[row.pop("base_question")]
            scenarios.append(Scenario(
                sid=f"{it.qid}~{r:04d}", base=it,
                factor_levels=row, mode="embedded"))
    return scenarios
