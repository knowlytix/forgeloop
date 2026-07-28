"""Design of experiments: balanced sampling over a factor table.

Each factor's levels appear approximately num_cases / len(levels) times
across the design. This is not a true orthogonal design (Plackett-Burman,
fractional factorial). The goal is coverage of the behavior space, not
statistical purity.
"""

from __future__ import annotations

import random
from typing import Any


def balanced_design(
    factors: dict[str, list[Any]],
    num_cases: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    if num_cases <= 0:
        return []
    if not factors:
        return [{} for _ in range(num_cases)]
    rng = random.Random(seed)
    out: list[dict[str, Any]] = [{} for _ in range(num_cases)]
    for fname, levels in factors.items():
        if not levels:
            continue
        assignments = [levels[i % len(levels)] for i in range(num_cases)]
        rng.shuffle(assignments)
        for i in range(num_cases):
            out[i][fname] = assignments[i]
    return out


def coverage_report(
    design: list[dict[str, Any]],
    factors: dict[str, list[Any]],
) -> dict[str, dict[Any, int]]:
    out: dict[str, dict[Any, int]] = {}
    for fname, levels in factors.items():
        counts: dict[Any, int] = {lv: 0 for lv in levels}
        for row in design:
            v = row.get(fname)
            if v in counts:
                counts[v] += 1
        out[fname] = counts
    return out
