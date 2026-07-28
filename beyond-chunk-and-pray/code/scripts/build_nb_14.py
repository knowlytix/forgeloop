"""Builder for notebooks/14_calibration_a_inline.ipynb (Ch12 — Calibrate, don't guess).

Emits a valid nbformat-4 notebook WITHOUT executing it (no store / no Qwen).
Run on CPU only:  python scripts/build_nb_12.py
"""
from __future__ import annotations

import os

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# --- 0. bootstrap (global brief, verbatim) --------------------------------
code(
    """
import os, sys
KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", "/path/to/GMS-knowlytix")
sys.path.insert(0, KNOWLYTIX_SRC)
"""
)

md(
    """
# Ch12 — Calibrate, don't guess

A bind threshold picked by eye is a number you cannot defend. This chapter sets
the embedding-binding similarity cut **from a labeled cohort**: paraphrases that
*should* resolve to a Northwind segment are positives, unrelated terms are
negatives, and `calibrate_bind_threshold` sweeps a grid for the operating point
that best separates them. The same discipline applies to the accept/abstain cuts
downstream — see Ch11 (abstention) and App C (the GMS calibration method).

We hold the binder's encoder deterministic here so the chapter runs in CI without
the GPU/MiniLM path. In production you inject the real encoder; the *procedure* is
identical. See Ch7 for embedding binding itself.
"""
)

# --- 1. deterministic encoder grounded in the corpus ----------------------
md(
    """
## A deterministic, corpus-grounded encoder

The real binder downloads a MiniLM encoder on first use. For a reproducible
calibration sweep we inject a fake encoder that maps each Northwind segment and
its paraphrases into the *same* semantic axis. The four axes are the four revenue
segments in `corpus_facts.md`: `cloud platform`, `devices`, `logistics`,
`retail`. Unrelated terms land at the origin and bind to nothing.
"""
)
code(
    '''
import torch

# Axes: [cloud_platform, devices, logistics, retail].
# Each row activates the axis whose segment the term denotes (canonical name or paraphrase).
_AXES = {
    0: ("cloud platform", "saas", "cloud"),       # cloud platform
    1: ("devices", "hardware", "gadget"),         # devices
    2: ("logistics", "shipping", "freight"),      # logistics
    3: ("retail", "stores", "storefront"),        # retail
}


def fake_encode(texts: list[str]) -> torch.Tensor:
    """4-dim semantic encoder over Northwind's revenue segments (no network)."""
    rows = []
    for t in texts:
        tl = t.lower()
        rows.append([1.0 if any(k in tl for k in keys) else 0.0
                     for _, keys in sorted(_AXES.items())])
    return torch.tensor(rows)


# Sanity: a paraphrase and its segment share an axis; an unrelated term is zero.
print("saas    ->", fake_encode(["saas"]).tolist()[0])
print("cloud platform ->", fake_encode(["cloud platform"]).tolist()[0])
print("weather ->", fake_encode(["weather"]).tolist()[0])
'''
)

# --- 2. load the store + build an embedding binder ------------------------
md(
    """
## Load the trained store and build an embedding-mode binder

`calibrate_bind_threshold` tunes a `TripleBinder` in **embedding mode** — fuzzy
exact/substring matches always win first, and the embedding fallback only fires
for terms fuzzy cannot resolve (Ch7). The store ships under
`data/gms_annual_report_store/`; we load it read-only.

> **CI cell.** Loads the trained store. The lead runs this in CI; it does not
> invoke Qwen.
"""
)
code(
    """
import torch
from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.store import GMSExpertStore
from knowlytix.knowledge.rag import TripleBinder

STORE = os.environ.get("GMS_STORE", "data/gms_annual_report_store")
store = GMSExpertStore(DocGMSConfig(store_path=STORE), device=torch.device("cpu"))
assert store.load(), f"no trained store at {STORE}; run scripts/build_store.py first"

binder = TripleBinder(store, mode="embedding", encoder=fake_encode,
                      bind_threshold=0.5, bind_margin=0.05)
print("entities:", sorted(store.adapter.entity_to_idx)[:8], "...")
print("starting bind_threshold:", binder.bind_threshold)
"""
)

# --- 3. labeled cohort -> calibrate ---------------------------------------
md(
    """
## Positives, negatives, and the sweep

Positives are `(term, expected_entity)` paraphrases that *should* bind to a
Northwind segment; negatives are terms that should bind to **nothing**. The grid
sweep picks the threshold that maximizes `(true-positives + true-negatives) /
total` and writes it back onto the binder.
"""
)
code(
    """
from knowlytix.knowledge.rag import calibrate_bind_threshold
from knowlytix.knowledge.rag.query_triples import QueryTriple

# Paraphrases that fuzzy string match misses -> must resolve via embeddings.
positives = [
    ("saas",     "cloud platform"),
    ("hardware", "devices"),
    ("shipping", "logistics"),
    ("stores",   "retail"),
]
# Unrelated terms -> must NOT bind to any segment.
negatives = ["weather", "moon", "guitar"]

best_th, acc = calibrate_bind_threshold(binder, positives, negatives)
print(f"chosen bind_threshold = {best_th:.3f}")
print(f"separation accuracy   = {acc:.3f}")
print("binder now set to     =", binder.bind_threshold)
"""
)

md(
    """
**What the output means.** The chosen threshold is the *operating point*: above
it the binder accepts a paraphrase as a segment match; below it the term is
refused (bind-check abstention, Ch11). `acc == 1.0` means the grid found a cut
that lets every paraphrase through and refuses every unrelated term. The binder
mutates in place — subsequent `bind` calls use the calibrated value.
"""
)

# --- 4. prove the calibrated binder behaves -------------------------------
md(
    """
## The calibrated binder resolves paraphrases and refuses noise
"""
)
code(
    '''
def bind_head(term: str) -> str | None:
    return binder.bind(QueryTriple(term, "has_revenue", "?")).head


for term in ["saas", "shipping", "stores"]:
    print(f"{term:10s} -> {bind_head(term)!r}")     # paraphrase -> canonical segment
for term in ["weather", "moon"]:
    print(f"{term:10s} -> {bind_head(term)!r}")     # noise -> None (refused)
'''
)

# --- 5. exercise: tighten the false-allow ceiling -------------------------
md(
    """
## Exercise — tighten the false-allow ceiling

The default sweep maximizes raw accuracy, treating a false-allow (binding noise)
the same as a false-refuse. A bank-grade system pays more for a false-allow. Here
we re-rank the grid under a **hard ceiling of zero false-allows** and watch the
threshold move up. (Worked solution; the assert below pins it.)
"""
)
code(
    """
def calibrate_under_ceiling(binder, positives, negatives, *,
                            max_false_allow=0,
                            grid=tuple(i / 20 for i in range(1, 20))):
    \"\"\"Pick the lowest threshold (max recall) whose false-allows <= ceiling.\"\"\"
    feasible = []
    for th in grid:
        binder.bind_threshold = th
        false_allow = sum(binder.bind(QueryTriple(t, "_", "?")).head is not None
                          for t in negatives)
        if false_allow <= max_false_allow:
            tp = sum(binder.bind(QueryTriple(t, "_", "?")).head == exp
                     for t, exp in positives)
            feasible.append((th, tp))
    if not feasible:
        raise ValueError("no threshold meets the false-allow ceiling")
    # lowest threshold that still admits the most positives
    best_th = max(feasible, key=lambda x: (x[1], -x[0]))[0]
    binder.bind_threshold = best_th
    return best_th

strict_th = calibrate_under_ceiling(binder, positives, negatives, max_false_allow=0)
print(f"strict (0 false-allow) threshold = {strict_th:.3f}")
print(f"data-accuracy threshold          = {best_th:.3f}")
print("noise still refused:",
      binder.bind(QueryTriple("weather", "has_revenue", "?")).head is None)
"""
)

md(
    """
\\paragraph{Honest limit.} Calibration is only as good as the cohort. With four
segments and three noise terms the grid finds a clean cut, but a threshold tuned
on this cohort does **not** generalize to a relation band it never saw — per-
relation bands need per-relation labels. And a deterministic fake encoder makes
the axes orthogonal by construction; the real MiniLM space is messier, so expect
a softer separation and a non-trivial `bind_margin`. Calibration sets where to
draw the line; it cannot manufacture signal the encoder does not carry.
"""
)

# --- 6. self-check --------------------------------------------------------
md(
    """
## Self-check — the calibrated threshold separates labeled pos/neg
"""
)
code(
    """
# Re-run the data-accuracy calibration so the assert is self-contained.
binder.bind_threshold = 0.5
best_th, acc = calibrate_bind_threshold(binder, positives, negatives)

# Claim of the chapter: the calibrated threshold separates pos from neg.
assert acc == 1.0, f"calibration failed to separate cohort: acc={acc}"
assert 0.0 < best_th <= 1.0, f"threshold out of range: {best_th}"
assert binder.bind_threshold == best_th, "binder not updated in place"

assert binder.bind(QueryTriple("saas", "has_revenue", "?")).head == "cloud platform"
assert binder.bind(QueryTriple("weather", "has_revenue", "?")).head is None

print("OK: calibrated bind_threshold separates the labeled cohort.")
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "notebooks", "14_calibration_a_inline.ipynb")
with open(OUT, "w") as fh:
    nbf.write(nb, fh)
print("wrote", OUT)
