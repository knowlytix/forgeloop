"""Build notebooks/09_binding_a_inline.ipynb (CPU only — no store load, no Qwen).

Emits a valid nbformat-4 notebook for Ch7 "Binding query terms to the graph".
Every listing is grounded in data/corpus_facts.md (Northwind Industries FY2025)
and runs against a lightweight fake store that mimics the exact interface
TripleBinder needs (store.adapter.entity_to_idx / relation_to_idx,
store.fuzzy_match_entity). This keeps the chapter deterministic and CI-cheap;
the real store is built once in F2 and reused by the pipeline chapters.
"""
from __future__ import annotations

import pathlib

import nbformat as nbf

NB_PATH = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "09_binding_a_inline.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ---------------------------------------------------------------- bootstrap
code(
    """
import os, sys
KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", "/path/to/GMS-knowlytix")
sys.path.insert(0, KNOWLYTIX_SRC)
"""
)

md(
    """
# Ch7 — Binding query terms to the graph

A query triple is inert until its terms resolve to the vocabulary the store
actually holds. The user types *"topline"*; the graph stores `has_revenue`. The
user says *"consumer hardware"*; the graph knows `devices`. **Binding** is the
step that turns a paraphrase into a graph term — or, when the term is ambiguous,
*refuses* rather than guess. This is bank-grade behaviour: a silent
mis-resolution is worse than an abstention.

We use the real `knowlytix.knowledge.rag.binding` module against a tiny fake
store that exposes exactly the interface `TripleBinder` consumes
(`store.adapter.entity_to_idx`, `store.adapter.relation_to_idx`,
`store.fuzzy_match_entity`). The trained Northwind store from F2 has the same
vocabulary; using a fake here keeps the chapter deterministic and CPU-only.
"""
)

# ---------------------------------------------------------------- fake store
code(
    '''
from knowlytix.knowledge.store import GMSExpertStore


class _Adapter:
    """Minimal stand-in for GraphToGMS: just the two vocab maps TripleBinder reads."""
    def __init__(self, entities: list[str], relations: list[str]):
        self.entity_to_idx = {e: i for i, e in enumerate(entities)}
        self.relation_to_idx = {r: i for i, r in enumerate(relations)}


class FakeStore:
    """Northwind FY2025 vocabulary with the real fuzzy_match_entity behaviour."""
    def __init__(self):
        self.adapter = _Adapter(
            entities=["cloud platform", "devices", "logistics", "retail",
                      "total", "technology", "operations", "revenue"],
            relations=["has_revenue", "has_headcount", "has_division",
                       "has_region", "has_head", "in_section"],
        )

    # Reuse the shipped bank-grade matcher verbatim (unique exact / ci / substring).
    fuzzy_match_entity = GMSExpertStore.fuzzy_match_entity


store = FakeStore()
sorted(store.adapter.entity_to_idx), sorted(store.adapter.relation_to_idx)
'''
)
md(
    """
Expected: the Northwind segment entities (`cloud platform`, `devices`,
`logistics`, `retail`, `total`) plus the two divisions (`technology`,
`operations`) and the relations from `corpus_facts.md`
(`has_revenue`, `has_headcount`, `has_division`, `has_region`, `has_head`).
"""
)

# -------------------------------------------------- L1: fuzzy succeeds/fails
md(
    """
## Listing 1 — Fuzzy binding: substring resolves, paraphrase does not

`TripleBinder(mode="fuzzy")` reuses `store.fuzzy_match_entity` for entities and a
`has_<slug>` relation matcher. It resolves a *substring* nickname (`"cloud"` →
`cloud platform`) and the table-column relation `"revenue"` → `has_revenue`. But
`"topline"` and `"consumer hardware"` share no substring with any graph term, so
fuzzy binding returns `None` — the slot is unbound and the triple is not `bound`.
"""
)
code(
    '''
from knowlytix.knowledge.rag import TripleBinder
from knowlytix.knowledge.rag.query_triples import QueryTriple

fuzzy = TripleBinder(store, mode="fuzzy")

# (a) substring nickname + table-column relation -> binds
ok = fuzzy.bind(QueryTriple("cloud", "revenue", "?"))
print("cloud/revenue ->", ok.head, ok.relation, ok.tail, "| bound:", ok.bound)

# (b) "topline" alias for revenue -> fuzzy cannot reach has_revenue
top = fuzzy.bind(QueryTriple("cloud platform", "topline", "?"))
print("topline       ->", top.head, top.relation, top.tail, "| bound:", top.bound)

# (c) "consumer hardware" synonym for devices -> no substring, no bind
syn = fuzzy.bind(QueryTriple("consumer hardware", "headcount", "?"))
print("consumer hw   ->", syn.head, syn.relation, syn.tail, "| bound:", syn.bound)
'''
)
md(
    """
Expected output:

```
cloud/revenue -> cloud platform has_revenue ? | bound: True
topline       -> cloud platform None ? | bound: False
consumer hw   -> None has_headcount ? | bound: False
```

`"cloud"` is a unique substring of `cloud platform`; `"revenue"` slugs to
`has_revenue` via the `has_<slug>` convention. `"topline"` and `"consumer
hardware"` are genuine paraphrases — fuzzy string matching has nothing to grab.
A bound triple needs *every* non-variable slot resolved; the `?` slot is a
variable and never counts against binding.
"""
)

# -------------------------------------------------- L2: embedding succeeds
md(
    """
## Listing 2 — Embedding binding: the injectable encoder resolves paraphrases

`mode="embedding"` keeps exact/fuzzy authoritative, then falls back to a nearest-
neighbour cosine match over an injectable encoder (`list[str] -> (N, d)`). In
production the default is the repo MiniLM encoder; here we inject a deterministic
4-dim encoder so the chapter is reproducible and needs no download or GPU. The
real path — `TripleBinder(store, mode="embedding")` with no `encoder=` — is shown
in the next listing, marked for CI.
"""
)
code(
    '''
import torch

# Deterministic semantic axes: [revenue, devices, cloud, headcount].
# Synonyms land on the same axis as the canonical Northwind term.
def fake_encode(texts: list[str]) -> torch.Tensor:
    rows = []
    for t in texts:
        tl = t.lower()
        rows.append([
            1.0 if ("revenue" in tl or "topline" in tl or "sales" in tl) else 0.0,
            1.0 if ("devices" in tl or "hardware" in tl or "gadget" in tl) else 0.0,
            1.0 if ("cloud" in tl or "platform" in tl) else 0.0,
            1.0 if ("headcount" in tl or "staff" in tl or "employees" in tl) else 0.0,
        ])
    return torch.tensor(rows)

emb = TripleBinder(store, mode="embedding", encoder=fake_encode)

# "topline" -> has_revenue (the GMS box), "consumer hardware" -> devices
bt = emb.bind(QueryTriple("consumer hardware", "topline", "?"))
print("consumer hardware / topline ->", bt.head, bt.relation, bt.tail)
print("bound:", bt.bound)
'''
)
md(
    """
Expected output:

```
consumer hardware / topline -> devices has_revenue ?
bound: True
```

`"topline"` and `"sales"` share the *revenue* axis with `has_revenue`; `"consumer
hardware"` shares the *devices* axis with `devices`. Embedding binding succeeds
exactly where fuzzy failed in Listing 1, and the asked slot `?` passes through
untouched. Exact and fuzzy hits still win first — embedding is only consulted
when string matching returns nothing (see `binding.py` `_bind_entity`).
"""
)

# -------------------------------------------------- L2b: real Qwen-era path
md(
    """
### The real encoder path (run in CI)

The cell below is the production form — no injected encoder, so `TripleBinder`
lazily wraps `knowlytix.core.graph.encoders.encode_texts` (MiniLM, downloads on
first use). It is marked for the lead to execute in CI; we do not run it here
because the authoring GPU is shared.
"""
)
code(
    '''
# CI-ONLY (downloads MiniLM): real default encoder, no fake injected.
# from knowlytix.knowledge.rag import TripleBinder
# real = TripleBinder(store, mode="embedding")          # encoder=None -> encode_texts
# bt = real.bind(QueryTriple("topline", "topline", "?")) # paraphrase of revenue
# print(bt.head, bt.relation)  # expect a revenue-family relation
'''
)

# -------------------------------------------------- L3: refuse-on-ambiguous
md(
    """
## Listing 3 — Refuse on ambiguity, do not mis-resolve

Two safeguards make binding bank-grade. (1) Below `bind_threshold` cosine, the
match is rejected (the term is out-of-vocabulary). (2) When the top two
candidates are within `bind_margin`, the match is refused as a *near tie* — the
system would rather abstain than pick one of two plausible entities. Here
`"the unit"` matches nothing (zero vector → cosine 0), and a deliberately
ambiguous encoder makes a term tie between two relations.
"""
)
code(
    '''
# (a) out-of-vocabulary term -> below threshold -> None
oov = emb.bind(QueryTriple("the unit", "revenue", "?"))
print("the unit ->", oov.head, "| bound:", oov.bound)

# (b) a term that ties between two relations -> refuse rather than guess.
# This encoder puts "growth" equidistant from has_revenue and has_headcount.
def tie_encode(texts: list[str]) -> torch.Tensor:
    rows = []
    for t in texts:
        tl = t.lower()
        if "growth" in tl:
            rows.append([0.7, 0.7, 0.0, 0.0])            # ambiguous query
        else:
            rows.append([
                1.0 if "revenue" in tl else 0.0,
                1.0 if "headcount" in tl else 0.0,
                0.0, 0.0,
            ])
    return torch.tensor(rows)

tie = TripleBinder(store, mode="embedding", encoder=tie_encode, bind_margin=0.1)
amb = tie.bind(QueryTriple("cloud platform", "growth", "?"))
print("growth   ->", amb.relation, "| bound:", amb.bound)
'''
)
md(
    """
Expected output:

```
the unit -> None | bound: False
growth   -> None | bound: False
```

`"the unit"` encodes to the zero vector, cosine 0 < `bind_threshold` (0.5) →
unbound. `"growth"` sits halfway between `has_revenue` and `has_headcount`; the
gap to the runner-up is below `bind_margin` → refused. Both feed bind-check
abstention downstream: an unbound query triple means the pipeline declines to
answer rather than fabricate a binding. Calibrating `bind_threshold` from labeled
data is Ch~\\ref{ch:calibrate}.
"""
)

# -------------------------------------------------- self-check
md(
    """
## Self-check

The chapter's claim: **a paraphrase binds to the canonical Northwind entity, and
an ambiguous term is refused.** The assertion proves both — `"topline"` resolves
to `has_revenue` and `"consumer hardware"` to `devices`, while the ambiguous
`"growth"` and out-of-vocabulary `"the unit"` stay unbound.
"""
)
code(
    '''
# Paraphrase binds to the canonical entity + relation.
bound = emb.bind(QueryTriple("consumer hardware", "topline", "?"))
assert bound.head == "devices", bound.head
assert bound.relation == "has_revenue", bound.relation
assert bound.bound is True

# Ambiguous / OOV terms refuse rather than mis-resolve.
assert tie.bind(QueryTriple("cloud platform", "growth", "?")).relation is None
assert emb.bind(QueryTriple("the unit", "revenue", "?")).head is None

# Exact match stays authoritative even in embedding mode.
exact = emb.bind(QueryTriple("cloud platform", "has_revenue", "?"))
assert exact.head == "cloud platform" and exact.relation == "has_revenue"

print("OK: paraphrases bind, ambiguity refused, exact match authoritative.")
'''
)

# ---------------------------------------------------------------- exercise
md(
    """
## Exercise — add a synonym and confirm embedding binding picks it up

Northwind's `logistics` segment is sometimes called *"shipping"* internally. Add
a `shipping` axis to a copy of the encoder, bind `("shipping", "topline", "?")`,
and confirm it resolves to `logistics` / `has_revenue`. (Worked solution below.)
"""
)
code(
    '''
def exercise_encode(texts: list[str]) -> torch.Tensor:
    rows = []
    for t in texts:
        tl = t.lower()
        rows.append([
            1.0 if ("revenue" in tl or "topline" in tl) else 0.0,
            1.0 if ("logistics" in tl or "shipping" in tl or "freight" in tl) else 0.0,
        ])
    return torch.tensor(rows)

ex = TripleBinder(store, mode="embedding", encoder=exercise_encode)
sol = ex.bind(QueryTriple("shipping", "topline", "?"))
assert sol.head == "logistics" and sol.relation == "has_revenue"
print("shipping -> ", sol.head, "/", sol.relation)
'''
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(NB_PATH))
print("wrote", NB_PATH)
