"""Base sources — where the base questions/answers come from.

Three sources, all yielding QAItem:

  CatalogBaseSource  mine a GMS store with knowlytix.benchmark generators
                     (ground truth is graph-derived and provably correct)
  SeedCaseSource     user-labeled cases carrying per-component ground truth
                     (the 20 complaint cases of the capstone)
  UserBaseSource     user supplies (query, answer) pairs directly — NO GMS,
                     no generation, no GMS expansion of queries

The enrichment layer (compose.py) treats all three identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .resolve import ResolvedSuite


@dataclass
class QAItem:
    """One base (query, answer) with optional per-component ground truth."""
    qid: str
    query: str
    answer: Any
    answer_type: str = "str"
    base: str | None = None                 # base-category name, or None (user)
    components: dict[str, Any] = field(default_factory=dict)  # per-tool truth
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BaseSource(Protocol):
    def items(self, suite: ResolvedSuite | None = None) -> list[QAItem]: ...


# ---------------------------------------------------------------------------
class UserBaseSource:
    """Bring-your-own (query, answer) pairs. No GMS is used.

    pairs: list of dicts with keys: query, answer, [answer_type], [metadata].
    The user's answer IS the ground truth; queries are NOT expanded by GMS.
    """

    def __init__(self, pairs: list[dict[str, Any]]):
        self._pairs = pairs

    def items(self, suite: ResolvedSuite | None = None) -> list[QAItem]:
        out = []
        for i, p in enumerate(self._pairs):
            out.append(QAItem(
                qid=p.get("qid", f"user-{i:04d}"),
                query=p["query"],
                answer=p["answer"],
                answer_type=p.get("answer_type", "str"),
                base=None,
                metadata=dict(p.get("metadata", {})),
            ))
        return out


# ---------------------------------------------------------------------------
class SeedCaseSource:
    """Hand-labeled cases carrying per-component ground truth.

    Each case is a dict; `query_key` selects the input text and `component_keys`
    name the per-tool ground-truth fields (classification, product, issue,
    policy, regulation, escalate, ...). `answer_key` selects the primary label.
    """

    def __init__(
        self,
        cases: list[dict[str, Any]],
        query_key: str = "message",
        answer_key: str = "expected_classification",
        id_key: str = "id",
        component_keys: tuple[str, ...] = (
            "expected_classification", "expected_product", "expected_issue",
            "expected_policy", "expected_regulation",
            "expected_escalation", "expected_escalate",
        ),
    ):
        self._cases = cases
        self._q, self._a, self._id = query_key, answer_key, id_key
        self._components = component_keys

    def items(self, suite: ResolvedSuite | None = None) -> list[QAItem]:
        out = []
        for i, c in enumerate(self._cases):
            comps = {k: c[k] for k in self._components if k in c}
            out.append(QAItem(
                qid=str(c.get(self._id, f"case-{i:04d}")),
                query=c[self._q],
                answer=c.get(self._a),
                answer_type="decision",
                base="seed_case",
                components=comps,
                metadata={k: v for k, v in c.items()
                          if k not in (self._q, self._id) and k not in comps},
            ))
        return out


# ---------------------------------------------------------------------------
class CatalogBaseSource:
    """Mine a GMS store using the selected base categories' generators.

    Requires a knowlytix GMSExpertStore exposing `doc_graph` and the
    knowlytix.benchmark generators. Ground truth comes from each
    GeneratedQuestion's `ground_truth` / `graph_answer_fn`.
    """

    def __init__(self, store: Any, max_per_category: int = 10, seed: int = 42):
        self._store = store
        self._max = max_per_category
        self._seed = seed

    def items(self, suite: ResolvedSuite | None = None) -> list[QAItem]:
        if suite is None:
            raise ValueError("CatalogBaseSource.items needs a resolved suite")
        gens = self._generators_for(suite)
        graph = getattr(self._store, "doc_graph", self._store)
        out: list[QAItem] = []
        for base_name, gen in gens:
            candidates = gen.generate(graph)
            if hasattr(gen, "sample"):
                candidates = gen.sample(candidates, self._max, seed=self._seed)
            else:
                candidates = candidates[: self._max]
            for j, q in enumerate(candidates):
                out.append(QAItem(
                    qid=getattr(q, "qid", f"{base_name}-{j:03d}"),
                    query=getattr(q, "natural_language", ""),
                    answer=getattr(q, "ground_truth", None),
                    answer_type=getattr(q, "answer_type", "str"),
                    base=base_name,
                    metadata={"category": getattr(q, "category", base_name),
                              **dict(getattr(q, "metadata", {}) or {})},
                ))
        return out

    def _generators_for(self, suite: ResolvedSuite):
        """Resolve base specs -> instantiated knowlytix generator objects."""
        import importlib

        gen_mod = importlib.import_module("knowlytix.benchmark.generators")
        resolved = []
        for b in suite.bases:
            if not b.generator:
                continue  # status: build / partial — no generator yet
            cls = getattr(gen_mod, b.generator, None)
            if cls is None:
                continue
            resolved.append((b.name, cls()))
        return resolved
