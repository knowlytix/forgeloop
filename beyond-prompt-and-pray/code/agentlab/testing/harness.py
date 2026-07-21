"""GraphDOEHarness: agent-shaped wrapper over the GMSH testing harness.

The underlying GMSH harness exposes ~40 configuration knobs across a dozen
modules. This adapter narrows the surface to the five methods the chapter
teaches: ingest, generate, run, analyze, report. The pipeline maps to the
five-stage flow:

    ingest (markdown -> store)
    generate (factor design + question generators -> rows)
    run (rows + agent -> trajectory + binary correct + geometric scores)
    analyze (rows -> factor attribution table)
    report (rows -> dashboard html)

Everything beyond these five methods stays in `knowlytix.harness.graphdoe.*`
and the GMS testing monograph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TestResult:
    """Per-row outcomes from one harness run.

    Wraps the underlying DOEBenchmarkResult to flatten the fields the
    chapter actually inspects.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    n_runs: int = 0
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorAttribution:
    """Output of `GraphDOEHarness.analyze`.

    The four tables the chapter prints: logistic deviance per factor with
    BH-corrected p-values, joint-model fit, failure ranking, and pairwise
    interactions.
    """

    logistic_table: list[dict[str, Any]] = field(default_factory=list)
    joint_model: dict[str, Any] = field(default_factory=dict)
    failure_table: list[dict[str, Any]] = field(default_factory=list)
    interactions: list[dict[str, Any]] = field(default_factory=list)

    def top_drivers(self, k: int = 3) -> list[dict[str, Any]]:
        """Return the top-k significant factors ranked by adjusted p-value."""
        sig = [r for r in self.logistic_table if r.get("significant_adj", False)]
        sig.sort(key=lambda r: r.get("p_value_adj", 1.0))
        return sig[:k]

    def top_failures(self, k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k (factor, level) combinations by failure rate."""
        ranked = sorted(self.failure_table, key=lambda r: -r.get("failure_rate", 0.0))
        return ranked[:k]


class GraphDOEHarness:
    """Agent-shaped wrapper around the GMSH DoE testing harness.

    The constructor builds a `DOEHarnessConfig` and a `DOEGMSBenchmark`
    under the hood. The five public methods map to the five stages of the
    testing pipeline (ingest, generate, run, analyze, report). This is the
    knowledge-QA-shaped harness; for the capstone agent (TaskSpec -> Trajectory)
    use `CapstoneTestHarness`.

    Parameters
    ----------
    document_path : str
        Path to the markdown ground-truth document the GMS store is built from.
    factor_group : str
        Catalog name of the factor set to vary. Default "quick_screen" (three
        factors: clarity, entity_aliasing, reasoning_cue).
    n_runs : int
        Number of DoE rows. 32 for screening, 128 for joint-model designs.
    method : str
        Design generator. "sobol", "lhs", or "grid" ship in graphdoe; the
        GMSH harness adds "sobol+refine" for production runs.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        document_path: str,
        factor_group: str = "quick_screen",
        n_runs: int = 32,
        method: str = "sobol",
        seed: int = 42,
        max_per_category: int = 5,
        target_type: str = "llm",
        target_model: str = "claude-opus-4-7",
        context_mode: str = "full",
    ) -> None:
        try:
            from knowlytix.harness.testing import DOEGMSBenchmark, DOEHarnessConfig
        except ImportError as e:
            raise ImportError(
                "knowlytix harness required for GraphDOEHarness. Install the "
                "knowlytix wheels (knowlytix_core, knowlytix_harness)."
            ) from e
        self._config = DOEHarnessConfig(
            markdown_path=document_path,
            factor_group=factor_group,
            n_runs=n_runs,
            method=method,
            seed=seed,
            max_per_category=max_per_category,
            target_type=target_type,
            target_model=target_model,
            context_mode=context_mode,
        )
        self._bench = DOEGMSBenchmark(self._config)
        self._store = None
        self._questions = None

    @property
    def benchmark(self):
        """Access the underlying DOEGMSBenchmark for advanced use."""
        return self._bench

    def ingest(self) -> None:
        """Stage 1: parse the document into a GMS store. Idempotent."""
        if self._store is None:
            self._store = self._bench.ingest()

    def generate(self, generators: list[str] | None = None) -> list[Any]:
        """Stage 2: produce one row per DoE-expanded question."""
        if self._store is None:
            self.ingest()
        self._questions = self._bench.generate_questions()
        return self._questions

    def run(self, agent: Callable[[str, str], str] | None = None) -> TestResult:
        """Stage 3: execute each row against an agent and collect outcomes.

        `agent` is a `(question, context) -> answer` callable. If omitted the
        harness uses the default LLM evaluator from the config.
        """
        if self._questions is None:
            self.generate()
        evaluator = self._wrap_agent(agent) if agent is not None else None
        raw = self._bench.run(evaluator=evaluator)
        return TestResult(
            rows=self._rows_from_raw(raw),
            n_runs=getattr(self._config, "n_runs", 0),
            summary={"target_model": self._config.target_model},
        )

    def analyze(self, result: TestResult) -> FactorAttribution:
        """Stage 4: factor attribution via logistic regression + analysis of deviance."""
        try:
            from knowlytix.harness.graphdoe import DOEAnalyzer
        except ImportError as e:
            raise ImportError(
                "knowlytix harness required for the analysis stage. Install the "
                "knowlytix wheels (knowlytix_core, knowlytix_harness)."
            ) from e
        import pandas as pd

        df = pd.DataFrame(result.rows)
        analyzer = DOEAnalyzer.from_dataframe(df, factors=self._factor_columns(df))
        logistic = analyzer.run_logistic(metric="correct")
        try:
            logistic = analyzer.apply_fdr_correction(logistic, alpha=0.05)
        except AttributeError:
            pass
        try:
            joint = analyzer.run_joint_model(metric="correct")
        except AttributeError:
            joint = {}
        try:
            failures = analyzer.failure_analysis()
        except AttributeError:
            failures = None
        try:
            interactions = analyzer.run_interaction_logistic()
        except AttributeError:
            interactions = None
        return FactorAttribution(
            logistic_table=logistic.to_dict("records") if hasattr(logistic, "to_dict") else [],
            joint_model=dict(joint) if joint else {},
            failure_table=failures.to_dict("records") if failures is not None and hasattr(failures, "to_dict") else [],
            interactions=interactions.to_dict("records") if interactions is not None and hasattr(interactions, "to_dict") else [],
        )

    def report(self, result: TestResult, output_path: str) -> None:
        """Stage 5: write the GMSH HTML dashboard for the run."""
        self._bench.generate_report(result, output_path)

    # --- private helpers ----------------------------------------------------

    @staticmethod
    def _wrap_agent(agent: Callable[[str, str], str]):
        class _AgentEvaluator:
            def __call__(self_inner, question: str, context: str) -> str:
                return agent(question, context)
        return _AgentEvaluator()

    @staticmethod
    def _rows_from_raw(raw) -> list[dict[str, Any]]:
        if hasattr(raw, "to_dataframe"):
            return raw.to_dataframe().to_dict("records")
        if hasattr(raw, "rows"):
            return list(raw.rows)
        return []

    @staticmethod
    def _factor_columns(df) -> list[str]:
        ignored = {"question", "context", "answer", "correct", "graph_answer", "category"}
        return [c for c in df.columns if c not in ignored and not c.startswith("_geo_")]
