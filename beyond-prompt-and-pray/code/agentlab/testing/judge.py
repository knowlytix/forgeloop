"""GeometricJudge: agent-shaped wrapper over GMSJudge + the hallucination tiers.

The underlying GMSJudge returns a `GeometricVerdict` carrying four continuous
signals (geodesic, tension, holonomy, exact) plus calibrated probabilities and
evidence. This adapter narrows to the surface the chapter teaches: a single
`judge(answer, ground_truth, metadata)` call returning a small dataclass with
the four scores and the four-band hallucination label.

The GMSH harness is a hard dependency. There is no offline fallback;
the chapter teaches real GMS-backed judging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

HallucinationLabel = Literal["grounded", "embellishment", "distortion", "fabrication"]


@dataclass(frozen=True)
class JudgeVerdict:
    """Output of `GeometricJudge.judge`.

    `passed` is the binary verdict; `confidence` is the aggregated continuous
    confidence from the underlying judge. The four geometric signals are
    surfaced individually so the caller can route by margin.
    """

    passed: bool
    confidence: float
    geodesic: float
    tension: float
    holonomy: float
    label: HallucinationLabel
    detail: str = ""


def _label_from_geodesic(d: float) -> HallucinationLabel:
    """Map mean geodesic distance to the four-band hallucination label.

    Bands match the harness defaults: grounded < 0.5 <= embellishment < 1.0
    <= distortion < 1.5 <= fabrication.
    """
    if d < 0.5:
        return "grounded"
    if d < 1.0:
        return "embellishment"
    if d < 1.5:
        return "distortion"
    return "fabrication"


class GeometricJudge:
    """Wraps `knowlytix.harness.testing.judge.GMSJudge` with a simpler return shape.

    Parameters
    ----------
    store : GMSExpertStore
        A trained GMS store (typically from `docgms.ingest.ingest_document`
        or `store.load()`).
    confidence_threshold : float
        Threshold for binary `passed`. Values >= this threshold are passing.
        Default 0.6.
    """

    def __init__(
        self,
        store,
        confidence_threshold: float = 0.6,
    ) -> None:
        try:
            from knowlytix.harness.testing.judge import GMSJudge
        except ImportError as e:
            raise ImportError(
                "knowlytix harness required for GeometricJudge. Install the "
                "knowlytix wheels (knowlytix_core, knowlytix_harness)."
            ) from e
        self._store = store
        self._threshold = float(confidence_threshold)
        self._impl = GMSJudge(store)

    def judge(
        self,
        answer: str,
        ground_truth: str,
        metadata: dict[str, Any] | None = None,
    ) -> JudgeVerdict:
        """Judge a single answer against ground truth."""
        meta = dict(metadata or {})
        verdict = self._impl.judge(
            llm_answer=answer,
            ground_truth=ground_truth,
            exact_correct=meta.get("exact_correct", False),
            exact_detail=meta.get("exact_detail", ""),
            question_metadata=meta,
        )
        geo_conf = float(getattr(verdict, "geodesic_confidence", 0.0))
        ten = float(getattr(verdict, "tension_score", 0.0))
        hol = float(getattr(verdict, "holonomy_score", 0.0))
        conf = float(getattr(verdict, "overall_confidence", geo_conf))
        # geodesic_confidence = exp(-mean_geodesic), so invert
        mean_geo = -math.log(max(geo_conf, 1e-6))
        return JudgeVerdict(
            passed=conf >= self._threshold,
            confidence=conf,
            geodesic=mean_geo,
            tension=ten,
            holonomy=hol,
            label=_label_from_geodesic(mean_geo),
            detail=str(getattr(verdict, "detail", "")),
        )
