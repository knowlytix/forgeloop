"""Tests for GeometricJudge against a real GMS store + GMSH harness."""

import pytest

from agentlab.testing import GeometricJudge, JudgeVerdict


def test_judge_constructs(gms_store):
    """The judge instantiates against a real loaded store."""
    judge = GeometricJudge(gms_store, confidence_threshold=0.6)
    assert judge._threshold == 0.6


def test_judge_returns_verdict_shape(gms_store):
    """A real judge call returns a structured verdict with the four signals."""
    judge = GeometricJudge(gms_store)
    v = judge.judge(
        answer="MDL-001 is classified as Medium risk.",
        ground_truth="MDL-001 has risk rating Medium.",
    )
    assert isinstance(v, JudgeVerdict)
    assert isinstance(v.passed, bool)
    assert isinstance(v.confidence, float)
    assert isinstance(v.geodesic, float)
    assert v.label in ("grounded", "embellishment", "distortion", "fabrication")


def test_verdict_is_immutable(gms_store):
    judge = GeometricJudge(gms_store)
    v = judge.judge(answer="x", ground_truth="x")
    with pytest.raises(Exception):
        v.passed = False


def test_threshold_routes_passed(gms_store):
    """A higher confidence threshold makes more verdicts fail."""
    strict = GeometricJudge(gms_store, confidence_threshold=0.99)
    lenient = GeometricJudge(gms_store, confidence_threshold=0.0)
    v_strict = strict.judge(answer="something", ground_truth="something else")
    v_lenient = lenient.judge(answer="something", ground_truth="something else")
    # Lenient threshold should never be stricter than strict.
    assert (not v_strict.passed) or v_lenient.passed
