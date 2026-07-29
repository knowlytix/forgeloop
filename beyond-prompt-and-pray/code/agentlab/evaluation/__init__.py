"""Trajectory evaluation: metrics, groundedness, failure modes, DoE.

See Chapters 10 and 11.
"""

from agentlab.evaluation.doe import balanced_design, coverage_report
from agentlab.evaluation.failure_modes import (
    ALL_INJECTORS,
    FailureInjection,
    FailureMode,
    inject,
)
from agentlab.evaluation.groundedness import (
    Claim,
    ClaimVerdict,
    GroundednessResult,
    check_groundedness,
    coverage,
    extract_claims,
    groundedness_report,
)
from agentlab.evaluation.metrics import (
    average_observation_size,
    escalated,
    failed,
    finished_cleanly,
    step_count,
    summarize,
    task_success,
    tool_call_count,
    tool_failure_count,
)
from agentlab.evaluation.test_cases import DEFAULT_FACTORS, TestCase, generate_test_cases
from agentlab.evaluation.trajectory import Trajectory, collect

__all__ = [
    "ALL_INJECTORS",
    "Claim",
    "ClaimVerdict",
    "DEFAULT_FACTORS",
    "FailureInjection",
    "FailureMode",
    "GroundednessResult",
    "TestCase",
    "Trajectory",
    "average_observation_size",
    "balanced_design",
    "check_groundedness",
    "collect",
    "coverage",
    "coverage_report",
    "escalated",
    "extract_claims",
    "failed",
    "finished_cleanly",
    "generate_test_cases",
    "groundedness_report",
    "inject",
    "step_count",
    "summarize",
    "task_success",
    "tool_call_count",
    "tool_failure_count",
]
