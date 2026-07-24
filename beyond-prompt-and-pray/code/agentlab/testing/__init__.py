"""Testing adapters: DoE-based testing for agents. See the testing chapter.

Thin adapters over the knowlytix harness (`knowlytix.harness.testing`,
`knowlytix.harness.graphdoe`). Two harnesses:

  * `GraphDOEHarness` --- the knowledge-QA-shaped harness (ingest a document,
    generate GMS-primitive questions, run a `(question, context) -> str`
    evaluator, attribute failures).
  * `CapstoneTestHarness` --- a `TaskSpec -> Trajectory` test stand for the
    Chapter 15 complaint agent: a factor-balanced complaint suite, the real
    governed agent, groundedness-as-distance judging and factor attribution.
"""

from agentlab.testing.capstone_harness import (
    CapstoneTestHarness,
    CapstoneTestResult,
    FaultInjectionResult,
    RagTestResult,
    SubstrateTestResult,
)
from agentlab.testing.harness import (
    FactorAttribution,
    GraphDOEHarness,
    TestResult,
)
from agentlab.testing.judge import GeometricJudge, JudgeVerdict

__all__ = [
    "CapstoneTestHarness",
    "CapstoneTestResult",
    "FactorAttribution",
    "FaultInjectionResult",
    "GeometricJudge",
    "GraphDOEHarness",
    "JudgeVerdict",
    "RagTestResult",
    "SubstrateTestResult",
    "TestResult",
]
