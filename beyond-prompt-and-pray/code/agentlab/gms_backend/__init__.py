"""GMS backend adapters. See Chapter 16.

Thin adapters that let an agentlab agent use the GMS substrate as a memory
backend and a plausibility gate. Everything imported here delegates to the
`gms` and `governed_agent` libraries; we wrap, we do not reimplement.

The chapter teaches *what* GMS does and *how* an agent uses it. The
mathematical substrate lives behind these adapters in the GMS library.
"""

from agentlab.gms_backend.gates import GMSPlausibilityGate
from agentlab.gms_backend.memory import GMSMemory
from agentlab.gms_backend.plan_gate import GMSPlanGate, PlanVerdict

__all__ = ["GMSMemory", "GMSPlanGate", "GMSPlausibilityGate", "PlanVerdict"]
