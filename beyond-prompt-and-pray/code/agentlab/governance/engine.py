"""PolicyEngine: a named collection of policies, convertible to a PolicyGate."""

from __future__ import annotations

from agentlab.governance.gates import PolicyCheck, PolicyGate


class PolicyEngine:
    def __init__(self, policies: list[PolicyCheck] | None = None) -> None:
        self._policies: list[PolicyCheck] = list(policies or [])

    def add(self, policy: PolicyCheck) -> PolicyEngine:
        self._policies.append(policy)
        return self

    def policies(self) -> list[PolicyCheck]:
        return list(self._policies)

    def as_gate(self) -> PolicyGate:
        return PolicyGate(policies=self._policies)
