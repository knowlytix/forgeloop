"""PolicyEngine: a named collection of policies, convertible to a PolicyGate."""

from __future__ import annotations

from forgeloop.agents.governance.gates import PolicyCheck, PolicyGate


class PolicyEngine:
    """Mutable collection of policy checks convertible to a PolicyGate."""

    def __init__(self, policies: list[PolicyCheck] | None = None) -> None:
        self._policies: list[PolicyCheck] = list(policies or [])

    def add(self, policy: PolicyCheck) -> "PolicyEngine":
        """Append a policy check and return self for chaining.

        Args:
            policy: The policy check to add.

        Returns:
            This engine, to allow chained calls.
        """
        self._policies.append(policy)
        return self

    def policies(self) -> list[PolicyCheck]:
        """Return a copy of the registered policy checks."""
        return list(self._policies)

    def as_gate(self) -> PolicyGate:
        """Return a PolicyGate wrapping the registered policy checks."""
        return PolicyGate(policies=self._policies)
