"""Banking-specific PolicyCheck functions.

These compose with the general policies (pii, prompt_injection,
prohibited_advice) defined in agentlab.governance.policies.
"""

from __future__ import annotations

import json
import re

from agentlab.tools.executor import GateDecision, GateResult

_WAIVER_PATTERNS = [
    r"\bwe\s+will\s+waive\b",
    r"\bwe\s+waive\b",
    r"\bfee\s+(?:will\s+be|has\s+been)\s+waived\b",
    r"\bwe\s+(?:are|will\s+be)\s+refunding\b",
]


def fee_waiver_policy(action, state) -> GateResult:
    """Forbid the agent from unilaterally promising a fee waiver in draft_response.

    Fee reversals require manager approval per fee_reversal.txt. The agent must
    escalate to a representative instead of committing the bank.
    """
    if action.tool_name != "draft_response":
        return GateResult(GateDecision.ALLOW, "fee_waiver_policy")
    body = json.dumps(action.arguments, default=str).lower()
    for pat in _WAIVER_PATTERNS:
        if re.search(pat, body):
            return GateResult(
                GateDecision.ESCALATE,
                "fee_waiver_policy",
                f"draft promises a fee waiver (matched {pat!r}) — must escalate",
            )
    return GateResult(GateDecision.ALLOW, "fee_waiver_policy")
