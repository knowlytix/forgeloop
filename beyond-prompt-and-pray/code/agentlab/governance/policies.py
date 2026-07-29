"""Policy library: PII, prompt injection, prohibited advice.

These are regex-based baselines. Production systems use higher-fidelity
detectors (LM classifiers, NER models). The pattern matters more than
the implementation: a policy is a callable taking (action, state) and
returning a GateResult.
"""

from __future__ import annotations

import json
import re

from agentlab.tools.executor import GateDecision, GateResult

_PII_PATTERNS: dict[str, str] = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "email": r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
}

_PROMPT_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(?:\w+\s+)*(?:prior|previous|above|earlier)\s+instructions",
    r"disregard\s+(?:the\s+)?(?:above|prior)",
    r"\bsystem\s*:\s*",
    r"new\s+instructions:",
    r"forget\s+everything",
]

_PROHIBITED_ADVICE_PATTERNS: list[str] = [
    r"\binvest\s+all\b",
    r"\bguaranteed\s+returns?\b",
    r"\btax\s+evasion\b",
    r"\bavoid\s+the\s+irs\b",
]


def contains_pii(text: str) -> list[str]:
    """Return the kinds of PII whose regex patterns match the text.

    Args:
        text: The text to scan.

    Returns:
        The matched PII kinds (for example ``ssn``, ``email``), empty if none.
    """
    found: list[str] = []
    for kind, pattern in _PII_PATTERNS.items():
        if re.search(pattern, text):
            found.append(kind)
    return found


def pii_policy(action, state) -> GateResult:
    """Escalate when the action arguments contain detectable PII.

    Args:
        action: The proposed action whose arguments are scanned.
        state: The current state (unused; present for the policy interface).

    Returns:
        An ESCALATE result listing the PII kinds found, else ALLOW.
    """
    args_text = json.dumps(action.arguments, default=str)
    found = contains_pii(args_text)
    if found:
        return GateResult(GateDecision.ESCALATE, "pii_policy", f"detected PII: {','.join(found)}")
    return GateResult(GateDecision.ALLOW, "pii_policy")


def prompt_injection_policy(action, state) -> GateResult:
    """Deny when the action arguments match a prompt-injection pattern.

    Args:
        action: The proposed action whose arguments are scanned.
        state: The current state (unused; present for the policy interface).

    Returns:
        A DENY result naming the matched pattern, else ALLOW.
    """
    args_text = json.dumps(action.arguments, default=str).lower()
    for pat in _PROMPT_INJECTION_PATTERNS:
        if re.search(pat, args_text):
            return GateResult(GateDecision.DENY, "prompt_injection_policy", f"matched pattern {pat!r}")
    return GateResult(GateDecision.ALLOW, "prompt_injection_policy")


def prohibited_advice_policy(action, state) -> GateResult:
    """Escalate when the action arguments match a prohibited-advice pattern.

    Args:
        action: The proposed action whose arguments are scanned.
        state: The current state (unused; present for the policy interface).

    Returns:
        An ESCALATE result naming the matched pattern, else ALLOW.
    """
    args_text = json.dumps(action.arguments, default=str).lower()
    for pat in _PROHIBITED_ADVICE_PATTERNS:
        if re.search(pat, args_text):
            return GateResult(GateDecision.ESCALATE, "prohibited_advice_policy", f"matched pattern {pat!r}")
    return GateResult(GateDecision.ALLOW, "prohibited_advice_policy")
